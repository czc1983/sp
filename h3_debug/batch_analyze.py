# -*- coding: utf-8 -*-
"""夜班批量·阶段三：自动分析每段人偶版出片（原片帧 vs 出片帧对照，VLM 打分）
用法: python batch_analyze.py [预算秒数，默认280]
输出: h3_debug/analysis/sNN.json + analysis/summary.json
"""
import base64
import json
import sys
import time
from pathlib import Path

import requests

SP = Path(r"E:\sp")
ROOT = SP / "h3_debug"
ANA = ROOT / "analysis"
ANA.mkdir(exist_ok=True)
SEGS = json.loads((ROOT / "segments.json").read_text(encoding="utf-8"))
FRAME_TIMES = json.loads((ROOT / "frame_times.json").read_text(encoding="utf-8"))
ROSTER = (ROOT / "global_roster.txt").read_text(encoding="utf-8")
MODEL = "qwen3-vl-30b-a3b-instruct"

cfg = json.loads((SP / ".dub_config" / "settings.json").read_text(encoding="utf-8"))["foreign_dub"]
BASE_URL = cfg["asr"]["base_url"].rstrip("/")
API_KEY = cfg["asr"]["api_key"]

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 280.0
t0 = time.time()


def img(path):
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


done_count = 0
for s in SEGS:
    name = s["file"].replace(".mp4", "")
    n = int(name[4:])
    prefix = f"s{n:02d}"
    out_json = ANA / f"{prefix}.json"
    if out_json.exists():
        done_count += 1
        continue
    # 出片帧
    out_frames = sorted((ROOT / "batch_out").glob(f"outf_{prefix}_*.jpg"))
    if prefix == "s10":
        out_frames = sorted(ROOT.glob("pf_00026_*.jpg"))
    elif prefix == "s24":
        out_frames = sorted(ROOT.glob("pf_00028_*.jpg"))
    if not out_frames:
        continue  # 还没收片
    # 原片帧：取首/中/尾 3 张
    times = FRAME_TIMES[prefix]
    picks = [times[0], times[len(times) // 2], times[-1]]
    ref_frames = [ROOT / f"{prefix}_{t}.jpg" for t in picks]
    if not all(p.exists() for p in ref_frames):
        continue
    if time.time() - t0 > BUDGET:
        print("[预算用完] 下次继续", flush=True)
        break

    content = []
    for p in ref_frames:
        content.append(img(p))
    content.append({"type": "text", "text": "↑ 参考视频（真人原片）抽帧"})
    for p in out_frames:
        content.append(img(p))
    content.append({"type": "text", "text": "↑ 人偶版出片抽帧"})
    prompt = f"""全片人物名册与配色规定：
{ROSTER}

上面先给了参考视频（真人原片）的 3 张抽帧，后给了"彩色树脂关节人偶版"出片的 {len(out_frames)} 张抽帧。请逐条核对（只看画面事实，拿不准写 uncertain）：
1. person_count：人偶版可见人偶数量是否与参考帧真实人数一致（镜中倒影不算独立人偶）
2. color：每个人偶颜色是否符合名册配色（哑光深灰蓝/哑光暖陶土/哑光墨绿），同人偶各帧颜色是否一致
3. mirror：参考帧若有镜子/倒影，人偶版是否保留倒影且与本体同色；参考帧无镜子则写 na
4. composition：构图、人物位置、姿态、遮挡关系是否与参考帧对应
5. material：是否为哑光树脂人偶素体（应无真实皮肤、头发丝、睫毛、布料质感）
6. no_text：人偶版画面是否无文字/字幕/水印残留

只输出一个 JSON 对象，不要任何其他文字：
{{"person_count":"ok|bad|uncertain","color":"ok|bad|uncertain","mirror":"ok|bad|na|uncertain","composition":"ok|bad|uncertain","material":"ok|bad|uncertain","no_text":"ok|bad|uncertain","verdict":"pass|issues|fail","notes":"不超过60字的问题说明，全对写全对"}}"""
    content.append({"type": "text", "text": prompt})

    payload = {"model": MODEL, "messages": [{"role": "user", "content": content}],
               "max_tokens": 800, "temperature": 0.1, "vl_high_resolution_images": True}
    try:
        r = requests.post(f"{BASE_URL}/chat/completions",
                          headers={"Authorization": f"Bearer {API_KEY}"},
                          json=payload, timeout=300)
        d = r.json()
        text = d["choices"][0]["message"]["content"].strip()
        # 提取 JSON
        i, j = text.find("{"), text.rfind("}")
        verdict = json.loads(text[i:j + 1])
    except Exception as e:
        verdict = {"verdict": "error", "notes": str(e)[:150]}
    verdict["_seg"] = name
    verdict["_duration"] = s["duration"]
    out_json.write_text(json.dumps(verdict, ensure_ascii=False, indent=1), encoding="utf-8")
    done_count += 1
    print(f"[{name}] {verdict.get('verdict')} | {verdict.get('notes','')}", flush=True)

# 汇总
rows = []
for f in sorted(ANA.glob("s*.json")):
    rows.append(json.loads(f.read_text(encoding="utf-8")))
(ANA / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[状态] 已分析 {done_count}/30", flush=True)
