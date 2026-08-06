# -*- coding: utf-8 -*-
"""夜班批量·阶段一：27 段反推 + 提交 H3（缓存可续跑，时间预算内尽量多做）
用法: python batch_submit.py [预算秒数，默认280]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

SP = Path(r"E:\sp")
ROOT = SP / "h3_debug"
SEGS = json.loads((ROOT / "segments.json").read_text(encoding="utf-8"))
FRAME_TIMES = json.loads((ROOT / "frame_times.json").read_text(encoding="utf-8"))
STATE_PATH = ROOT / "batch_state.json"
SKIP = {"seg_010", "seg_016", "seg_024"}  # 010/024 旧片仍有效；016 已用新名册单独提交
HOST = "root@cpod-1sqfx2anig0i.podtcp.compshare.cn"
SSH = str(SP / "scripts" / "cpod_ssh.sh")
GIT_BASH = r"C:\Program Files\Git\bin\bash.exe"  # 必须用 Git Bash；裸 "bash" 会落到 WSL


def out_text(r):
    return ((r.stdout or b"") + (r.stderr or b"")).decode("utf-8", "replace")

state = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {}


def save():
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 280.0
t0 = time.time()

for s in SEGS:
    name = s["file"].replace(".mp4", "")
    if name in SKIP:
        continue
    st = state.setdefault(name, {})
    if st.get("prompt_id"):
        continue
    if time.time() - t0 > BUDGET:
        print(f"[预算用完] 停在 {name}，下次继续", flush=True)
        break
    n = int(name[4:])
    prefix = f"s{n:02d}"
    slug = f"kimi-k2_5_{prefix}_hybrid_roster"
    prompt_file = ROOT / f"prompt_v4_{slug}.txt"

    # 1) 反推（本地，带缓存可续）
    if not prompt_file.exists():
        print(f"[{name}] 反推中 ...", flush=True)
        r = subprocess.run([sys.executable, str(ROOT / "bailian_reverse.py"),
                            "kimi-k2.5", "qwen3-vl-30b-a3b-instruct",
                            ",".join(str(t) for t in FRAME_TIMES[prefix]),
                            prefix, str(ROOT / "global_roster.txt")],
                           capture_output=True, timeout=600)
        if r.returncode != 0 or not prompt_file.exists():
            st["error"] = "reverse_failed"
            st["err_tail"] = out_text(r)[-400:]
            save()
            print(f"[{name}] 反推失败，跳过", flush=True)
            continue
    # 2) 上传提示词
    up = subprocess.run([GIT_BASH, SSH, "-scp", f"h3_debug/{prompt_file.name}",
                         f"{HOST}:/root/prompt_{prefix}.txt"],
                        cwd=str(SP), capture_output=True,
                        stdin=subprocess.DEVNULL, timeout=120)
    if up.returncode != 0:
        st["error"] = "scp_failed"
        st["err_tail"] = out_text(up)[-300:]
        save()
        print(f"[{name}] 上传失败，跳过", flush=True)
        continue
    # 3) 提交 H3
    sub = subprocess.run([GIT_BASH, SSH,
                          f"python3 /root/submit_h3_seg.py /root/prompt_{prefix}.txt {name}.mp4 h3_wf_{prefix}.json"],
                         cwd=str(SP), capture_output=True,
                         stdin=subprocess.DEVNULL, timeout=120)
    try:
        pid = json.loads(out_text(sub).strip().splitlines()[-1])["prompt_id"]
        st["prompt_id"] = pid
        st.pop("error", None)
        st.pop("err_tail", None)
        print(f"[{name}] 已提交 prompt_id={pid}", flush=True)
    except Exception:
        st["error"] = "submit_failed"
        st["err_tail"] = out_text(sub)[-400:]
        print(f"[{name}] 提交失败: {st['err_tail'][:200]}", flush=True)
    save()

done = sum(1 for v in state.values() if v.get("prompt_id"))
print(f"[状态] 已提交 {done}/27", flush=True)
