# -*- coding: utf-8 -*-
"""夜班批量·阶段四：失败段打补丁重跑
- 角色名替换为颜色名（消除"衬衫"字面诱导）
- 末尾追加最高优先级禁令（通体同色/严禁真实衣物/严禁真人脸）
- 上传 + 重新提交，状态文件改写为新 prompt_id
"""
import json
import subprocess
import sys
import time
from pathlib import Path

SP = Path(r"E:\sp")
ROOT = SP / "h3_debug"
STATE_PATH = ROOT / "batch_state.json"
HOST = "root@cpod-1sqfx2anig0i.podtcp.compshare.cn"
SSH = str(SP / "scripts" / "cpod_ssh.sh")
GIT_BASH = r"C:\Program Files\Git\bin\bash.exe"

REROLL = ["seg_001", "seg_002", "seg_003", "seg_005", "seg_006", "seg_007",
          "seg_008", "seg_009", "seg_011", "seg_012", "seg_015",
          "seg_020", "seg_021", "seg_023"]

NAME_MAP = [
    ("白色衬衫男（囚室）", "墨绿男人偶"),
    ("深色暗纹衬衫男", "深灰蓝男人偶"),
    ("白衣衬衫裙女", "暖陶土女人偶"),
    ("白色衬衫男", "墨绿男人偶"),
]

EXTRA_BAN = """

特别禁令（最高优先级，覆盖一切其他描述）：每个人偶的头部、面部、躯干、四肢必须是同一颜色的哑光树脂整体，全身无任何第二颜色区块；严禁生成任何真实衣物、衬衫、外套、领口、袖口、纽扣、拉链、布料纹理，人偶身上只允许出现哑光树脂；严禁保留真人皮肤、真人面孔、眉毛、睫毛、头发丝；镜子与倒影中同样只允许出现树脂人偶。"""

state = json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save():
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def out_text(r):
    return ((r.stdout or b"") + (r.stderr or b"")).decode("utf-8", "replace")


BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 280.0
t0 = time.time()

for name in REROLL:
    st = state.setdefault(name, {})
    if st.get("reroll_pid"):
        continue
    if time.time() - t0 > BUDGET:
        print("[预算用完] 下次继续", flush=True)
        break
    n = int(name[4:])
    prefix = f"s{n:02d}"
    src = ROOT / f"prompt_v4_kimi-k2_5_{prefix}_hybrid_roster.txt"
    text = src.read_text(encoding="utf-8")
    for old, new in NAME_MAP:
        text = text.replace(old, new)
    text += EXTRA_BAN
    dst = ROOT / f"prompt_v5_{prefix}.txt"
    dst.write_text(text, encoding="utf-8")

    up = subprocess.run([GIT_BASH, SSH, "-scp", f"h3_debug/prompt_v5_{prefix}.txt",
                         f"{HOST}:/root/prompt_v5_{prefix}.txt"],
                        cwd=str(SP), capture_output=True, stdin=subprocess.DEVNULL, timeout=120)
    if up.returncode != 0:
        print(f"[{name}] 上传失败: {out_text(up)[-150:]}", flush=True)
        continue
    sub = subprocess.run([GIT_BASH, SSH,
                          f"python3 /root/submit_h3_seg.py /root/prompt_v5_{prefix}.txt {name}.mp4 h3_wf_{prefix}_v5.json"],
                         cwd=str(SP), capture_output=True, stdin=subprocess.DEVNULL, timeout=120)
    try:
        pid = json.loads(out_text(sub).strip().splitlines()[-1])["prompt_id"]
    except Exception:
        print(f"[{name}] 提交失败: {out_text(sub)[-200:]}", flush=True)
        continue
    # 状态切到新 pid，清掉旧出片与分析，等待重新收片
    st["prompt_id"] = pid
    st["reroll_pid"] = pid
    st["downloaded"] = False
    for f in (ROOT / "batch_out").glob(f"outf_{prefix}_*.jpg"):
        f.unlink()
    old_mp4 = ROOT / "batch_out" / f"{name}_doll.mp4"
    if old_mp4.exists():
        old_mp4.unlink()
    ana = ROOT / "analysis" / f"{prefix}.json"
    if ana.exists():
        ana.unlink()
    save()
    print(f"[{name}] 已重提 prompt_id={pid}", flush=True)

done = sum(1 for k in REROLL if state.get(k, {}).get("reroll_pid"))
print(f"[状态] 重跑已提交 {done}/{len(REROLL)}", flush=True)
