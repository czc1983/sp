# -*- coding: utf-8 -*-
"""夜班批量·阶段二：轮询 history，完成的段下载 mp4 + 服务器抽 3 张分析帧（可续跑）
用法: python batch_collect.py [预算秒数，默认280]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

SP = Path(r"E:\sp")
ROOT = SP / "h3_debug"
SEGS = {s["file"].replace(".mp4", ""): s for s in json.loads((ROOT / "segments.json").read_text(encoding="utf-8"))}
STATE_PATH = ROOT / "batch_state.json"
OUT_DIR = ROOT / "batch_out"
OUT_DIR.mkdir(exist_ok=True)
HOST = "root@cpod-1sqfx2anig0i.podtcp.compshare.cn"
SSH = str(SP / "scripts" / "cpod_ssh.sh")
GIT_BASH = r"C:\Program Files\Git\bin\bash.exe"
FF = "/root/miniconda3/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"

# seg_016 新版 + seg_010/seg_024 旧版（已在本地的跳过下载）
EXTRA = {"seg_016": "4ebc2c6c-8ae0-479e-8836-3e063599cdf8"}

state = json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save():
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def out_text(r):
    return ((r.stdout or b"") + (r.stderr or b"")).decode("utf-8", "replace")


BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 280.0
t0 = time.time()

# 收集所有 pid
pids = {}
for name, st in state.items():
    if st.get("prompt_id") and not st.get("downloaded"):
        pids[name] = st["prompt_id"]
for name, pid in EXTRA.items():
    if not state.get(name, {}).get("downloaded"):
        pids[name] = pid

if not pids:
    print("[收片] 没有待收任务", flush=True)
    sys.exit(0)

# 一次性查询所有 pid 的 history
pid_list = " ".join(pids.values())
check = subprocess.run([GIT_BASH, SSH, f"python3 /root/check_history.py {pid_list}"],
                       cwd=str(SP), capture_output=True, stdin=subprocess.DEVNULL, timeout=120)
try:
    hist = json.loads(out_text(check).strip().splitlines()[-1])
except Exception:
    print("[收片] history 查询失败:", out_text(check)[-300:], flush=True)
    sys.exit(1)

pid2name = {v: k for k, v in pids.items()}
done_now = []
for pid, info in hist.items():
    name = pid2name.get(pid)
    if not name:
        continue
    if info.get("status") == "done" and info.get("file"):
        done_now.append((name, info["file"]))
    elif info.get("status") == "missing":
        state.setdefault(name, {})["collect_note"] = "history_missing"
        save()

print(f"[收片] 本轮完成 {len(done_now)} / 待收 {len(pids)}", flush=True)

for name, remote_file in done_now:
    if time.time() - t0 > BUDGET:
        print("[预算用完] 剩余下次收", flush=True)
        break
    n = int(name[4:])
    prefix = f"s{n:02d}"
    dur = SEGS[name]["duration"]
    times = [round(dur * f, 2) for f in (0.2, 0.5, 0.8)]
    # 服务器抽分析帧 + 打包
    remote_dir = "/root/ComfyUI-H3/output"  # history 的 subfolder 是 "video"
    base = remote_file.split("/")[-1]
    sub = remote_file.rsplit("/", 1)[0] if "/" in remote_file else ""
    full = f"{remote_dir}/{sub}/{base}" if sub else f"{remote_dir}/{base}"
    cmds = [f"cd {remote_dir}/{sub}" if sub else f"cd {remote_dir}"]
    for i, t in enumerate(times, 1):
        cmds.append(f"{FF} -y -loglevel error -ss {t} -i '{base}' -frames:v 1 -q:v 2 /root/outf_{prefix}_{i}.jpg")
    cmds.append(f"tar czf /root/outpkg_{prefix}.tar.gz -C {remote_dir}{'/' + sub if sub else ''} '{base}' -C /root " +
                " ".join(f"outf_{prefix}_{i}.jpg" for i in (1, 2, 3)))
    pk = subprocess.run([GIT_BASH, SSH, " && ".join(cmds)],
                        cwd=str(SP), capture_output=True, stdin=subprocess.DEVNULL, timeout=180)
    if pk.returncode != 0:
        print(f"[{name}] 服务器打包失败: {out_text(pk)[-200:]}", flush=True)
        continue
    dl = subprocess.run([GIT_BASH, SSH, "-scp", f"{HOST}:/root/outpkg_{prefix}.tar.gz",
                         f"h3_debug/batch_out/outpkg_{prefix}.tar.gz"],
                        cwd=str(SP), capture_output=True, stdin=subprocess.DEVNULL, timeout=180)
    if dl.returncode != 0:
        print(f"[{name}] 下载失败: {out_text(dl)[-200:]}", flush=True)
        continue
    ex = subprocess.run(["tar", "xzf", f"outpkg_{prefix}.tar.gz"], cwd=str(OUT_DIR),
                        capture_output=True, timeout=60)
    if ex.returncode == 0 and (OUT_DIR / base).exists():
        (OUT_DIR / base).rename(OUT_DIR / f"{name}_doll.mp4")
        state.setdefault(name, {})["downloaded"] = True
        state[name].pop("collect_note", None)
        save()
        print(f"[{name}] 已收片 {name}_doll.mp4", flush=True)
    else:
        print(f"[{name}] 解包失败: {out_text(ex)[-200:]}", flush=True)

got = sum(1 for v in state.values() if v.get("downloaded"))
print(f"[状态] 已收片 {got}", flush=True)
