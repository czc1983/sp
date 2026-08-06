#!/usr/bin/env python3
"""T0 快测：pruned 裁剪模型 + turbo LoRA + 双时钟 4 步（全量模型下载完之前的探针）。"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["H3_TURBO"] = "on"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spvideo import minimax_h3_client as m  # noqa: E402

REF_FILE = "S024_00019000_00021440_h3ref_73f_eb9c70ca57fc.mp4"
SEED = 861774310677723
OUT = Path(__file__).resolve().parent.parent / "temp" / "h3_turbo_test"
OUT.mkdir(parents=True, exist_ok=True)

client = m.MiniMaxH3Client()
wf = m.build_white_mask_workflow(
    REF_FILE, width=480, height=864, length=124, steps=20, seed=SEED,
    filename_prefix="h3_turbo_t0_pruned",
)
wf[m.NODE_UNET]["inputs"]["unet_name"] = m.PRUNED_UNET_NAME  # 强行用回裁剪版
print(">>> 提交 T0: pruned + LoRA + 双时钟4步")
t0 = time.time()
try:
    prompt_id, history = client.comfy.run_workflow(wf, log=lambda s: print("  ", s))
except Exception as e:
    print(f">>> T0 失败: {e}")
    sys.exit(1)
elapsed = time.time() - t0
print(f">>> T0 完成，耗时 {elapsed:.1f}s")
item = history.get(prompt_id, {})
asset = m.ComfyClient.first_output_asset(item.get("outputs", {}).get(m.NODE_SAVE_VIDEO))
target = OUT / "t0_pruned.mp4"
client.comfy.download_output_asset(asset, target)
print(f">>> 视频: {target}")
import subprocess
FFMPEG = str(Path(__file__).resolve().parent.parent / "assets" / "ffmpeg-full" / "ffmpeg.exe")
for i, pct in enumerate((0.05, 0.5, 0.95)):
    subprocess.run([FFMPEG, "-y", "-ss", str(pct * 5.1), "-i", str(target), "-frames:v", "1",
                    str(OUT / f"t0_f{i}.png")], capture_output=True)
print(">>> 抽帧完成，人工看 t0_f*.png 判断 LoRA 是否生效")
