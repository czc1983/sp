# -*- coding: utf-8 -*-
"""服务器端：为全部 30 段各抽 6 帧（段内相对时间），产出 /root/seg_frames/ 与 frame_times.json"""
import json
import subprocess
from pathlib import Path

FF = "/root/miniconda3/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
segs = json.load(open("/root/segments/segments.json"))
out = Path("/root/seg_frames")
out.mkdir(exist_ok=True)
frame_times = {}
for s in segs:
    n = int(s["file"][4:7])  # seg_001 -> 1
    prefix = f"s{n:02d}"
    dur = s["duration"]
    lo, hi = 0.15, max(0.2, dur - 0.15)
    times = [round(lo + i * (hi - lo) / 5, 2) for i in range(6)]
    frame_times[prefix] = times
    for t in times:
        dst = out / f"{prefix}_{t}.jpg"
        subprocess.run([FF, "-y", "-loglevel", "error", "-ss", str(t),
                        "-i", f"/root/segments/{s['file']}",
                        "-frames:v", "1", "-q:v", "2", str(dst)], check=True)
json.dump(frame_times, open("/root/seg_frames/frame_times.json", "w"), indent=1)
print("frames:", len(list(out.glob("*.jpg"))))
