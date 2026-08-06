# -*- coding: utf-8 -*-
"""镜头检测 + 按 H3 上限切段（保留音频），输出 segments.json 与缩略图"""
import json
import subprocess
from pathlib import Path

import av
import numpy as np
import imageio_ffmpeg

SRC = "/root/full_drama.mp4"
OUT_DIR = Path("/root/segments")
OUT_DIR.mkdir(exist_ok=True)
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

SAMPLE_FPS = 5          # 检测采样率
MAX_SEG = 4.8           # H3 单段上限
MIN_SHOT = 1.0          # 短于此的镜头并入相邻段
MIN_SEG = 1.5           # 短于此的段并入前一段

# ---------- 1. 帧差检测镜头边界 ----------
print("decoding for scene detection...", flush=True)
container = av.open(SRC)
prev = None
diffs = []       # (time, diff)
for frame in container.decode(video=0):
    t = float(frame.pts * frame.time_base) if frame.pts is not None else 0.0
    if len(diffs) == 0 or t - diffs[-1][0] >= 1.0 / SAMPLE_FPS - 1e-6:
        img = frame.to_ndarray(format="gray", width=96, height=170).astype(np.int16)
        if prev is not None:
            diffs.append((t, float(np.abs(img - prev).mean())))
        prev = img
container.close()

vals = np.array([d for _, d in diffs])
thr = max(vals.mean() + 3 * vals.std(), np.percentile(vals, 95))
cuts = [0.0] + [t for t, d in diffs if d > thr]
dur = float(av.open(SRC).duration) / 1e6
cuts.append(dur)
print(f"duration {dur:.2f}s, diff mean {vals.mean():.1f} std {vals.std():.1f}, threshold {thr:.1f}, raw cuts {len(cuts)-2}", flush=True)

# ---------- 2. 合并过短镜头 ----------
shots = [[cuts[i], cuts[i + 1]] for i in range(len(cuts) - 1)]
merged = []
for s in shots:
    if merged and (s[1] - s[0]) < MIN_SHOT:
        merged[-1][1] = s[1]
    else:
        merged.append(list(s))
shots = merged

# ---------- 3. 打包成 ≤4.8s 的段 ----------
segments = []
cur = None
for s in shots:
    if cur is None:
        cur = list(s)
        continue
    if s[1] - cur[0] <= MAX_SEG:
        cur[1] = s[1]
    else:
        segments.append(cur)
        cur = list(s)
if cur:
    segments.append(cur)
# 超长镜头硬切
final = []
for seg in segments:
    while seg[1] - seg[0] > MAX_SEG:
        final.append([seg[0], seg[0] + MAX_SEG])
        seg = [seg[0] + MAX_SEG, seg[1]]
    final.append(seg)
# 过短尾段并入前段（合并后不超过 5.2s 才并；先剔除零长段）
packed = []
for seg in final:
    if seg[1] - seg[0] < 0.3:
        continue
    if packed and (seg[1] - seg[0]) < MIN_SEG and (seg[1] - packed[-1][0]) <= 5.25:
        packed[-1][1] = seg[1]
    else:
        packed.append(seg)

print(f"shots merged: {len(shots)}, segments: {len(packed)}", flush=True)
for i, (a, b) in enumerate(packed):
    print(f"  seg_{i+1:03d}: {a:6.2f} - {b:6.2f}  ({b-a:.2f}s)", flush=True)

# ---------- 4. ffmpeg 切段（重编码，含音频） ----------
meta = []
for i, (a, b) in enumerate(packed):
    name = f"seg_{i+1:03d}.mp4"
    out = OUT_DIR / name
    cmd = [FFMPEG, "-y", "-ss", f"{a:.3f}", "-to", f"{b:.3f}", "-i", SRC,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-c:a", "aac", "-b:a", "128k", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"CUT FAIL {name}: {r.stderr[-300:]}", flush=True)
        continue
    meta.append({"file": name, "start": round(a, 2), "end": round(b, 2), "duration": round(b - a, 2)})
    print(f"cut {name}", flush=True)

(OUT_DIR / "segments.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

# ---------- 5. 每段中点缩略图 ----------
for m in meta:
    mid = m["start"] + m["duration"] / 2
    subprocess.run([FFMPEG, "-y", "-ss", f"{mid:.3f}", "-i", SRC, "-frames:v", "1",
                    "-vf", "scale=144:256", str(OUT_DIR / f"thumb_{m['file'][:-4]}.jpg")],
                   capture_output=True)
print("ALL DONE", flush=True)
