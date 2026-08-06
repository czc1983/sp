#!/usr/bin/env python3
"""T1 turbo 4 步实测：测速 + 抽帧 + 爆音（白噪声）音频分析。

用法: python scripts/test_h3_turbo.py [steps]
复刻基准测试配置: S024 73f 参考片 → 124 帧 480×864，同种子 861774310677723。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["H3_TURBO"] = "on"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spvideo import minimax_h3_client as m  # noqa: E402

REF_FILE = "S024_00019000_00021440_h3ref_73f_eb9c70ca57fc.mp4"
SEED = 861774310677723
OUT = Path(__file__).resolve().parent.parent / "temp" / "h3_turbo_test"
FFMPEG = str(Path(__file__).resolve().parent.parent / "assets" / "ffmpeg-full" / "ffmpeg.exe")


def analyze_audio(video: Path) -> dict:
    """爆音=白噪声特征: 峰值削波 + 高谱平坦度。返回统计 dict。"""
    wav = OUT / "audio.wav"
    subprocess.run(
        [FFMPEG, "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(wav)],
        capture_output=True, check=True,
    )
    import numpy as np
    import wave

    with wave.open(str(wav), "rb") as w:
        n = w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float64) / 32768.0
    if len(data) == 0:
        return {"error": "empty audio"}
    rms = float(np.sqrt(np.mean(data**2)))
    peak = float(np.max(np.abs(data)))
    clip_ratio = float(np.mean(np.abs(data) > 0.98))
    # 谱平坦度（白噪声→接近1，语音/音乐→0.01~0.3）
    seg = data[: 16000 * min(4, len(data) // 16000)]
    if len(seg) >= 4096:
        frames = seg[: len(seg) // 1024 * 1024].reshape(-1, 1024)
        spec = np.abs(np.fft.rfft(frames * np.hanning(1024), axis=1)) + 1e-12
        flat = float(np.mean(np.exp(np.mean(np.log(spec), axis=1)) / np.mean(spec, axis=1)))
    else:
        flat = -1.0
    return {"rms": round(rms, 4), "peak": round(peak, 4), "clip_ratio": round(clip_ratio, 5), "spectral_flatness": round(flat, 4)}


def main() -> None:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    OUT.mkdir(parents=True, exist_ok=True)
    m.TURBO_STEPS = steps  # 4 或 8（8 步为防暴音退路）

    client = m.MiniMaxH3Client()
    wf = m.build_white_mask_workflow(
        REF_FILE, width=480, height=864, length=124, steps=20, seed=SEED,
        filename_prefix=f"h3_turbo_t1_{steps}step",
    )
    if steps != 4:  # 8 步退路: 双时钟 steps 调高即可，其余不变
        wf[m.NODE_DUALCLOCK]["inputs"]["steps"] = steps
    (OUT / f"t1_{steps}step_workflow.json").write_text(json.dumps(wf, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f">>> 提交 T1 turbo {steps} 步到 {client.base_url}")
    t0 = time.time()
    prompt_id, history = client.comfy.run_workflow(wf, log=lambda s: print("  ", s))
    elapsed = time.time() - t0
    print(f">>> 完成，耗时 {elapsed:.1f}s（基准 131.6s / BlockCache 72.3s）")

    item = history.get(prompt_id, {})
    node_out = item.get("outputs", {}).get(m.NODE_SAVE_VIDEO)
    asset = m.ComfyClient.first_output_asset(node_out)
    target = OUT / f"t1_{steps}step.mp4"
    client.comfy.download_output_asset(asset, target)
    print(f">>> 视频已下载: {target}")

    # 抽 4 帧看画质
    for i, pct in enumerate((0.05, 0.35, 0.65, 0.95)):
        subprocess.run(
            [FFMPEG, "-y", "-ss", str(pct * 5.1), "-i", str(target), "-frames:v", "1",
             str(OUT / f"t1_{steps}step_f{i}.png")],
            capture_output=True,
        )
    stats = analyze_audio(target)
    print(f">>> 音频分析: {stats}")
    verdict = "疑似爆音/白噪声" if (stats.get("spectral_flatness", 0) > 0.5 or stats.get("clip_ratio", 0) > 0.01) else "音频正常"
    print(f">>> 音频判定: {verdict}")
    print(json.dumps({"elapsed_s": round(elapsed, 1), "audio": stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
