from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .ffmpeg_tools import probe_video

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_REPO = "uva-cv-lab/OmniShotCut"
DEFAULT_CHECKPOINT_FILE = "OmniShotCut_ckpt.pth"

_model_cache: dict[tuple[str, str], Any] = {}


def detect_omnishotcut_shots(
    video_path: str | Path,
    checkpoint: str = DEFAULT_CHECKPOINT_REPO,
    filename: str = DEFAULT_CHECKPOINT_FILE,
    mode: str = "clean_shot",
    overlap: int = 20,
    min_duration: float = 0.25,
) -> dict[str, Any]:
    """Run OmniShotCut and return contiguous shot candidates in seconds.

    OmniShotCut predicts frame ranges. The pipeline needs second-based,
    contiguous ranges so later YOLO/identity layers can keep using the same
    segmentation contract.
    """
    try:
        import torch
        import omnishotcut
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OmniShotCut is not installed. Install with: "
            "python -m pip install "
            "\"git+https://github.com/UVA-Computer-Vision-Lab/OmniShotCut.git\""
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("OmniShotCut requires CUDA in the current package build, but torch.cuda is unavailable.")

    video_path = Path(video_path)
    meta = probe_video(video_path)
    fps = float(meta.fps or 0.0)
    if fps <= 0:
        raise RuntimeError("Cannot run OmniShotCut because source FPS could not be detected.")

    model = _load_model(omnishotcut, checkpoint, filename)
    logger.info("[OmniShotCut] 推理 %s (mode=%s, overlap=%d)...", video_path, mode, overlap)

    if mode == "default":
        ranges, intra_labels, inter_labels = model.inference(str(video_path), mode="default", overlap=overlap)
    else:
        ranges = model.inference(str(video_path), mode=mode, overlap=overlap)
        intra_labels = ["general"] * len(ranges)
        inter_labels = ["unknown"] * len(ranges)

    raw_shots: list[dict[str, Any]] = []
    for idx, frame_range in enumerate(ranges):
        if len(frame_range) != 2:
            continue
        start_frame = int(frame_range[0])
        end_frame = int(frame_range[1])
        if end_frame <= start_frame:
            continue

        start = max(0.0, start_frame / fps)
        end = min(float(meta.duration), end_frame / fps)
        if end - start < min_duration:
            continue

        raw_shots.append(
            {
                "index": len(raw_shots),
                "start": round(start, 3),
                "end": round(end, 3),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "intra_label": intra_labels[idx] if idx < len(intra_labels) else "",
                "inter_label": inter_labels[idx] if idx < len(inter_labels) else "",
            }
        )

    if not raw_shots:
        raise RuntimeError("OmniShotCut returned no usable shot ranges.")

    shots = _normalize_to_contiguous_shots(raw_shots, float(meta.duration), min_duration=min_duration)
    logger.info("[OmniShotCut] %d raw ranges → %d contiguous candidates", len(raw_shots), len(shots))

    return {
        "detector": "omnishotcut",
        "checkpoint": checkpoint,
        "filename": filename,
        "mode": mode,
        "fps": fps,
        "raw_shots": raw_shots,
        "shots": shots,
    }


def _load_model(omnishotcut_module: Any, checkpoint: str, filename: str) -> Any:
    cache_key = (checkpoint, filename)
    if cache_key not in _model_cache:
        logger.info("[OmniShotCut] 加载模型权重 %s/%s", checkpoint, filename)
        _model_cache[cache_key] = omnishotcut_module.load(checkpoint, filename=filename)
    return _model_cache[cache_key]


def _normalize_to_contiguous_shots(
    raw_shots: list[dict[str, Any]],
    duration: float,
    min_duration: float,
    tolerance: float = 0.05,
) -> list[dict[str, Any]]:
    boundaries = [0.0, duration]
    for shot in raw_shots:
        boundaries.append(float(shot["start"]))
        boundaries.append(float(shot["end"]))

    merged = _merge_boundaries(boundaries, duration, tolerance)
    shots: list[dict[str, Any]] = []
    for start, end in zip(merged, merged[1:]):
        if end - start < min_duration:
            continue
        shots.append(
            {
                "index": len(shots),
                "start": round(start, 3),
                "end": round(end, 3),
            }
        )
    return shots


def _merge_boundaries(boundaries: list[float], duration: float, tolerance: float) -> list[float]:
    cleaned = sorted(max(0.0, min(duration, float(value))) for value in boundaries)
    merged: list[float] = []
    for boundary in cleaned:
        if not merged or abs(boundary - merged[-1]) > tolerance:
            merged.append(boundary)
        else:
            merged[-1] = round((merged[-1] + boundary) / 2, 3)

    if not merged:
        merged.insert(0, 0.0)
    else:
        merged[0] = 0.0
    if merged[-1] < duration - tolerance:
        merged.append(duration)
    else:
        merged[-1] = duration
    return merged
