from __future__ import annotations

import bisect
import json
from pathlib import Path


def composite_protected_region(
    source_video: str | Path,
    generated_video: str | Path,
    track_dir: str | Path,
    output_path: str | Path,
    *,
    feather_pixels: int = 7,
    dilate_pixels: int = 2,
) -> dict[str, object]:
    """Restore tracked source pixels over a generated video."""
    import cv2
    import numpy as np

    source_path = Path(source_video)
    generated_path = Path(generated_video)
    track_path = Path(track_dir)
    destination = Path(output_path)
    summary_path = track_path / "track_summary.json"

    if not source_path.exists():
        raise ValueError(f"source_video_not_found: {source_path}")
    if not generated_path.exists():
        raise ValueError(f"generated_video_not_found: {generated_path}")
    if not summary_path.exists():
        raise ValueError(f"protection_track_not_found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    clip_start_frame = int(summary.get("clip_start_frame") or 0)
    track_fps = float(summary.get("track_fps") or 0)
    clip_start_time = float(summary.get("clip_start_time") or 0)
    mask_paths = _index_masks(track_path)
    if not mask_paths:
        raise ValueError("protection_masks_not_found")
    mask_indices = sorted(mask_paths)

    source = cv2.VideoCapture(str(source_path))
    generated = cv2.VideoCapture(str(generated_path))
    if not source.isOpened() or not generated.isOpened():
        source.release()
        generated.release()
        raise ValueError("cannot_open_protection_videos")

    source_fps = float(source.get(cv2.CAP_PROP_FPS) or 0) or 24.0
    generated_fps = float(generated.get(cv2.CAP_PROP_FPS) or 0) or source_fps
    width = int(generated.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(generated.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        source.release()
        generated.release()
        raise ValueError("invalid_generated_video_size")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_video = destination.parent / f".{destination.stem}.video.mp4"
    writer = cv2.VideoWriter(
        str(temp_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        generated_fps,
        (width, height),
    )
    if not writer.isOpened():
        source.release()
        generated.release()
        raise ValueError("cannot_create_protected_video")

    total_frames = 0
    protected_frames = 0
    skipped_large_masks = 0
    try:
        while True:
            ok_generated, generated_frame = generated.read()
            if not ok_generated or generated_frame is None:
                break

            time_seconds = total_frames / generated_fps
            source_frame_idx = max(0, int(round(time_seconds * source_fps)))
            source.set(cv2.CAP_PROP_POS_FRAMES, source_frame_idx)
            ok_source, source_frame = source.read()
            output_frame = generated_frame

            if track_fps > 0:
                track_frame_idx = int(round((time_seconds - clip_start_time) * track_fps))
            else:
                track_frame_idx = source_frame_idx - clip_start_frame
            mask_path = _nearest_mask(mask_paths, mask_indices, track_frame_idx)
            if ok_source and source_frame is not None and mask_path is not None:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
                    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    area_ratio = float(np.count_nonzero(mask)) / float(width * height)
                    if area_ratio <= 0.95:
                        if dilate_pixels > 0:
                            kernel_size = dilate_pixels * 2 + 1
                            kernel = cv2.getStructuringElement(
                                cv2.MORPH_ELLIPSE,
                                (kernel_size, kernel_size),
                            )
                            mask = cv2.dilate(mask, kernel, iterations=1)
                        blur_size = max(1, int(feather_pixels))
                        if blur_size % 2 == 0:
                            blur_size += 1
                        alpha = cv2.GaussianBlur(mask, (blur_size, blur_size), 0).astype(np.float32) / 255.0
                        alpha = alpha[..., None]
                        source_frame = cv2.resize(source_frame, (width, height), interpolation=cv2.INTER_LINEAR)
                        output_frame = np.clip(
                            source_frame.astype(np.float32) * alpha
                            + generated_frame.astype(np.float32) * (1.0 - alpha),
                            0,
                            255,
                        ).astype(np.uint8)
                        protected_frames += 1
                    else:
                        skipped_large_masks += 1

            writer.write(output_frame)
            total_frames += 1
    finally:
        writer.release()
        source.release()
        generated.release()

    if total_frames <= 0:
        temp_video.unlink(missing_ok=True)
        raise ValueError("empty_generated_video")
    if protected_frames <= 0:
        temp_video.unlink(missing_ok=True)
        raise ValueError("no_protection_frames_composited")

    from .ffmpeg_tools import ffmpeg_path, run_command

    try:
        run_command([
            ffmpeg_path(),
            "-y",
            "-loglevel", "error",
            "-i", str(temp_video),
            "-i", str(generated_path),
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-c:a", "copy",
            "-movflags", "+faststart",
            "-shortest",
            str(destination),
        ])
    finally:
        temp_video.unlink(missing_ok=True)

    return {
        "output_path": str(destination),
        "total_frames": total_frames,
        "protected_frames": protected_frames,
        "skipped_large_masks": skipped_large_masks,
        "source_fps": source_fps,
        "generated_fps": generated_fps,
    }


def _index_masks(track_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in track_dir.glob("mask_*.png"):
        try:
            result[int(path.stem.rsplit("_", 1)[-1])] = path
        except ValueError:
            continue
    return result


def _nearest_mask(
    masks: dict[int, Path],
    indices: list[int],
    target: int,
    tolerance: int = 2,
) -> Path | None:
    direct = masks.get(target)
    if direct is not None:
        return direct
    position = bisect.bisect_left(indices, target)
    candidates = indices[max(0, position - 1): position + 1]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda value: abs(value - target))
    return masks[nearest] if abs(nearest - target) <= tolerance else None
