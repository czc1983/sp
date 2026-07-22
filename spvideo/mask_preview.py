from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_mask_comparison_video(
    video_path: str | Path,
    track_dir: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Render original and SAM mask overlay side by side for visual inspection."""
    import cv2
    import numpy as np

    source = Path(video_path)
    track = Path(track_dir)
    destination = Path(output_path)
    summary_path = track / "track_summary.json"
    if not source.exists():
        raise ValueError(f"mask_preview_video_not_found: {source}")
    if not summary_path.exists():
        raise ValueError(f"mask_preview_summary_not_found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    track_fps = float(summary.get("track_fps") or 0.0)
    clip_start_time = float(summary.get("clip_start_time") or 0.0)
    masks = _mask_index(track)
    if not masks:
        raise ValueError(f"mask_preview_masks_not_found: {track}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"mask_preview_cannot_open_video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or track_fps or 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("mask_preview_invalid_video_size")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.parent / f".{destination.stem}.mp4v.mp4"
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"mask_preview_cannot_create_video: {temp_path}")

    total_frames = 0
    masked_frames = 0
    area_ratios: list[float] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            time_seconds = total_frames / fps
            mask_index = int(round((time_seconds - clip_start_time) * (track_fps or fps)))
            mask_path = masks.get(mask_index)
            overlay = frame.copy()
            if mask_path is not None:
                mask = _read_grayscale_image(mask_path)
                if mask is not None:
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    selected = mask > 0
                    if np.any(selected):
                        tint = np.empty_like(frame)
                        tint[:] = (40, 220, 70)
                        blended = cv2.addWeighted(frame, 0.55, tint, 0.45, 0)
                        overlay[selected] = blended[selected]
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2, cv2.LINE_AA)
                        area_ratios.append(float(np.count_nonzero(selected)) / float(width * height))
                        masked_frames += 1
            writer.write(np.concatenate([frame, overlay], axis=1))
            total_frames += 1
    finally:
        capture.release()
        writer.release()

    if total_frames <= 0:
        temp_path.unlink(missing_ok=True)
        raise ValueError("mask_preview_empty_video")

    from .ffmpeg_tools import ffmpeg_path, run_command

    try:
        run_command([
            ffmpeg_path(), "-y", "-loglevel", "error",
            "-i", str(temp_path),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(destination),
        ])
    finally:
        temp_path.unlink(missing_ok=True)

    return {
        "output_path": str(destination),
        "total_frames": total_frames,
        "masked_frames": masked_frames,
        "mask_coverage": masked_frames / total_frames,
        "mean_area_ratio": sum(area_ratios) / len(area_ratios) if area_ratios else 0.0,
        "min_area_ratio": min(area_ratios) if area_ratios else 0.0,
        "max_area_ratio": max(area_ratios) if area_ratios else 0.0,
        "width": width * 2,
        "height": height,
        "fps": fps,
    }


def render_multi_mask_comparison_video(
    video_path: str | Path,
    tracks: list[dict[str, Any]],
    output_path: str | Path,
) -> dict[str, Any]:
    """Render multiple role masks together and measure cross-role overlap."""
    import cv2
    import numpy as np

    source = Path(video_path)
    destination = Path(output_path)
    if not source.exists():
        raise ValueError(f"mask_preview_video_not_found: {source}")
    if not tracks:
        raise ValueError("mask_preview_tracks_required")

    loaded_tracks: list[dict[str, Any]] = []
    default_colors = [(255, 80, 40), (40, 40, 255), (40, 220, 70)]
    for index, item in enumerate(tracks):
        track_dir = Path(str(item.get("track_dir") or ""))
        summary_path = track_dir / "track_summary.json"
        if not summary_path.exists():
            raise ValueError(f"mask_preview_summary_not_found: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        masks = _mask_index(track_dir)
        if not masks:
            raise ValueError(f"mask_preview_masks_not_found: {track_dir}")
        color = item.get("color_bgr") or default_colors[index % len(default_colors)]
        loaded_tracks.append({
            "name": str(item.get("name") or track_dir.name),
            "track_dir": str(track_dir),
            "track_fps": float(summary.get("track_fps") or 0.0),
            "clip_start_time": float(summary.get("clip_start_time") or 0.0),
            "masks": masks,
            "color": tuple(int(value) for value in color),
            "masked_frames": 0,
            "area_ratios": [],
        })

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"mask_preview_cannot_open_video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError("mask_preview_invalid_video_size")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.parent / f".{destination.stem}.mp4v.mp4"
    writer = cv2.VideoWriter(
        str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"mask_preview_cannot_create_video: {temp_path}")

    total_frames = 0
    overlap_frames = 0
    overlap_ratios: list[float] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            time_seconds = total_frames / fps
            overlay = frame.copy()
            frame_masks: list[Any] = []
            for item in loaded_tracks:
                track_fps = float(item["track_fps"] or fps)
                mask_index = int(round((time_seconds - float(item["clip_start_time"])) * track_fps))
                mask_path = item["masks"].get(mask_index)
                mask = None
                if mask_path is not None:
                    mask = _read_grayscale_image(mask_path)
                if mask is None:
                    frame_masks.append(np.zeros((height, width), dtype=bool))
                    continue
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                selected = mask > 0
                frame_masks.append(selected)
                if not np.any(selected):
                    continue
                tint = np.empty_like(frame)
                tint[:] = item["color"]
                blended = cv2.addWeighted(frame, 0.55, tint, 0.45, 0)
                overlay[selected] = blended[selected]
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, item["color"], 2, cv2.LINE_AA)
                item["masked_frames"] += 1
                item["area_ratios"].append(float(np.count_nonzero(selected)) / float(width * height))

            overlap_count = np.sum(np.stack(frame_masks, axis=0), axis=0) if frame_masks else None
            if overlap_count is not None:
                overlapping = overlap_count > 1
                if np.any(overlapping):
                    overlap_frames += 1
                    overlap_ratios.append(float(np.count_nonzero(overlapping)) / float(width * height))
                    overlap_mask = (overlapping.astype(np.uint8) * 255)
                    overlay[overlapping] = (0, 255, 255)
                    contours, _ = cv2.findContours(
                        overlap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                    )
                    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2, cv2.LINE_AA)

            writer.write(np.concatenate([frame, overlay], axis=1))
            total_frames += 1
    finally:
        capture.release()
        writer.release()

    if total_frames <= 0:
        temp_path.unlink(missing_ok=True)
        raise ValueError("mask_preview_empty_video")

    from .ffmpeg_tools import ffmpeg_path, run_command

    try:
        run_command([
            ffmpeg_path(), "-y", "-loglevel", "error", "-i", str(temp_path),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
        ])
    finally:
        temp_path.unlink(missing_ok=True)

    role_stats = []
    for item in loaded_tracks:
        area_ratios = item["area_ratios"]
        role_stats.append({
            "name": item["name"],
            "track_dir": item["track_dir"],
            "masked_frames": item["masked_frames"],
            "mask_coverage": item["masked_frames"] / total_frames,
            "mean_area_ratio": sum(area_ratios) / len(area_ratios) if area_ratios else 0.0,
            "min_area_ratio": min(area_ratios) if area_ratios else 0.0,
            "max_area_ratio": max(area_ratios) if area_ratios else 0.0,
        })
    return {
        "output_path": str(destination),
        "total_frames": total_frames,
        "roles": role_stats,
        "overlap_frames": overlap_frames,
        "overlap_frame_ratio": overlap_frames / total_frames,
        "mean_overlap_area_ratio": (
            sum(overlap_ratios) / len(overlap_ratios) if overlap_ratios else 0.0
        ),
        "max_overlap_area_ratio": max(overlap_ratios) if overlap_ratios else 0.0,
        "width": width * 2,
        "height": height,
        "fps": fps,
    }


def _mask_index(track_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in track_dir.glob("mask_*.png"):
        try:
            result[int(path.stem.rsplit("_", 1)[-1])] = path
        except ValueError:
            continue
    return result


def _read_grayscale_image(path: Path):
    """Read masks through bytes so OpenCV handles Windows unicode paths."""
    import cv2
    import numpy as np

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size <= 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
