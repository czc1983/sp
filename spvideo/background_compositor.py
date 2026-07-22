from __future__ import annotations

import bisect
import json
from pathlib import Path


def compose_background(
    foreground_video: str | Path,
    background_asset: str | Path,
    track_dirs: list[str | Path],
    output_path: str | Path,
    *,
    fit_mode: str = "cover",
    feather_pixels: int = 9,
    dilate_pixels: int = 6,
    preserve_audio: bool = True,
) -> dict[str, object]:
    """Replace the background after person transfer using tracked foreground masks."""
    import cv2
    import numpy as np

    foreground_path = Path(foreground_video)
    background_path = Path(background_asset)
    destination = Path(output_path)
    if not foreground_path.exists():
        raise ValueError(f"foreground_video_not_found: {foreground_path}")
    if not background_path.exists():
        raise ValueError(f"background_asset_not_found: {background_path}")
    if fit_mode not in {"cover", "contain", "stretch"}:
        raise ValueError("invalid_background_fit_mode")

    tracks = [_load_track(Path(path)) for path in track_dirs]
    if not tracks:
        raise ValueError("background_foreground_track_not_ready")

    foreground = cv2.VideoCapture(str(foreground_path))
    if not foreground.isOpened():
        raise ValueError("cannot_open_foreground_video")
    fps = float(foreground.get(cv2.CAP_PROP_FPS) or 0) or 24.0
    width = int(foreground.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(foreground.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        foreground.release()
        raise ValueError("invalid_foreground_video_size")

    image_background = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
    background_video = None
    if image_background is None:
        background_video = cv2.VideoCapture(str(background_path))
        if not background_video.isOpened():
            foreground.release()
            raise ValueError("cannot_open_background_asset")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_video = destination.parent / f".{destination.stem}.video.mp4"
    writer = cv2.VideoWriter(
        str(temp_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        foreground.release()
        if background_video:
            background_video.release()
        raise ValueError("cannot_create_background_video")

    total_frames = 0
    composited_frames = 0
    try:
        while True:
            ok, frame = foreground.read()
            if not ok or frame is None:
                break
            if image_background is not None:
                bg_frame = image_background
            else:
                ok_bg, bg_frame = background_video.read()
                if not ok_bg or bg_frame is None:
                    background_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok_bg, bg_frame = background_video.read()
                if not ok_bg or bg_frame is None:
                    raise ValueError("empty_background_video")
            bg_frame = _fit_frame(bg_frame, width, height, fit_mode)

            mask = np.zeros((height, width), dtype=np.uint8)
            time_seconds = total_frames / fps
            for track in tracks:
                mask_path = _mask_at_time(track, time_seconds)
                if mask_path is None:
                    continue
                current = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if current is None:
                    continue
                current = cv2.resize(current, (width, height), interpolation=cv2.INTER_LINEAR)
                _, current = cv2.threshold(current, 127, 255, cv2.THRESH_BINARY)
                mask = cv2.bitwise_or(mask, current)

            if np.count_nonzero(mask) > 0:
                if dilate_pixels > 0:
                    size = dilate_pixels * 2 + 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
                    mask = cv2.dilate(mask, kernel, iterations=1)
                blur_size = max(1, int(feather_pixels))
                if blur_size % 2 == 0:
                    blur_size += 1
                alpha = cv2.GaussianBlur(mask, (blur_size, blur_size), 0).astype(np.float32) / 255.0
                alpha = alpha[..., None]
                frame = np.clip(
                    frame.astype(np.float32) * alpha + bg_frame.astype(np.float32) * (1.0 - alpha),
                    0,
                    255,
                ).astype(np.uint8)
                composited_frames += 1
            writer.write(frame)
            total_frames += 1
    finally:
        writer.release()
        foreground.release()
        if background_video:
            background_video.release()

    if total_frames <= 0:
        temp_video.unlink(missing_ok=True)
        raise ValueError("empty_foreground_video")
    if composited_frames <= 0:
        temp_video.unlink(missing_ok=True)
        raise ValueError("no_foreground_masks_composited")

    if preserve_audio:
        from .ffmpeg_tools import ffmpeg_path, run_command

        try:
            run_command([
                ffmpeg_path(), "-y", "-loglevel", "error",
                "-i", str(temp_video), "-i", str(foreground_path),
                "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-c:a", "copy", "-movflags", "+faststart", "-shortest",
                str(destination),
            ])
        finally:
            temp_video.unlink(missing_ok=True)
    else:
        temp_video.replace(destination)

    return {
        "output_path": str(destination),
        "total_frames": total_frames,
        "composited_frames": composited_frames,
        "foreground_tracks": len(tracks),
        "fps": fps,
    }


def _load_track(track_dir: Path) -> dict[str, object]:
    summary_path = track_dir / "track_summary.json"
    if not summary_path.exists():
        raise ValueError(f"background_track_not_found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    masks: dict[int, Path] = {}
    for path in track_dir.glob("mask_*.png"):
        try:
            masks[int(path.stem.rsplit("_", 1)[-1])] = path
        except ValueError:
            continue
    if not masks:
        raise ValueError(f"background_masks_not_found: {track_dir}")
    return {"summary": summary, "masks": masks, "indices": sorted(masks)}


def _mask_at_time(track: dict[str, object], time_seconds: float) -> Path | None:
    summary = track["summary"]
    masks = track["masks"]
    indices = track["indices"]
    track_fps = float(summary.get("track_fps") or 0) or 24.0
    clip_start_time = float(summary.get("clip_start_time") or 0)
    target = int(round((time_seconds - clip_start_time) * track_fps))
    direct = masks.get(target)
    if direct is not None:
        return direct
    position = bisect.bisect_left(indices, target)
    candidates = indices[max(0, position - 1): position + 1]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda value: abs(value - target))
    return masks[nearest] if abs(nearest - target) <= 2 else None


def _fit_frame(frame, width: int, height: int, fit_mode: str):
    import cv2
    import numpy as np

    if fit_mode == "stretch":
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    source_height, source_width = frame.shape[:2]
    if source_width <= 0 or source_height <= 0:
        raise ValueError("invalid_background_frame_size")
    scale = max(width / source_width, height / source_height) if fit_mode == "cover" else min(
        width / source_width, height / source_height
    )
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    if fit_mode == "cover":
        left = max(0, (resized_width - width) // 2)
        top = max(0, (resized_height - height) // 2)
        return resized[top:top + height, left:left + width]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return canvas
