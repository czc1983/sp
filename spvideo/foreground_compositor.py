from __future__ import annotations

import bisect
import json
from pathlib import Path


def composite_generated_foreground(
    base_video: str | Path,
    generated_video: str | Path,
    track_dir: str | Path,
    output_path: str | Path,
    *,
    feather_pixels: int = 7,
    dilate_pixels: int = 2,
) -> dict[str, object]:
    """Replace one tracked foreground in ``base_video`` with generated pixels."""
    import cv2
    import numpy as np

    base_path = Path(base_video)
    generated_path = Path(generated_video)
    track_path = Path(track_dir)
    destination = Path(output_path)
    summary_path = track_path / "track_summary.json"
    if not base_path.exists() or not generated_path.exists():
        raise ValueError("foreground_composite_video_not_found")
    if not summary_path.exists():
        raise ValueError(f"foreground_track_not_found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    track_fps = float(summary.get("track_fps") or 0)
    clip_start_time = float(summary.get("clip_start_time") or 0)
    masks = _index_masks(track_path)
    indices = sorted(masks)
    if not masks:
        raise ValueError("foreground_masks_not_found")

    base = cv2.VideoCapture(str(base_path))
    generated = cv2.VideoCapture(str(generated_path))
    if not base.isOpened() or not generated.isOpened():
        base.release()
        generated.release()
        raise ValueError("cannot_open_foreground_videos")
    fps = float(base.get(cv2.CAP_PROP_FPS) or 0) or 24.0
    generated_fps = float(generated.get(cv2.CAP_PROP_FPS) or 0) or fps
    generated_frame_count = int(generated.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(base.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(base.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        base.release()
        generated.release()
        raise ValueError("invalid_foreground_video_size")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_video = destination.parent / f".{destination.stem}.video.mp4"
    writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        base.release()
        generated.release()
        raise ValueError("cannot_create_foreground_video")

    total_frames = 0
    composited_frames = 0
    cached_generated_index = -1
    cached_generated_frame = None

    def read_generated_at(time_seconds: float):
        nonlocal cached_generated_index, cached_generated_frame
        target_index = int(round(time_seconds * generated_fps)) if generated_fps else total_frames
        if generated_frame_count > 0 and target_index >= generated_frame_count:
            return None
        if target_index == cached_generated_index and cached_generated_frame is not None:
            return cached_generated_frame
        if target_index != cached_generated_index + 1:
            generated.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_index))
        ok, frame = generated.read()
        if not ok or frame is None:
            return None
        cached_generated_index = target_index
        cached_generated_frame = frame
        return frame

    try:
        while True:
            ok_base, base_frame = base.read()
            if not ok_base or base_frame is None:
                break
            time_seconds = total_frames / fps
            track_index = int(round((time_seconds - clip_start_time) * track_fps)) if track_fps else total_frames
            mask_path = _nearest_mask(masks, indices, track_index)
            output_frame = base_frame
            if mask_path is not None:
                mask = _read_grayscale_image(mask_path)
                generated_frame = read_generated_at(time_seconds)
                if mask is not None and generated_frame is not None:
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
                    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    if dilate_pixels > 0:
                        size = dilate_pixels * 2 + 1
                        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
                    blur = max(1, int(feather_pixels))
                    if blur % 2 == 0:
                        blur += 1
                    alpha = cv2.GaussianBlur(mask, (blur, blur), 0).astype(np.float32)[..., None] / 255.0
                    generated_frame = cv2.resize(generated_frame, (width, height), interpolation=cv2.INTER_LINEAR)
                    output_frame = np.clip(
                        generated_frame.astype(np.float32) * alpha + base_frame.astype(np.float32) * (1.0 - alpha),
                        0, 255,
                    ).astype(np.uint8)
                    composited_frames += 1
            writer.write(output_frame)
            total_frames += 1
    finally:
        writer.release()
        base.release()
        generated.release()
    if total_frames <= 0 or composited_frames <= 0:
        temp_video.unlink(missing_ok=True)
        raise ValueError("no_foreground_frames_composited")

    from .ffmpeg_tools import ffmpeg_path, run_command
    try:
        run_command([
            ffmpeg_path(), "-y", "-loglevel", "error", "-i", str(temp_video), "-i", str(base_path),
            "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "copy", "-movflags", "+faststart", "-shortest", str(destination),
        ])
    finally:
        temp_video.unlink(missing_ok=True)
    return {"output_path": str(destination), "total_frames": total_frames, "composited_frames": composited_frames}


def build_focus_video(
    base_video: str | Path,
    track_dir: str | Path,
    output_path: str | Path,
    *,
    feather_pixels: int = 25,
    dilate_pixels: int = 10,
    background_dim: float = 0.18,
    background_blur_pixels: int = 0,
) -> dict[str, object]:
    """Create a full-frame guide video where only the tracked role stays prominent."""
    import cv2
    import numpy as np

    base_path = Path(base_video)
    track_path = Path(track_dir)
    destination = Path(output_path)
    summary_path = track_path / "track_summary.json"
    if not base_path.exists() or not summary_path.exists():
        raise ValueError("focus_video_input_not_found")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    track_fps = float(summary.get("track_fps") or 0)
    clip_start_time = float(summary.get("clip_start_time") or 0)
    masks = _index_masks(track_path)
    indices = sorted(masks)
    if not masks:
        raise ValueError("focus_video_masks_not_found")

    base = cv2.VideoCapture(str(base_path))
    if not base.isOpened():
        base.release()
        raise ValueError("cannot_open_focus_base_video")
    fps = float(base.get(cv2.CAP_PROP_FPS) or 0) or 24.0
    width = int(base.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(base.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        base.release()
        raise ValueError("invalid_focus_video_size")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_video = destination.parent / f".{destination.stem}.focus.mp4"
    writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        base.release()
        raise ValueError("cannot_create_focus_video")

    total_frames = 0
    focused_frames = 0
    try:
        while True:
            ok, frame = base.read()
            if not ok or frame is None:
                break
            time_seconds = total_frames / fps
            track_index = int(round((time_seconds - clip_start_time) * track_fps)) if track_fps else total_frames
            mask_path = _nearest_mask(masks, indices, track_index)
            mask = _read_grayscale_image(mask_path) if mask_path is not None else None
            if mask is None:
                output_frame = np.clip(frame.astype(np.float32) * background_dim, 0, 255).astype(np.uint8)
            else:
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                if dilate_pixels > 0:
                    size = dilate_pixels * 2 + 1
                    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
                blur = max(1, int(feather_pixels))
                if blur % 2 == 0:
                    blur += 1
                alpha = cv2.GaussianBlur(mask, (blur, blur), 0).astype(np.float32)[..., None] / 255.0
                blur = max(0, int(background_blur_pixels))
                if blur > 0:
                    if blur % 2 == 0:
                        blur += 1
                    background_source = cv2.GaussianBlur(frame, (blur, blur), 0)
                else:
                    background_source = frame
                background = background_source.astype(np.float32) * background_dim
                output_frame = np.clip(
                    frame.astype(np.float32) * alpha + background * (1.0 - alpha),
                    0,
                    255,
                ).astype(np.uint8)
                focused_frames += 1
            writer.write(output_frame)
            total_frames += 1
    finally:
        writer.release()
        base.release()

    if total_frames <= 0 or focused_frames <= 0:
        temp_video.unlink(missing_ok=True)
        raise ValueError("no_focus_frames_written")

    from .ffmpeg_tools import ffmpeg_path, run_command
    try:
        run_command([
            ffmpeg_path(), "-y", "-loglevel", "error", "-i", str(temp_video), "-i", str(base_path),
            "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
            "-c:a", "copy", "-movflags", "+faststart", "-shortest", str(destination),
        ])
    finally:
        temp_video.unlink(missing_ok=True)
    return {
        "output_path": str(destination),
        "total_frames": total_frames,
        "focused_frames": focused_frames,
        "feather_pixels": feather_pixels,
        "dilate_pixels": dilate_pixels,
        "background_dim": background_dim,
        "background_blur_pixels": background_blur_pixels,
    }


def build_focus_video_from_color_mask_video(
    base_video: str | Path,
    mask_video: str | Path,
    output_path: str | Path,
    *,
    color: str,
    feather_pixels: int = 5,
    erode_pixels: int = 0,
    dilate_pixels: int = 0,
    background_dim: float = 0.78,
    background_blur_pixels: int = 0,
) -> dict[str, object]:
    """Create a guide video from a colored SAM3 mask video.

    The SCAIL2/SAM3 colored-mask video uses palette colors for identities.
    This helper extracts one palette color and builds the Wan2.2 focus input
    without relying on local per-role mask tracks.
    """
    import cv2
    import numpy as np

    base_path = Path(base_video)
    mask_path = Path(mask_video)
    destination = Path(output_path)
    if not base_path.exists() or not mask_path.exists():
        raise ValueError("color_focus_video_input_not_found")

    base = cv2.VideoCapture(str(base_path))
    masks = cv2.VideoCapture(str(mask_path))
    if not base.isOpened() or not masks.isOpened():
        base.release()
        masks.release()
        raise ValueError("cannot_open_color_focus_videos")

    fps = float(base.get(cv2.CAP_PROP_FPS) or 0) or 24.0
    mask_fps = float(masks.get(cv2.CAP_PROP_FPS) or 0) or fps
    mask_frame_count = int(masks.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(base.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(base.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        base.release()
        masks.release()
        raise ValueError("invalid_color_focus_video_size")

    color_key = str(color or "").strip().lower()
    if color_key not in _MASK_COLOR_CHANNELS:
        raise ValueError(f"unsupported_mask_color: {color}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_video = destination.parent / f".{destination.stem}.focus.mp4"
    writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        base.release()
        masks.release()
        raise ValueError("cannot_create_color_focus_video")

    total_frames = 0
    focused_frames = 0
    mask_ratios: list[float] = []
    cached_mask_index = -1
    cached_mask_frame = None

    def read_mask_at(time_seconds: float):
        nonlocal cached_mask_index, cached_mask_frame
        target_index = int(round(time_seconds * mask_fps)) if mask_fps else total_frames
        if mask_frame_count > 0 and target_index >= mask_frame_count:
            target_index = mask_frame_count - 1
        target_index = max(0, target_index)
        if target_index == cached_mask_index and cached_mask_frame is not None:
            return cached_mask_frame
        if target_index != cached_mask_index + 1:
            masks.set(cv2.CAP_PROP_POS_FRAMES, target_index)
        ok, frame = masks.read()
        if not ok or frame is None:
            return None
        cached_mask_index = target_index
        cached_mask_frame = frame
        return frame

    try:
        while True:
            ok, frame = base.read()
            if not ok or frame is None:
                break
            time_seconds = total_frames / fps
            mask_frame = read_mask_at(time_seconds)
            if mask_frame is None:
                output_frame = np.clip(frame.astype(np.float32) * background_dim, 0, 255).astype(np.uint8)
            else:
                mask_frame = cv2.resize(mask_frame, (width, height), interpolation=cv2.INTER_LINEAR)
                mask = _palette_color_mask(mask_frame, color_key)
                if erode_pixels > 0:
                    size = erode_pixels * 2 + 1
                    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
                if dilate_pixels > 0:
                    size = dilate_pixels * 2 + 1
                    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
                mask_ratio = float(np.count_nonzero(mask > 127)) / float(width * height)
                mask_ratios.append(mask_ratio)
                blur = max(1, int(feather_pixels))
                if blur % 2 == 0:
                    blur += 1
                alpha = cv2.GaussianBlur(mask, (blur, blur), 0).astype(np.float32)[..., None] / 255.0
                blur = max(0, int(background_blur_pixels))
                if blur > 0:
                    if blur % 2 == 0:
                        blur += 1
                    background_source = cv2.GaussianBlur(frame, (blur, blur), 0)
                else:
                    background_source = frame
                background = background_source.astype(np.float32) * background_dim
                output_frame = np.clip(
                    frame.astype(np.float32) * alpha + background * (1.0 - alpha),
                    0,
                    255,
                ).astype(np.uint8)
                if np.any(mask > 127):
                    focused_frames += 1
            writer.write(output_frame)
            total_frames += 1
    finally:
        writer.release()
        base.release()
        masks.release()

    if total_frames <= 0 or focused_frames <= 0:
        temp_video.unlink(missing_ok=True)
        raise ValueError("no_color_focus_frames_written")

    from .ffmpeg_tools import ffmpeg_path, run_command
    try:
        run_command([
            ffmpeg_path(), "-y", "-loglevel", "error", "-i", str(temp_video), "-i", str(base_path),
            "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
            "-c:a", "copy", "-movflags", "+faststart", "-shortest", str(destination),
        ])
    finally:
        temp_video.unlink(missing_ok=True)
    return {
        "output_path": str(destination),
        "total_frames": total_frames,
        "focused_frames": focused_frames,
        "feather_pixels": feather_pixels,
        "erode_pixels": erode_pixels,
        "dilate_pixels": dilate_pixels,
        "background_dim": background_dim,
        "background_blur_pixels": background_blur_pixels,
        "mask_color": color_key,
        "mask_video": str(mask_path),
        "mean_mask_ratio": float(sum(mask_ratios) / len(mask_ratios)) if mask_ratios else 0.0,
        "min_mask_ratio": float(min(mask_ratios)) if mask_ratios else 0.0,
        "max_mask_ratio": float(max(mask_ratios)) if mask_ratios else 0.0,
    }


def masked_mean_absdiff(
    base_video: str | Path,
    generated_video: str | Path,
    track_dir: str | Path,
    *,
    max_samples: int = 12,
) -> dict[str, object]:
    """Measure how much the generated video changed the tracked role area."""
    import cv2
    import numpy as np

    base_path = Path(base_video)
    generated_path = Path(generated_video)
    track_path = Path(track_dir)
    summary_path = track_path / "track_summary.json"
    if not base_path.exists() or not generated_path.exists() or not summary_path.exists():
        raise ValueError("masked_diff_input_not_found")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    track_fps = float(summary.get("track_fps") or 0)
    clip_start_time = float(summary.get("clip_start_time") or 0)
    masks = _index_masks(track_path)
    indices = sorted(masks)
    if not masks:
        raise ValueError("masked_diff_masks_not_found")

    base = cv2.VideoCapture(str(base_path))
    generated = cv2.VideoCapture(str(generated_path))
    if not base.isOpened() or not generated.isOpened():
        base.release()
        generated.release()
        raise ValueError("cannot_open_masked_diff_videos")

    fps = float(base.get(cv2.CAP_PROP_FPS) or 0) or 24.0
    generated_fps = float(generated.get(cv2.CAP_PROP_FPS) or 0) or fps
    base_frame_count = int(base.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    generated_frame_count = int(generated.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(base.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(base.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if base_frame_count <= 0:
        base.release()
        generated.release()
        raise ValueError("masked_diff_empty_base")

    sample_count = max(1, min(max_samples, base_frame_count))
    frame_indices = np.linspace(0, max(0, base_frame_count - 1), sample_count).astype(int).tolist()
    values: list[float] = []
    try:
        for frame_idx in frame_indices:
            time_seconds = frame_idx / fps
            track_index = int(round((time_seconds - clip_start_time) * track_fps)) if track_fps else frame_idx
            mask_path = _nearest_mask(masks, indices, track_index)
            mask = _read_grayscale_image(mask_path) if mask_path is not None else None
            if mask is None:
                continue
            base.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            generated_index = int(round(time_seconds * generated_fps)) if generated_fps else frame_idx
            if generated_frame_count > 0:
                generated_index = min(max(0, generated_index), generated_frame_count - 1)
            generated.set(cv2.CAP_PROP_POS_FRAMES, generated_index)
            ok_base, base_frame = base.read()
            ok_generated, generated_frame = generated.read()
            if not ok_base or not ok_generated or base_frame is None or generated_frame is None:
                continue
            generated_frame = cv2.resize(generated_frame, (width, height), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            region = mask > 127
            if not np.any(region):
                continue
            values.append(
                float(np.mean(np.abs(base_frame.astype(np.int16)[region] - generated_frame.astype(np.int16)[region])))
            )
    finally:
        base.release()
        generated.release()

    return {
        "samples": len(values),
        "mean_absdiff": float(sum(values) / len(values)) if values else 0.0,
        "min_absdiff": float(min(values)) if values else 0.0,
        "max_absdiff": float(max(values)) if values else 0.0,
    }


def _index_masks(track_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in track_dir.glob("mask_*.png"):
        try:
            result[int(path.stem.rsplit("_", 1)[-1])] = path
        except ValueError:
            continue
    return result


def _nearest_mask(masks: dict[int, Path], indices: list[int], target: int, tolerance: int = 2) -> Path | None:
    if target in masks:
        return masks[target]
    position = bisect.bisect_left(indices, target)
    candidates = indices[max(0, position - 1): position + 1]
    if not candidates:
        return None
    closest = min(candidates, key=lambda value: abs(value - target))
    return masks[closest] if abs(closest - target) <= tolerance else None


def _read_grayscale_image(path: Path):
    """Read images through np.fromfile so Windows unicode paths work with OpenCV."""
    import cv2
    import numpy as np

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size <= 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


_MASK_COLOR_CHANNELS = {
    "blue": 0,
    "red": 2,
    "green": 1,
    "magenta": 2,
    "cyan": 0,
    "yellow": 1,
}


def _palette_color_mask(frame, color: str):
    import numpy as np

    b = frame[..., 0].astype(np.int16)
    g = frame[..., 1].astype(np.int16)
    r = frame[..., 2].astype(np.int16)
    margin = 24
    threshold = 80
    if color == "blue":
        region = (b > threshold) & (b > g + margin) & (b > r + margin)
    elif color == "red":
        region = (r > threshold) & (r > g + margin) & (r > b + margin)
    elif color == "green":
        region = (g > threshold) & (g > r + margin) & (g > b + margin)
    elif color == "magenta":
        region = (r > threshold) & (b > threshold) & (g < min(r, b) - margin)
    elif color == "cyan":
        region = (b > threshold) & (g > threshold) & (r < min(b, g) - margin)
    elif color == "yellow":
        region = (r > threshold) & (g > threshold) & (b < min(r, g) - margin)
    else:
        raise ValueError(f"unsupported_mask_color: {color}")
    return np.where(region, 255, 0).astype(np.uint8)
