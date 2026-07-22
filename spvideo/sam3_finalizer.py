from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger("spvideo")

SAM3_MODULE_CANDIDATES = (
    "sam3",
    "sam2",
    "segment_anything_3",
    "segment_anything_2",
    "segment_anything",
)
TRACK_FPS = 8.0
MAX_TRACK_FRAMES = 48
MIN_TRACK_DURATION = 1.5
MIN_STABLE_TRACK_FRAMES = 3
MIN_PERSISTENT_LOSS_FRAMES = 6
MIN_SPLIT_DURATION = 0.75


def find_sam3_runtime() -> dict[str, Any]:
    """Return local SAM/SAM3 runtime availability without importing heavy modules."""
    checked: list[str] = []
    for module_name in SAM3_MODULE_CANDIDATES:
        checked.append(module_name)
        if importlib.util.find_spec(module_name) is not None:
            return {
                "available": True,
                "module": module_name,
                "checked": checked,
            }
    return {
        "available": False,
        "module": None,
        "checked": checked,
    }


def finalize_segments_with_sam3(
    video_path: str | Path,
    sub_segments: list[dict[str, Any]],
    work_dir: str | Path,
) -> dict[str, Any]:
    """Use SAM3 tracks to add safe cuts when a tracked subject permanently disappears.

    Existing scene cuts are intentionally preserved. This layer only subdivides a
    single-person segment after SAM3 loses the prompted subject for the remainder
    of the clip, which catches exits and many one-person-to-another replacements.
    """
    runtime = find_sam3_runtime()
    work_dir = Path(work_dir)
    result: dict[str, Any] = {
        "enabled": True,
        "available": runtime["available"],
        "module": runtime["module"],
        "checked_modules": runtime["checked"],
        "input_count": len(sub_segments),
        "output_count": len(sub_segments),
        "changed": False,
        "skipped": False,
        "reason": "",
        "video_path": str(video_path),
        "work_dir": str(work_dir),
        "tracks": [],
        "added_cuts": [],
    }

    if not runtime["available"]:
        result["skipped"] = True
        result["reason"] = "No local SAM3/SAM tracker package was found."
        return {"segments": sub_segments, "result": result}

    try:
        tracker = _create_tracker()
    except Exception as exc:  # noqa: BLE001
        result["skipped"] = True
        result["reason"] = f"SAM3 tracker initialization failed: {exc}"
        return {"segments": sub_segments, "result": result}

    work_dir.mkdir(parents=True, exist_ok=True)
    finalized: list[dict[str, Any]] = []
    try:
        for index, segment in enumerate(sub_segments, start=1):
            track_record: dict[str, Any] = {
                "segment_index": index,
                "start": round(float(segment.get("start", 0.0)), 3),
                "end": round(float(segment.get("end", 0.0)), 3),
            }
            prompt = _segment_prompt_point(segment, video_path)
            duration = track_record["end"] - track_record["start"]
            if int(segment.get("person_count", -1)) != 1:
                track_record["skipped"] = "requires_exactly_one_person"
                result["tracks"].append(track_record)
                finalized.append(segment)
                continue
            if not _needs_sam3_tracking(segment):
                track_record["skipped"] = "no_trajectory_risk"
                result["tracks"].append(track_record)
                finalized.append(segment)
                continue
            if duration < MIN_TRACK_DURATION:
                track_record["skipped"] = "segment_too_short"
                result["tracks"].append(track_record)
                finalized.append(segment)
                continue
            if prompt is None:
                track_record["skipped"] = "missing_yolo_subject_prompt"
                result["tracks"].append(track_record)
                finalized.append(segment)
                continue

            try:
                clip_path, frame_count, track_fps = _make_track_clip(
                    video_path,
                    start=track_record["start"],
                    end=track_record["end"],
                    output_path=work_dir / f"segment_{index:03d}.mp4",
                )
                track = tracker.track_by_point(
                    video_path=str(clip_path),
                    point=prompt,
                    frame_idx=0,
                    max_frames=frame_count,
                )
                tracked = _tracked_flags(track)
                loss_index = _persistent_loss_index(tracked)
                track_record.update(
                    {
                        "clip_path": str(clip_path),
                        "prompt_point": prompt,
                        "frame_count": frame_count,
                        "track_fps": track_fps,
                        "tracked_frames": sum(tracked),
                        "persistent_loss_frame": loss_index,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[SAM3] segment [%.2f-%.2f] failed: %s",
                               track_record["start"], track_record["end"], exc)
                track_record["error"] = str(exc)
                result["tracks"].append(track_record)
                finalized.append(segment)
                continue

            if loss_index is None:
                result["tracks"].append(track_record)
                finalized.append(segment)
                continue

            cut = round(track_record["start"] + loss_index / track_fps, 3)
            split_segments = _split_segment_at(segment, cut)
            if len(split_segments) == 1:
                track_record["ignored_cut"] = cut
                track_record["skip_reason"] = "would_create_short_segment"
                result["tracks"].append(track_record)
                finalized.append(segment)
                continue

            track_record["cut"] = cut
            track_record["action"] = "split_on_persistent_subject_loss"
            result["tracks"].append(track_record)
            result["added_cuts"].append(
                {
                    "time": cut,
                    "segment_index": index,
                    "reason": "sam3_persistent_subject_loss",
                }
            )
            finalized.extend(split_segments)
    finally:
        tracker.close()

    result["output_count"] = len(finalized)
    result["changed"] = len(finalized) != len(sub_segments)
    result["reason"] = (
        "SAM3 checked single-person segments and added cuts only when the "
        "prompted subject was persistently lost."
    )
    return {"segments": finalized, "result": result}


def _create_tracker() -> Any:
    from .sam3_tracker import SAM3Tracker

    return SAM3Tracker()


def _needs_sam3_tracking(segment: dict[str, Any]) -> bool:
    """Run the expensive tracker only where prior layers saw temporal risk."""
    sources = set(segment.get("start_sources") or []) | set(segment.get("end_sources") or [])
    if sources & {"yolo", "yolo_transient_multi", "face_id", "sam3_track_lost"}:
        return True
    if segment.get("_face_split") or segment.get("transient_multi_person"):
        return True

    detections = segment.get("_all_frame_detections") or []
    counts = [int(item.get("person_count", -1)) for item in detections]
    return bool(counts and any(count != 1 for count in counts))


def _segment_prompt_point(
    segment: dict[str, Any],
    video_path: str | Path | None = None,
) -> list[float] | None:
    bbox = segment.get("main_person_bbox")
    if not bbox:
        detections = segment.get("_all_frame_detections") or []
        if detections:
            persons = detections[0].get("persons") or []
            if persons:
                bbox = persons[0].get("bbox")
    if not bbox or len(bbox) != 4:
        return None

    # YOLO boxes are pixel coordinates. The first detector frame has the same
    # aspect ratio as the SAM3 clip, so normalized coordinates remain valid.
    frame_path = ""
    detections = segment.get("_all_frame_detections") or []
    if detections:
        frame_path = str(detections[0].get("frame_path") or "")
    width, height = _image_size(frame_path)
    if width <= 0 or height <= 0:
        width, height = _video_size(video_path)
    if width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = (float(value) for value in bbox)
    return [
        max(0.02, min(0.98, ((x1 + x2) / 2.0) / width)),
        max(0.02, min(0.98, ((y1 + y2) / 2.0) / height)),
    ]


def _image_size(path: str) -> tuple[int, int]:
    if not path:
        return 0, 0
    try:
        import cv2

        image = cv2.imread(path)
        if image is None:
            return 0, 0
        height, width = image.shape[:2]
        return int(width), int(height)
    except Exception:  # noqa: BLE001
        return 0, 0


def _video_size(video_path: str | Path | None) -> tuple[int, int]:
    if not video_path:
        return 0, 0
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            return width, height
        finally:
            cap.release()
    except Exception:  # noqa: BLE001
        return 0, 0


def _make_track_clip(
    video_path: str | Path,
    *,
    start: float,
    end: float,
    output_path: Path,
) -> tuple[Path, int, float]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("cannot_open_video")
    try:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            raise ValueError("empty_video")
        track_fps = min(source_fps, TRACK_FPS)
        start_frame = max(0, min(total_frames - 1, int(round(start * source_fps))))
        end_frame = max(start_frame + 1, min(total_frames, int(round(end * source_fps))))
        duration = (end_frame - start_frame) / source_fps
        frame_count = max(1, min(MAX_TRACK_FRAMES, int(round(duration * track_fps))))
        source_indices = [
            min(end_frame - 1, int(round(start_frame + frame_index * source_fps / track_fps)))
            for frame_index in range(frame_count)
        ]

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError("cannot_read_video_frame")
        height, width = frame.shape[:2]
        scale = min(1.0, 720 / max(width, height))
        output_size = (max(2, int(width * scale)), max(2, int(height * scale)))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            track_fps,
            output_size,
        )
        if not writer.isOpened():
            raise ValueError("cannot_create_track_clip")
        written = 0
        current_index = start_frame
        try:
            for desired_index in source_indices:
                while ok and frame is not None and current_index < desired_index:
                    ok, frame = cap.read()
                    current_index += 1
                if not ok or frame is None:
                    break
                output_frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
                writer.write(output_frame)
                written += 1
        finally:
            writer.release()
        if written < 1:
            raise ValueError("empty_track_clip")
        return output_path, written, track_fps
    finally:
        cap.release()


def _tracked_flags(track: dict[str, Any]) -> list[bool]:
    boxes = track.get("boxes") or []
    scores = track.get("scores") or []
    count = max(len(boxes), len(scores))
    return [
        index < len(boxes)
        and boxes[index] is not None
        and index < len(scores)
        and float(scores[index]) >= 0.35
        for index in range(count)
    ]


def _persistent_loss_index(tracked: list[bool]) -> int | None:
    if len(tracked) < MIN_STABLE_TRACK_FRAMES + MIN_PERSISTENT_LOSS_FRAMES:
        return None
    if sum(tracked[:MIN_STABLE_TRACK_FRAMES]) < MIN_STABLE_TRACK_FRAMES:
        return None

    for index in range(MIN_STABLE_TRACK_FRAMES, len(tracked) - MIN_PERSISTENT_LOSS_FRAMES + 1):
        if not any(tracked[index:]):
            return index
    return None


def _split_segment_at(segment: dict[str, Any], cut: float) -> list[dict[str, Any]]:
    start = float(segment["start"])
    end = float(segment["end"])
    if cut - start < MIN_SPLIT_DURATION or end - cut < MIN_SPLIT_DURATION:
        return [segment]

    left = dict(segment)
    right = dict(segment)
    left["end"] = cut
    right["start"] = cut
    left["end_sources"] = sorted(set(left.get("end_sources", [])) | {"sam3_track_lost"})
    right["start_sources"] = sorted(set(right.get("start_sources", [])) | {"sam3_track_lost"})
    left["sam3_track_status"] = "subject_tracked"
    right["sam3_track_status"] = "subject_lost"
    return [left, right]
