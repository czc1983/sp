from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .ffmpeg_tools import extract_frame
from .models import ensure_dir

logger = logging.getLogger(__name__)


def merge_visually_similar_segments(
    video_path: str | Path,
    sub_segments: list[dict[str, Any]],
    work_dir: str | Path,
    *,
    fps: float,
    diff_threshold: float = 0.075,
    tiny_diff_threshold: float = 0.13,
    tiny_duration: float = 0.9,
    transition_cluster_max_duration: float = 3.5,
) -> dict[str, Any]:
    """Merge adjacent segments when the boundary is visually weak.

    This is meant to clean up OmniShotCut over-fragmentation. It validates a
    proposed boundary by comparing real frames on both sides of that boundary.
    """
    if len(sub_segments) <= 1:
        return {
            "segments": sub_segments,
            "result": {
                "enabled": True,
                "input_count": len(sub_segments),
                "output_count": len(sub_segments),
                "merged_count": 0,
                "decisions": [],
            },
        }

    work_dir = ensure_dir(work_dir)
    frame_step = 1.0 / max(fps, 1.0)
    offset = max(frame_step * 2.0, 0.08)

    merged: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    current = dict(sub_segments[0])

    for index, next_seg in enumerate(sub_segments[1:], start=1):
        if index == 1 or index == len(sub_segments) - 1 or index % 25 == 0:
            logger.info("[画面合并] 检查边界 %d/%d", index, len(sub_segments) - 1)
        next_copy = dict(next_seg)
        boundary = float(current["end"])
        before_time = _clamp(boundary - offset, float(current["start"]), float(current["end"]) - frame_step * 0.25)
        after_time = _clamp(boundary + offset, float(next_copy["start"]) + frame_step * 0.25, float(next_copy["end"]))
        score = _boundary_diff_score(video_path, before_time, after_time, work_dir, index)
        cur_duration = float(current["end"]) - float(current["start"])
        next_duration = float(next_copy["end"]) - float(next_copy["start"])
        cur_transition = _segment_transition_stats(video_path, current, work_dir, index, "current")
        next_transition = _segment_transition_stats(video_path, next_copy, work_dir, index, "next")
        weak = score <= diff_threshold
        tiny_weak = min(cur_duration, next_duration) <= tiny_duration and score <= tiny_diff_threshold
        omni_boundary = _has_source(current, "end_sources", "omnishotcut") or _has_source(next_copy, "start_sources", "omnishotcut")
        protected_sources = {"pyscene", "yolo", "yolo_transient_multi", "face_id", "gemini", "pre_director"}
        protected = (
            _has_any_source(current, "end_sources", protected_sources)
            or _has_any_source(next_copy, "start_sources", protected_sources)
        )
        transition_fragment = _should_merge_transition_fragment(
            current,
            next_copy,
            cur_duration=cur_duration,
            next_duration=next_duration,
            cur_transition=cur_transition,
            next_transition=next_transition,
            tiny_duration=tiny_duration,
            max_cluster_duration=transition_cluster_max_duration,
        )
        # 仅当至少一侧是真正碎片时才合并 Omni 边界，防止连锁合并成长段
        omni_fragment_side = omni_boundary and min(cur_duration, next_duration) <= max(tiny_duration, 2.5)
        should_merge = not protected and (transition_fragment or (omni_fragment_side and (weak or tiny_weak)))
        if protected:
            reason = "protected_boundary"
        elif transition_fragment:
            reason = "fade_transition_fragment"
        elif not omni_fragment_side and omni_boundary:
            reason = "omni_too_long_to_merge"
        elif weak:
            reason = "weak_visual_boundary"
        elif tiny_weak:
            reason = "tiny_weak_visual_boundary"
        else:
            reason = "kept"

        decision = {
            "boundary": boundary,
            "before_time": before_time,
            "after_time": after_time,
            "diff_score": round(score, 5),
            "current_duration": round(cur_duration, 3),
            "next_duration": round(next_duration, 3),
            "current_transition": cur_transition,
            "next_transition": next_transition,
            "merge": should_merge,
            "reason": reason,
        }
        decisions.append(decision)

        if should_merge:
            current = _merge_pair(current, next_copy, reason=reason)
        else:
            merged.append(current)
            current = next_copy

    merged.append(current)
    result = {
        "enabled": True,
        "input_count": len(sub_segments),
        "output_count": len(merged),
        "merged_count": len(sub_segments) - len(merged),
        "diff_threshold": diff_threshold,
        "tiny_diff_threshold": tiny_diff_threshold,
        "tiny_duration": tiny_duration,
        "transition_cluster_max_duration": transition_cluster_max_duration,
        "decisions": decisions,
    }
    (Path(work_dir) / "visual_merge.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"segments": merged, "result": result}


def _boundary_diff_score(
    video_path: str | Path,
    before_time: float,
    after_time: float,
    work_dir: Path,
    index: int,
) -> float:
    before_path = work_dir / f"{index:04d}_before.jpg"
    after_path = work_dir / f"{index:04d}_after.jpg"
    extract_frame(video_path, before_time, before_path)
    extract_frame(video_path, after_time, after_path)
    before = _read_image(before_path)
    after = _read_image(after_path)
    if before is None or after is None:
        return 1.0
    return _image_diff(before, after)


def _read_image(path: Path):
    try:
        import cv2

        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        return cv2.imread(str(path))
    except Exception:
        return None


def _image_diff(before, after) -> float:
    import cv2

    before_small = cv2.resize(before, (160, 90), interpolation=cv2.INTER_AREA)
    after_small = cv2.resize(after, (160, 90), interpolation=cv2.INTER_AREA)
    before_gray = cv2.cvtColor(before_small, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after_small, cv2.COLOR_BGR2GRAY)
    mean_abs = float(np.mean(cv2.absdiff(before_gray, after_gray)) / 255.0)

    before_hist = cv2.calcHist([before_small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    after_hist = cv2.calcHist([after_small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(before_hist, before_hist)
    cv2.normalize(after_hist, after_hist)
    hist_distance = float(cv2.compareHist(before_hist, after_hist, cv2.HISTCMP_BHATTACHARYYA))

    return mean_abs * 0.65 + hist_distance * 0.35


def _segment_transition_stats(
    video_path: str | Path,
    segment: dict[str, Any],
    work_dir: Path,
    index: int,
    suffix: str,
) -> dict[str, Any]:
    start = float(segment["start"])
    end = float(segment["end"])
    mid = _clamp((start + end) / 2.0, start, end)
    frame_path = work_dir / f"{index:04d}_{suffix}_mid.jpg"
    extract_frame(video_path, mid, frame_path)
    image = _read_image(frame_path)
    if image is None:
        return {"time": round(mid, 3), "transition_like": False}
    stats = _frame_luma_stats(image)
    stats["time"] = round(mid, 3)
    stats["transition_like"] = _is_transition_like(stats)
    return stats


def _frame_luma_stats(image) -> dict[str, float]:
    import cv2

    small = cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return {
        "mean": round(float(np.mean(gray)), 4),
        "std": round(float(np.std(gray)), 4),
        "bright_ratio": round(float(np.mean(gray >= 0.82)), 4),
        "dark_ratio": round(float(np.mean(gray <= 0.08)), 4),
    }


def _is_transition_like(stats: dict[str, Any]) -> bool:
    mean = float(stats.get("mean", 0.0))
    std = float(stats.get("std", 1.0))
    bright_ratio = float(stats.get("bright_ratio", 0.0))
    dark_ratio = float(stats.get("dark_ratio", 0.0))
    low_contrast_wash = 0.25 <= mean <= 0.82 and std <= 0.11
    soft_bright_fade = mean >= 0.68 and std <= 0.17 and bright_ratio >= 0.12
    white_fade = bright_ratio >= 0.35 or (mean >= 0.74 and std <= 0.24) or soft_bright_fade
    black_fade = dark_ratio >= 0.45 or (mean <= 0.14 and std <= 0.18)
    return low_contrast_wash or white_fade or black_fade


def _should_merge_transition_fragment(
    current: dict[str, Any],
    next_seg: dict[str, Any],
    *,
    cur_duration: float,
    next_duration: float,
    cur_transition: dict[str, Any],
    next_transition: dict[str, Any],
    tiny_duration: float,
    max_cluster_duration: float,
) -> bool:
    cur_tiny = cur_duration <= tiny_duration
    next_tiny = next_duration <= tiny_duration
    cur_like = bool(cur_transition.get("transition_like"))
    next_like = bool(next_transition.get("transition_like"))
    current_cluster = bool(current.get("visual_transition_merged"))
    merged_duration = float(next_seg["end"]) - float(current["start"])
    if merged_duration > max_cluster_duration:
        return False

    if cur_tiny and cur_like:
        return True
    if current_cluster and next_tiny:
        return True
    if current_cluster and next_like and next_duration <= tiny_duration * 1.5:
        return True
    return False


def _merge_pair(left: dict[str, Any], right: dict[str, Any], *, reason: str | None = None) -> dict[str, Any]:
    merged = dict(left)
    merged["end"] = right["end"]
    merged["end_sources"] = right.get("end_sources", [])
    merged["visual_merged"] = True
    if reason:
        merged["visual_merge_reason"] = reason
    if reason == "fade_transition_fragment" or left.get("visual_transition_merged") or right.get("visual_transition_merged"):
        merged["visual_transition_merged"] = True
    merged["merged_from"] = list(left.get("merged_from", [left.get("shot_index")])) + list(
        right.get("merged_from", [right.get("shot_index")])
    )
    if left.get("person_count") != right.get("person_count"):
        merged["person_count"] = -1
    return merged


def _has_source(segment: dict[str, Any], key: str, source: str) -> bool:
    return source in (segment.get(key) or [])


def _has_any_source(segment: dict[str, Any], key: str, sources: set[str]) -> bool:
    return bool(set(segment.get(key) or []) & sources)


def _clamp(value: float, low: float, high: float) -> float:
    if high < low:
        return max(0.0, value)
    return min(max(value, low), high)
