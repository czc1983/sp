"""H3 白膜链路的切分后处理（Mode 2 专用）。

在 run_segmentation_v2 的技术切分/人脸/视觉裁判之后运行，负责：
  1. 超长段按 H3 帧网格（17k+5 @24fps，本地工作流上限 124 帧 = 5.17s）切开，
     每段保留自己的属性（修复旧强制切分统一复制第一段属性的 bug）。
  2. 过短碎片（< 1.5s）向「同人数、弱边界」的相邻段合并；无法合并的标记待人工。
  3. 为每段标注 h3_frames / h3_pad_seconds（H3 对齐所需帧数与补帧量），
     人数 > 3 的段标记 h3_review（H3 角色参考图最多 3 张）。

只动 sub_segments 列表，不触碰 Mode 1 的 transfer/render 链路。
"""
from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

H3_FPS = 24
H3_FRAME_STEP = 17
H3_FRAME_BASE = 5
H3_MAX_FRAMES = 124  # 本地 4090 工作流实测上限：124f = 5.1667s

MIN_SEGMENT_DURATION = 1.5
MAX_SEGMENT_DURATION = H3_MAX_FRAMES / H3_FPS  # 5.1667s

# 硬边界来源：含这些来源的边界不允许被合并跨越
HARD_SOURCES = {
    "pyscene", "omnishotcut", "face_id", "visual_model", "yolo_transient_multi",
}


def h3_grid_frames(seconds: float, *, max_frames: int = H3_MAX_FRAMES) -> int:
    """时长 → H3 需要的最小网格帧数（向上吸附，不截内容）。"""
    frames = max(H3_FRAME_BASE, int(math.ceil(seconds * H3_FPS)))
    k = max(0, math.ceil((frames - H3_FRAME_BASE) / H3_FRAME_STEP))
    return H3_FRAME_STEP * k + H3_FRAME_BASE


def h3_grid_durations(*, max_frames: int = H3_MAX_FRAMES) -> list[float]:
    """17k+5 网格对应的秒数表（升序，含上限）。"""
    out: list[float] = []
    frames = H3_FRAME_BASE
    while frames <= max_frames:
        out.append(round(frames / H3_FPS, 3))
        frames += H3_FRAME_STEP
    return out


def _is_hard_boundary(segment: dict[str, Any], side: str) -> bool:
    sources = set(segment.get(f"{side}_sources") or [])
    return bool(sources & HARD_SOURCES)


def split_overlong_segments(
    segments: list[dict[str, Any]],
    *,
    max_duration: float = MAX_SEGMENT_DURATION,
) -> list[dict[str, Any]]:
    """把超过 max_duration 的段均衡切开（网格感知），属性从被切的那一段复制。"""
    if max_duration <= 0:
        return segments
    result: list[dict[str, Any]] = []
    split_count = 0
    for seg in segments:
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or 0.0)
        duration = end - start
        if duration <= max_duration + 1e-6:
            result.append(seg)
            continue
        # 均衡切：n 段等长，避免前 N-1 段顶格、最后剩一条碎尾巴
        n = int(math.ceil(duration / max_duration))
        piece = duration / n
        split_count += n - 1
        for index in range(n):
            child = dict(seg)  # 属性来自被切的本段，而不是列表第一段
            child["start"] = round(start + piece * index, 3)
            child["end"] = round(end if index == n - 1 else start + piece * (index + 1), 3)
            child["start_sources"] = list(seg.get("start_sources") or [])
            child["end_sources"] = list(seg.get("end_sources") or [])
            if index > 0:
                child["start_sources"] = ["h3_grid_split"]
            if index < n - 1:
                child["end_sources"] = ["h3_grid_split"]
            result.append(child)
    if split_count:
        logger.info("[H3网格] 超长段均衡切分: 新增 %d 个切点（max=%.2fs）", split_count, max_duration)
    return result


def merge_tiny_segments(
    segments: list[dict[str, Any]],
    *,
    min_duration: float = MIN_SEGMENT_DURATION,
    max_duration: float = MAX_SEGMENT_DURATION,
) -> list[dict[str, Any]]:
    """过短碎片向「同人数、弱边界」的相邻段合并；合不了的标记 h3_review。"""
    if not segments:
        return segments
    merged: list[dict[str, Any]] = []
    i = 0
    merge_count = 0
    while i < len(segments):
        seg = dict(segments[i])
        duration = float(seg.get("end") or 0) - float(seg.get("start") or 0)
        if duration >= min_duration or len(segments) == 1:
            merged.append(seg)
            i += 1
            continue
        # 候选邻居：优先下一段，其次上一段（已入 merged 的最后一条）
        absorbed = False
        if i + 1 < len(segments):
            nxt = segments[i + 1]
            same_cast = int(seg.get("person_count") or -1) == int(nxt.get("person_count") or -1)
            combined = float(nxt.get("end") or 0) - float(seg.get("start") or 0)
            if same_cast and combined <= max_duration and not _is_hard_boundary(seg, "end"):
                nxt = dict(nxt)
                nxt["start"] = seg["start"]
                nxt["start_sources"] = list(seg.get("start_sources") or [])
                nxt.setdefault("h3_notes", []).append(f"absorbed_tiny_{duration:.2f}s")
                segments[i + 1] = nxt
                absorbed = True
                merge_count += 1
        if not absorbed and merged:
            prev = merged[-1]
            same_cast = int(seg.get("person_count") or -1) == int(prev.get("person_count") or -1)
            combined = float(seg.get("end") or 0) - float(prev.get("start") or 0)
            if same_cast and combined <= max_duration and not _is_hard_boundary(seg, "start"):
                prev["end"] = seg["end"]
                prev["end_sources"] = list(seg.get("end_sources") or [])
                prev.setdefault("h3_notes", []).append(f"absorbed_tiny_{duration:.2f}s")
                absorbed = True
                merge_count += 1
        if not absorbed:
            seg["h3_review"] = f"too_short_{duration:.2f}s_unique_cast"
            merged.append(seg)
        i += 1
    if merge_count:
        logger.info("[H3网格] 过短碎片合并: %d 段被吸收（min=%.2fs）", merge_count, min_duration)
    return merged


def annotate_h3_targets(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """为每段标注 H3 对齐信息：目标帧数、补帧量、人数超额标记。"""
    for seg in segments:
        duration = float(seg.get("end") or 0) - float(seg.get("start") or 0)
        frames = h3_grid_frames(duration)
        seg["h3_frames"] = frames
        seg["h3_pad_seconds"] = round(frames / H3_FPS - duration, 3)
        person_count = int(seg.get("person_count") or -1)
        if person_count > 3 and "h3_review" not in seg:
            seg["h3_review"] = f"cast_over_3 ({person_count})"
    return segments


def tune_segments_for_h3(
    segments: list[dict[str, Any]],
    *,
    min_duration: float = MIN_SEGMENT_DURATION,
    max_duration: float = MAX_SEGMENT_DURATION,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """H3 白膜专用切分后处理：超长切分 → 碎片合并 → 网格标注。"""
    before = len(segments)
    tuned = split_overlong_segments(segments, max_duration=max_duration)
    after_split = len(tuned)
    tuned = merge_tiny_segments(tuned, min_duration=min_duration, max_duration=max_duration)
    tuned = annotate_h3_targets(tuned)
    report = {
        "before": before,
        "after_split": after_split,
        "after_merge": len(tuned),
        "min_duration": min_duration,
        "max_duration": round(max_duration, 3),
        "grid_durations": h3_grid_durations(),
        "review_segments": [
            {"start": s.get("start"), "end": s.get("end"), "reason": s["h3_review"]}
            for s in tuned if s.get("h3_review")
        ],
    }
    logger.info(
        "[H3网格] 切分后处理完成: %d → 切 %d → 并 %d 段；待人工 %d 段",
        before, after_split, len(tuned), len(report["review_segments"]),
    )
    return tuned, report
