"""两层镜头检测：PySceneDetect 硬切 + YOLO 人物进出细分。

原理：
  第一层 —— PySceneDetect 找导演剪辑的"硬切点"（不会被人进出画面误导）
  第二层 —— YOLO 在每个镜头内检测人物数量变化，发现"人进入/离开画面"
           的帧就细分，把"纯背景"和"有人"分开

输出：镜头分段列表，每段标记 人物数量、坐标、置信度、是否有纯背景帧
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── PySceneDetect ────────────────────────────────────────────────────────


def detect_hard_cuts(
    video_path: str | Path,
    min_scene_duration: float = 0.8,
    threshold: float = 27.0,
) -> list[tuple[float, float]]:
    """第一层：PySceneDetect 检测视频中的硬切点。

    Parameters
    ----------
    video_path
        输入视频路径
    min_scene_duration
        仅用于记录短镜头数量；真实硬切不会因时长过短而合并
    threshold
        内容变化敏感度，越低越敏感（默认27，范围0-255）

    Returns
    -------
    list of (start, end) 每个镜头的起止时间
    """
    try:
        from scenedetect import ContentDetector, SceneManager, open_video
    except ImportError:
        logger.error("PySceneDetect 未安装。请运行: pip install scenedetect[opencv]")
        raise

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    # 真实镜头边界不能仅仅因为片段较短而合并回去。保留检测器自身的
    # 抗抖间隔，避免人物快速运动被连续误判成多个硬切。
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)

    raw_scenes = scene_manager.get_scene_list()

    if not raw_scenes:
        duration = video.duration.get_seconds()
        return [(0.0, duration)]

    duration = video.duration.get_seconds()

    scenes = [
        (round(start.get_seconds(), 3), round(end.get_seconds(), 3))
        for start, end in raw_scenes
    ]
    short_count = sum(1 for start, end in scenes if end - start < min_scene_duration)
    if short_count:
        logger.info(
            "  保留 %d 个短硬切镜头（人物身份纯净优先，不按最短时长合并）",
            short_count,
        )

    # 确保最后一个镜头的结束时间不超过视频长度
    if scenes:
        scenes[-1] = (scenes[-1][0], round(min(scenes[-1][1], duration), 3))

    return scenes


# ── YOLO 人物检测 ────────────────────────────────────────────────────────

_person_model_gpu = None   # GPU 模型单例
_person_model_cpu = None   # CPU 模型单例

EDGE_PARTIAL_CONFIDENCE = 0.25
EDGE_PARTIAL_MIN_HEIGHT_RATIO = 0.35
EDGE_PARTIAL_MIN_AREA_RATIO = 0.025
EDGE_PARTIAL_MARGIN_RATIO = 0.03
DENSE_PERSON_SCAN_EDGE_SECONDS = 0.30
DENSE_PERSON_SCAN_FULL_SHOT_SECONDS = 2.0
YOLO_CUDA_BATCH_SIZE = 8
YOLO_CPU_BATCH_SIZE = 1


_device_cache: str | None = None

def _auto_detect_device() -> str:
    """自动检测可用设备（结果缓存，只打印一次）。"""
    global _device_cache
    if _device_cache is not None:
        return _device_cache
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda:0"
            gpu_name = torch.cuda.get_device_name(0)
            logger.info("检测到 GPU: %s，使用 CUDA 加速", gpu_name)
            _device_cache = device
            return device
    except ImportError:
        pass
    logger.info("未检测到 GPU，使用 CPU")
    _device_cache = "cpu"
    return "cpu"


def _get_yolo_model(device: str | None = None):
    """懒加载 YOLOv8n 模型。

    Parameters
    ----------
    device : str | None
        "cuda:0" / "cuda" → GPU 加速
        "cpu" → CPU
        None / "auto" → 自动检测
    """
    if device is None or device == "auto":
        device = _auto_detect_device()

    is_gpu = device.startswith("cuda")

    global _person_model_gpu, _person_model_cpu
    if is_gpu and _person_model_gpu is not None:
        return _person_model_gpu
    if not is_gpu and _person_model_cpu is not None:
        return _person_model_cpu

    try:
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        if is_gpu:
            model.to(device)
            _person_model_gpu = model
            logger.info("YOLOv8n 模型加载完成 (GPU)")
        else:
            _person_model_cpu = model
            logger.info("YOLOv8n 模型加载完成 (CPU)")
        return model
    except ImportError:
        logger.error("ultralytics 未安装。请运行: pip install ultralytics")
        raise


def detect_persons_in_frame(
    frame_path: str | Path,
    conf_threshold: float = 0.35,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """检测单帧中的人物。

    Parameters
    ----------
    frame_path
        帧图片路径
    conf_threshold
        YOLO 置信度阈值

    Returns
    -------
    list of dict:
        {bbox: [x1,y1,x2,y2], confidence: float, center: (cx,cy), area_ratio: float}
        按面积从大到小排列（最大的通常是主角）
    """
    model = _get_yolo_model(device)
    # 贴边、被截断的人物置信度通常低于完整人物。先用较低门槛取候选，
    # 再只对满足明显贴边形态的候选放宽最终阈值。
    inference_threshold = min(float(conf_threshold), 0.20)
    inference_threshold = min(float(conf_threshold), 0.20)
    results = model(str(frame_path), verbose=False, conf=inference_threshold)

    persons = []
    if results[0].boxes is not None:
        boxes = results[0].boxes
        img_w = results[0].orig_shape[1]
        img_h = results[0].orig_shape[0]

        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id != 0:  # YOLO class 0 = person
                continue
            conf = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            area_ratio = (w * h) / (img_w * img_h)
            edge_partial = _is_edge_partial_person(
                x1,
                y1,
                x2,
                y2,
                confidence=conf,
                image_width=img_w,
                image_height=img_h,
            )
            if conf < conf_threshold and not edge_partial:
                continue

            persons.append({
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
                "confidence": round(conf, 3),
                "center": (round((x1 + x2) / 2), round((y1 + y2) / 2)),
                "area_ratio": round(area_ratio, 4),
                "edge_partial": edge_partial,
            })

    # 按面积从大到小排
    persons.sort(key=lambda p: -p["area_ratio"])
    return _filter_nested_person_artifacts(persons)


def _yolo_batch_size(device: str | None = None, batch_size: int | None = None) -> int:
    if batch_size is not None:
        return max(1, int(batch_size))

    raw = os.environ.get("SP_YOLO_BATCH_SIZE")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning("SP_YOLO_BATCH_SIZE=%r invalid, using default batch size", raw)

    if device is None or device == "auto":
        device = _auto_detect_device()
    return YOLO_CUDA_BATCH_SIZE if str(device).startswith("cuda") else YOLO_CPU_BATCH_SIZE


def _persons_from_yolo_result(result: Any, conf_threshold: float) -> list[dict[str, Any]]:
    persons = []
    if result.boxes is not None:
        boxes = result.boxes
        img_w = result.orig_shape[1]
        img_h = result.orig_shape[0]

        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id != 0:  # YOLO class 0 = person
                continue
            conf = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            area_ratio = (w * h) / (img_w * img_h)
            edge_partial = _is_edge_partial_person(
                x1,
                y1,
                x2,
                y2,
                confidence=conf,
                image_width=img_w,
                image_height=img_h,
            )
            if conf < conf_threshold and not edge_partial:
                continue

            persons.append({
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
                "confidence": round(conf, 3),
                "center": (round((x1 + x2) / 2), round((y1 + y2) / 2)),
                "area_ratio": round(area_ratio, 4),
                "edge_partial": edge_partial,
            })

    persons.sort(key=lambda p: -p["area_ratio"])
    return _filter_nested_person_artifacts(persons)


def _filter_nested_person_artifacts(persons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop weak duplicate person boxes nested inside the main person.

    Product cards, hands, or high-contrast body regions can occasionally be
    detected as a second low-confidence person. Counting those boxes makes the
    splitter create tiny 1人/2人/1人 fragments inside an otherwise unchanged
    shot.
    """
    if len(persons) <= 1:
        return persons

    main = persons[0]
    filtered = [main]
    main_bbox = main.get("bbox")
    for person in persons[1:]:
        confidence = float(person.get("confidence", 0.0))
        area_ratio = float(person.get("area_ratio", 0.0))
        nested_ratio = _contained_ratio(person.get("bbox"), main_bbox)

        edge_partial = bool(person.get("edge_partial"))
        weak_nested = confidence < 0.55 and nested_ratio >= 0.80 and not edge_partial
        weak_small = confidence < 0.50 and area_ratio < 0.18 and not edge_partial
        if weak_nested or weak_small:
            continue
        filtered.append(person)

    return filtered


def _is_edge_partial_person(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    confidence: float,
    image_width: int,
    image_height: int,
) -> bool:
    if confidence < EDGE_PARTIAL_CONFIDENCE or image_width <= 0 or image_height <= 0:
        return False
    width_ratio = max(0.0, x2 - x1) / image_width
    height_ratio = max(0.0, y2 - y1) / image_height
    area_ratio = width_ratio * height_ratio
    margin_x = image_width * EDGE_PARTIAL_MARGIN_RATIO
    margin_y = image_height * EDGE_PARTIAL_MARGIN_RATIO
    touches_edge = (
        x1 <= margin_x
        or x2 >= image_width - margin_x
        or y1 <= margin_y
        or y2 >= image_height - margin_y
    )
    tall_fragment = height_ratio >= EDGE_PARTIAL_MIN_HEIGHT_RATIO and height_ratio >= width_ratio * 1.35
    return touches_edge and tall_fragment and area_ratio >= EDGE_PARTIAL_MIN_AREA_RATIO


def _contained_ratio(inner_bbox: Any, outer_bbox: Any) -> float:
    if not inner_bbox or not outer_bbox:
        return 0.0
    ix1, iy1, ix2, iy2 = [float(v) for v in inner_bbox]
    ox1, oy1, ox2, oy2 = [float(v) for v in outer_bbox]
    inner_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inner_area <= 0:
        return 0.0
    inter_w = max(0.0, min(ix2, ox2) - max(ix1, ox1))
    inter_h = max(0.0, min(iy2, oy2) - max(iy1, oy1))
    return (inter_w * inter_h) / inner_area


def detect_persons_in_frames(
    frame_paths: list[Path],
    conf_threshold: float = 0.35,
    device: str | None = None,
    batch_size: int | None = None,
) -> list[dict[str, Any]]:
    """批量检测多帧中的人物 → 返回每帧的检测结果。"""
    if not frame_paths:
        return []

    model = _get_yolo_model(device)
    inference_threshold = min(float(conf_threshold), 0.20)
    actual_batch_size = _yolo_batch_size(device, batch_size)
    logger.info("[YOLO batch] %d frames, batch=%d", len(frame_paths), actual_batch_size)

    yolo_results = model(
        [str(fp) for fp in frame_paths],
        verbose=False,
        conf=inference_threshold,
        batch=actual_batch_size,
    )

    results = []
    for fp, yolo_result in zip(frame_paths, yolo_results):
        persons = _persons_from_yolo_result(yolo_result, conf_threshold)
        results.append({
            "frame_path": str(fp),
            "person_count": len(persons),
            "persons": persons,
        })
    # 汇总日志
    counts = [r["person_count"] for r in results]
    logger.info(
        "[YOLO检测] %d帧: 人物数分布 min=%d max=%d (0人:%d, 1人:%d, 2+人:%d)",
        len(results),
        min(counts) if counts else 0,
        max(counts) if counts else 0,
        counts.count(0),
        counts.count(1),
        sum(1 for c in counts if c >= 2),
    )
    return results


def _extract_frames_at_times(
    video_path: str | Path,
    samples: list[tuple[Path, float]],
) -> list[tuple[Path, float]]:
    if not samples:
        return []

    from .ffmpeg_tools import extract_frame

    try:
        import cv2
    except ImportError:
        extracted = []
        for frame_path, t in samples:
            try:
                extract_frame(video_path, t, frame_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("FFmpeg frame extraction failed at %.3fs: %s", t, exc)
                continue
            if frame_path.exists() and frame_path.stat().st_size > 0:
                extracted.append((frame_path, t))
        return extracted

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("OpenCV failed to open video, falling back to per-frame FFmpeg extraction")
        extracted = []
        for frame_path, t in samples:
            try:
                extract_frame(video_path, t, frame_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("FFmpeg frame extraction failed at %.3fs: %s", t, exc)
                continue
            if frame_path.exists() and frame_path.stat().st_size > 0:
                extracted.append((frame_path, t))
        return extracted

    extracted: list[tuple[Path, float]] = []
    try:
        for frame_path, t in samples:
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t)) * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                logger.warning("OpenCV frame extraction failed at %.3fs, falling back to FFmpeg", t)
                try:
                    extract_frame(video_path, t, frame_path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("FFmpeg frame extraction failed at %.3fs: %s", t, exc)
                    continue
            else:
                ok = cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    logger.warning("Failed to write frame: %s", frame_path)
                    continue
            if frame_path.exists() and frame_path.stat().st_size > 0:
                extracted.append((frame_path, t))
            else:
                logger.warning("Frame file missing after extraction at %.3fs: %s", t, frame_path)
    finally:
        cap.release()

    return extracted


def refine_person_transition_frames(
    video_path: str | Path,
    shot_idx: int,
    frame_detections: list[dict[str, Any]],
    temp_dir: Path,
    conf_threshold: float = 0.35,
    device: str | None = None,
    yolo_batch_size: int | None = None,
    step: float = 0.05,
) -> list[dict[str, Any]]:
    """Add dense samples around person/no-person transitions for cleaner cuts."""
    if len(frame_detections) < 2:
        return frame_detections

    detections = sorted(frame_detections, key=lambda d: d.get("time", 0.0))
    known_times = {round(float(d.get("time", 0.0)), 3) for d in detections}
    added: list[dict[str, Any]] = []

    for left, right in zip(detections, detections[1:]):
        left_count = int(left.get("person_count", 0))
        right_count = int(right.get("person_count", 0))
        if left_count == right_count:
            continue

        start = float(left.get("time", 0.0))
        end = float(right.get("time", 0.0))
        if end - start <= step:
            continue

        samples: list[tuple[Path, float]] = []
        t = start + step
        while t < end - 1e-6:
            rounded_t = round(t, 3)
            if rounded_t in known_times:
                t += step
                continue

            frame_path = temp_dir / f"shot{shot_idx:03d}_refine_{rounded_t:.3f}s.jpg"
            samples.append((frame_path, rounded_t))
            t += step

        extracted = _extract_frames_at_times(video_path, samples)
        frame_paths = [frame_path for frame_path, _ in extracted]
        detections = detect_persons_in_frames(
            frame_paths,
            conf_threshold=conf_threshold,
            device=device,
            batch_size=yolo_batch_size,
        )
        for detection, (_, rounded_t) in zip(detections, extracted):
            persons = detection["persons"]
            added.append({
                "frame_path": detection["frame_path"],
                "person_count": len(persons),
                "persons": persons,
                "time": rounded_t,
                "diff_prev": 0.0,
                "refined": True,
            })
            known_times.add(rounded_t)

            if len(persons) != left_count:
                break

    if not added:
        return detections

    merged = detections + added
    merged.sort(key=lambda d: d.get("time", 0.0))
    return merged


# ── 第二层：人物进出细分 ──────────────────────────────────────────────


def subdivide_shot_by_person_presence(
    shot_start: float,
    shot_end: float,
    frame_detections: list[dict[str, Any]],
    min_sub_duration: float = 1.0,
    start_sources: list[str] | None = None,
    end_sources: list[str] | None = None,
) -> list[dict[str, Any]]:
    """在一个 PySceneDetect 切出的镜头内，按人物数量的变化再细分。

    原理：
      镜头内采样帧 → YOLO 检测每帧人物数 → 人物数量突变 → 在此处细分

    输出：
      子片段列表，每个包含 起止时间、人物数量、纯背景标记、边界来源
    """
    if start_sources is None:
        start_sources = []
    if end_sources is None:
        end_sources = []
    if not frame_detections:
        return [{
            "start": shot_start,
            "end": shot_end,
            "person_count": -1,  # 未知
            "is_pure_background": False,
            "max_person_bbox": None,
            "start_sources": list(start_sources),
            "end_sources": list(end_sources),
        }]

    # ── 找人物数量变化的边界（带来源标签）──────────────────────────
    # boundary_sources: {time: set(sources)}
    boundary_sources: dict[float, set[str]] = {
        shot_start: set(start_sources),
    }
    prev_count = int(frame_detections[0]["person_count"])
    prev_state = _person_count_state(prev_count)
    prev_time = float(frame_detections[0].get("time", shot_start))

    for i, det in enumerate(frame_detections[1:], 1):
        cur_count = int(det["person_count"])
        cur_state = _person_count_state(cur_count)
        cur_time = det.get("time", shot_start + i * (shot_end - shot_start) / len(frame_detections))

        if cur_state != prev_state:
            transient_multi = 2 in (prev_state, cur_state)
            logger.info(
                "[人物进出] %.2fs: person %d→%d (%.2fs ~ %.2fs之间)",
                (prev_time + cur_time) / 2, prev_count, cur_count,
                prev_time, cur_time,
            )
            boundary_time = round((prev_time + cur_time) / 2, 3)
            last_boundary = max(boundary_sources.keys())
            # 一闪而过的第二个人物同样会污染 SCAIL 身份轨迹。允许为这种
            # 1人→多人→1人变化保留帧级风险段；其他变化继续使用常规最短时长。
            required_duration = 0.02 if transient_multi else min_sub_duration
            if (
                boundary_time - last_boundary >= required_duration
                and shot_end - boundary_time >= required_duration
            ):
                sources = {"yolo"}
                if transient_multi:
                    sources.add("yolo_transient_multi")
                boundary_sources[boundary_time] = sources
            prev_state = cur_state

        # 如果之前一段时间稳定，但 diff_prev 很高（画面突变），也设边界
        elif det.get("diff_prev", 0) > 0.40:
            boundary_time = cur_time
            if boundary_time - max(boundary_sources.keys()) >= min_sub_duration:
                boundary_sources[boundary_time] = {"yolo"}

        prev_count = cur_count
        prev_time = cur_time

    boundary_sources[shot_end] = set(end_sources)
    boundaries = sorted(boundary_sources.keys())

    # 确保首尾是 shot_start / shot_end
    if boundaries[0] != shot_start:
        boundaries.insert(0, shot_start)
        boundary_sources[shot_start] = set(start_sources)
    if boundaries[-1] < shot_end:
        if shot_end - boundaries[-1] >= 0.3:
            boundaries.append(shot_end)
            boundary_sources[shot_end] = set(end_sources)
        else:
            boundaries[-1] = shot_end
            boundary_sources[shot_end] = boundary_sources.pop(boundaries[-1], set(end_sources))

    # ── 每段标记属性 ─────────────────────────────────────────────────
    sub_segments = []
    for start, end in zip(boundaries, boundaries[1:]):
        mid = (start + end) / 2
        # 找最接近中点的帧的检测结果
        segment_detections = [
            det for det in frame_detections
            if start - 1e-6 <= det.get("time", mid) <= end + 1e-6
        ] or frame_detections
        person_detections = [
            det for det in segment_detections
            if det.get("person_count", 0) > 0
        ]

        if person_detections:
            def person_score(det: dict[str, Any]) -> tuple[float, float, float, float]:
                persons = det.get("persons", [])
                max_area = max((p.get("area_ratio", 0.0) for p in persons), default=0.0)
                max_conf = max((p.get("confidence", 0.0) for p in persons), default=0.0)
                return (
                    float(det.get("person_count", 0)),
                    float(max_area),
                    float(max_conf),
                    -abs(det.get("time", mid) - mid),
                )

            closest = max(person_detections, key=person_score)
        else:
            closest = min(segment_detections, key=lambda d: abs(d.get("time", mid) - mid))
        person_count = closest["person_count"]
        persons = closest.get("persons", [])
        max_bbox = persons[0]["bbox"] if persons else None

        # 找到落在这个片段内的 YOLO 检测帧
        sub_detections = [
            det for det in frame_detections
            if start - 0.05 <= det.get("time", -1) <= end + 0.05
        ]
        if not sub_detections:
            sub_detections = [closest]

        sub_segments.append({
            "start": start,
            "end": end,
            "person_count": person_count,
            "is_pure_background": person_count == 0,
            "main_person_bbox": max_bbox,
            "all_persons": persons,
            "representative_frame": closest["frame_path"],
            "_all_frame_detections": sub_detections,  # 传给 Gemini 做精确切点
            "transient_multi_person": (
                person_count >= 2
                and end - start <= 0.5
                and (
                    "yolo_transient_multi" in boundary_sources.get(start, set())
                    or "yolo_transient_multi" in boundary_sources.get(end, set())
                )
            ),
            "start_sources": sorted(boundary_sources.get(start, set())),
            "end_sources": sorted(boundary_sources.get(end, set())),
        })

    return sub_segments


def _person_count_state(person_count: int) -> int:
    if person_count <= 0:
        return 0
    if person_count == 1:
        return 1
    return 2


def _shot_sample_times(
    shot_start: float,
    shot_end: float,
    *,
    sample_interval: float,
    fps: float,
    edge_window: float = DENSE_PERSON_SCAN_EDGE_SECONDS,
    full_scan_max_duration: float = DENSE_PERSON_SCAN_FULL_SHOT_SECONDS,
) -> list[float]:
    """Return regular samples plus frame-level samples where brief people matter.

    Short shots are scanned at source frame cadence. Longer shots keep regular
    samples and scan their first/last ``edge_window`` seconds densely, where
    remnants from adjacent shots most commonly appear.
    """
    start = float(shot_start)
    end = float(shot_end)
    duration = end - start
    if duration <= 0:
        return []

    source_fps = float(fps) if fps and fps > 0 else 25.0
    frame_step = 1.0 / source_fps
    safe_end = max(start, end - min(frame_step * 0.5, duration * 0.5))
    times: set[float] = {
        round(start, 3),
        round((start + end) / 2.0, 3),
        round(safe_end, 3),
    }

    interval = max(float(sample_interval), frame_step)
    regular_time = start + interval
    while regular_time < safe_end - 1e-6:
        times.add(round(regular_time, 3))
        regular_time += interval

    dense_ranges: list[tuple[float, float]]
    if duration <= full_scan_max_duration:
        dense_ranges = [(start, safe_end)]
    else:
        dense_ranges = [
            (start, min(safe_end, start + edge_window)),
            (max(start, end - edge_window), safe_end),
        ]

    for dense_start, dense_end in dense_ranges:
        if dense_end < dense_start:
            continue
        dense_time = dense_start
        while dense_time <= dense_end + 1e-6:
            times.add(round(dense_time, 3))
            dense_time += frame_step

    return sorted(t for t in times if start - 1e-6 <= t < end - 1e-6)


# ── 完整两层检测入口 ──────────────────────────────────────────────────────


def two_pass_segmentation(
    video_path: str | Path,
    sample_interval: float = 1.0,
    min_scene_duration: float = 0.8,
    min_sub_duration: float = 1.0,
    content_threshold: float = 27.0,
    yolo_conf_threshold: float = 0.35,
    device: str | None = None,
    yolo_batch_size: int | None = None,
    use_omnishotcut: bool = False,
    use_pyscene_detect: bool = True,
) -> dict[str, Any]:
    """完整镜头检测：OmniShotCut/PySceneDetect 候选 + YOLO 人物细分。

    Parameters
    ----------
    video_path
        输入视频路径
    sample_interval
        YOLO 检测的采样间隔（秒）
    min_scene_duration
        最短镜头时长
    min_sub_duration
        人物进出细分后的最短片段时长
    content_threshold
        PySceneDetect 内容敏感度
    yolo_conf_threshold
        YOLO 人物检测置信阈值
    device : str | None
        "cuda" / "cuda:0" / "cpu" / None(自动检测)

    Returns
    -------
    {
        "hard_cut_scenes": [(start, end), ...],           ← 第一层结果
        "sub_segments": [{start, end, person_count, ...}], ← 第二层结果
        "background_segments": [...],                      ← 纯背景片段
        "person_segments": [...],                          ← 有人片段
        "stats": {...},                                     ← 统计
    }
    """
    from .ffmpeg_tools import extract_frame, probe_video

    try:
        import cv2
    except ImportError:
        logger.error("opencv-python 未安装。请运行: pip install opencv-python")
        raise

    video_path = Path(video_path)
    meta = probe_video(video_path)
    duration = meta.duration

    # ── 第0层：OmniShotCut 候选镜头（可选）───────────────────────────
    omnishotcut_result: dict[str, Any] | None = None
    if use_omnishotcut:
        logger.info("第0层：OmniShotCut 语义镜头候选...")
        try:
            from .omnishotcut_detector import detect_omnishotcut_shots

            omnishotcut_result = detect_omnishotcut_shots(video_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[OmniShotCut] 跳过，回退 PySceneDetect: %s", exc)
            omnishotcut_result = None

    # ── 第一层：确定镜头候选 ──────────────────────────────────────────
    pyscene_cuts: list[tuple[float, float]] = []
    if use_pyscene_detect:
        logger.info("第一层：PySceneDetect 检测硬切点...")
        pyscene_cuts = detect_hard_cuts(
            video_path,
            min_scene_duration=min_scene_duration,
            threshold=content_threshold,
        )
        if pyscene_cuts:
            pyscene_cuts[-1] = (pyscene_cuts[-1][0], round(duration, 3))
        logger.info("  检测到 %d 个硬切镜头", len(pyscene_cuts))

    if omnishotcut_result and use_pyscene_detect:
        hard_cuts = fuse_shot_candidates(
            omnishotcut_result["shots"],
            pyscene_cuts,
            duration=duration,
            min_scene_duration=min_scene_duration,
        )
        logger.info(
            "[OmniShotCut] 与 PySceneDetect 融合: Omni=%d, PyScene=%d → %d 镜头",
            len(omnishotcut_result["shots"]),
            len(pyscene_cuts),
            len(hard_cuts),
        )
    elif omnishotcut_result:
        hard_cuts = [
            {
                "start": float(shot["start"]),
                "end": float(shot["end"]),
                "start_sources": ["omnishotcut"],
                "end_sources": ["omnishotcut"],
            }
            for shot in omnishotcut_result["shots"]
        ]
        logger.info("[OmniShotCut] 保留 %d 个镜头，并继续运行 YOLO 人数细分", len(hard_cuts))
    elif use_pyscene_detect:
        hard_cuts = [
            {"start": s, "end": e, "start_sources": ["pyscene"], "end_sources": ["pyscene"]}
            for s, e in pyscene_cuts
        ]
    else:
        logger.warning("镜头检测器未返回结果，整段视频继续运行 YOLO 人数细分")
        hard_cuts = [{
            "start": 0.0,
            "end": round(duration, 3),
            "start_sources": [],
            "end_sources": [],
        }]

    # ── 第二层：每个镜头内 YOLO 人物检测 + 细分 ───────────────────────
    logger.info("第二层：YOLO 人物检测 + 细分...")

    # 预加载 YOLO 模型
    _get_yolo_model(device)

    all_sub_segments: list[dict[str, Any]] = []
    temp_dir = Path(video_path).parent / "_temp_yolo_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        for shot_idx, shot in enumerate(hard_cuts):
            shot_start = shot["start"]
            shot_end = shot["end"]
            sample_times = _shot_sample_times(
                shot_start,
                shot_end,
                sample_interval=sample_interval,
                fps=meta.fps,
            )
            samples = [
                (
                    temp_dir / f"shot{shot_idx:03d}_s{sample_idx:03d}_{int(round(t * 1000)):09d}ms.jpg",
                    t,
                )
                for sample_idx, t in enumerate(sample_times)
            ]
            extracted = _extract_frames_at_times(video_path, samples)
            detections = detect_persons_in_frames(
                [frame_path for frame_path, _ in extracted],
                conf_threshold=yolo_conf_threshold,
                device=device,
                batch_size=yolo_batch_size,
            )
            frame_detections = []
            for detection, (_, t) in zip(detections, extracted):
                frame_detections.append({
                    "frame_path": detection["frame_path"],
                    "person_count": detection["person_count"],
                    "persons": detection["persons"],
                    "time": t,
                    "diff_prev": 0.0,
                })
            detected_time_keys = {int(round(float(item["time"]) * 1000)) for item in frame_detections}
            fallback_sample_times = [
                t for t in sample_times
                if int(round(float(t) * 1000)) not in detected_time_keys
            ]

            for sample_idx, t in enumerate(fallback_sample_times):
                frame_path = temp_dir / f"shot{shot_idx:03d}_s{sample_idx:03d}_{int(round(t * 1000)):09d}ms.jpg"
                try:
                    extract_frame(video_path, t, frame_path)
                    persons = detect_persons_in_frame(frame_path, conf_threshold=yolo_conf_threshold, device=device)
                    frame_detections.append({
                        "frame_path": str(frame_path),
                        "person_count": len(persons),
                        "persons": persons,
                        "time": t,
                        "diff_prev": 0.0,
                    })
                except Exception as e:
                    logger.warning("  帧检测失败 t=%.2f: %s", t, e)
                    continue

            # 加入像素差异信息（用于判断人物进出之外的变化）
            frame_detections = refine_person_transition_frames(
                video_path=video_path,
                shot_idx=shot_idx,
                frame_detections=frame_detections,
                temp_dir=temp_dir,
                conf_threshold=yolo_conf_threshold,
                device=device,
                yolo_batch_size=yolo_batch_size,
            )

            for i in range(1, len(frame_detections)):
                try:
                    prev_img = cv2.imread(frame_detections[i - 1]["frame_path"])
                    cur_img = cv2.imread(frame_detections[i]["frame_path"])
                    if prev_img is not None and cur_img is not None:
                        prev_img = cv2.resize(prev_img, (160, 284))
                        cur_img = cv2.resize(cur_img, (160, 284))
                        diff = np.mean(np.abs(cur_img.astype(np.float32) - prev_img.astype(np.float32))) / 255.0
                        frame_detections[i]["diff_prev"] = round(float(diff), 4)
                except Exception:
                    pass

            # 人物进出细分
            subs = subdivide_shot_by_person_presence(
                shot_start, shot_end,
                frame_detections,
                min_sub_duration=min_sub_duration,
                start_sources=shot.get("start_sources", []),
                end_sources=shot.get("end_sources", []),
            )
            for sub in subs:
                sub["shot_index"] = shot_idx
            all_sub_segments.extend(subs)

    finally:
        # 清理临时帧
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ── 分类统计 ──────────────────────────────────────────────────────
    bg_segments = [s for s in all_sub_segments if s["is_pure_background"]]
    person_segments = [s for s in all_sub_segments if not s["is_pure_background"]]

    stats = {
        "total_hard_cuts": len(hard_cuts),
        "total_sub_segments": len(all_sub_segments),
        "pure_background_count": len(bg_segments),
        "person_segment_count": len(person_segments),
        "background_ratio": round(len(bg_segments) / max(1, len(all_sub_segments)), 3),
        "multi_person_segments": len([s for s in person_segments if s["person_count"] >= 2]),
        "omnishotcut_enabled": bool(use_omnishotcut),
        "omnishotcut_count": len(omnishotcut_result["shots"]) if omnishotcut_result else 0,
        "pyscene_enabled": bool(use_pyscene_detect),
        "yolo_enabled": True,
        "transient_multi_person_segments": len([
            s for s in person_segments if s.get("transient_multi_person")
        ]),
    }

    return {
        "hard_cut_scenes": hard_cuts,
        "omnishotcut": omnishotcut_result,
        "sub_segments": all_sub_segments,
        "background_segments": bg_segments,
        "person_segments": person_segments,
        "stats": stats,
    }


def fuse_shot_candidates(
    primary_shots: list[dict[str, Any]],
    secondary_scenes: list[tuple[float, float]],
    duration: float,
    min_scene_duration: float = 0.8,
    tolerance: float = 0.20,
) -> list[dict[str, Any]]:
    """Fuse OmniShotCut shots with PySceneDetect hard cuts.

    OmniShotCut is treated as the primary proposal layer. PySceneDetect
    boundaries are added only when they are not already close to an Omni
    boundary, preserving the old detector as a safety net.

    Returns each scene as a dict with ``start_sources`` / ``end_sources``
    so downstream consumers can tell which detector(s) created each boundary.
    """
    # ── 收集带来源标签的边界点 ────────────────────────────────────────
    tagged: dict[float, set[str]] = {}

    # OmniShotCut 边界
    primary_boundaries = _boundaries_from_dict_shots(primary_shots, duration)
    for b in primary_boundaries:
        tagged.setdefault(round(b, 3), set()).add("omnishotcut")

    # PySceneDetect 边界 — 靠近已有边界的合并标签，否则独立添加
    for start, end in secondary_scenes:
        for boundary in (start, end):
            if boundary <= 0 or boundary >= duration:
                continue
            key = round(float(boundary), 3)
            match = None
            for existing in tagged:
                if abs(key - existing) <= tolerance:
                    match = existing
                    break
            if match is not None:
                tagged[match].add("pyscene")
            else:
                tagged.setdefault(key, set()).add("pyscene")

    # ── 确保 0.0 和 duration 存在 ─────────────────────────────────────
    if 0.0 not in tagged:
        tagged[0.0] = set()
    if duration not in tagged:
        tagged[duration] = set()

    # ── 合并过近的同源边界（tolerance 减半用于同源去重） ──────────────
    sorted_bounds = sorted(tagged.keys())
    merged_bounds: list[float] = []
    merged_sources: dict[float, set[str]] = {}
    tight_tolerance = tolerance / 2
    for b in sorted_bounds:
        if not merged_bounds or abs(b - merged_bounds[-1]) > tight_tolerance:
            merged_bounds.append(b)
            merged_sources[b] = tagged[b].copy()
        else:
            # 合并到前一个边界点，取平均值，并集来源
            prev = merged_bounds[-1]
            avg = round((prev + b) / 2, 3)
            merged_sources[avg] = merged_sources.pop(prev) | tagged[b]
            merged_bounds[-1] = avg

    # ── 构建带来源的片段 ──────────────────────────────────────────────
    scenes: list[dict[str, Any]] = []
    for start, end in zip(merged_bounds, merged_bounds[1:]):
        if end - start < min(min_scene_duration, 0.25):
            continue
        scenes.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "start_sources": sorted(merged_sources.get(start, set())),
            "end_sources": sorted(merged_sources.get(end, set())),
        })

    if scenes:
        scenes[-1]["end"] = round(duration, 3)
    if not scenes:
        # 保底：用 PySceneDetect 的结果
        for s, e in secondary_scenes:
            scenes.append({
                "start": round(s, 3),
                "end": round(e, 3),
                "start_sources": ["pyscene"],
                "end_sources": ["pyscene"],
            })
    return scenes


def _boundaries_from_dict_shots(shots: list[dict[str, Any]], duration: float) -> list[float]:
    boundaries = [0.0, duration]
    for shot in shots:
        boundaries.append(float(shot.get("start", 0.0)))
        boundaries.append(float(shot.get("end", duration)))
    return _merge_time_boundaries(boundaries, duration=duration, tolerance=0.05)


def _merge_time_boundaries(boundaries: list[float], duration: float, tolerance: float) -> list[float]:
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
