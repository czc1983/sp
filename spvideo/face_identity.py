"""
第三层：人脸身份追踪 —— 分镜内角色切换检测

两阶段定位：
  1. 粗检：采样帧 → 人脸嵌入比对 → 发现身份变化的帧区间
  2. 精确定位：在变化区间内二分搜索 → 精确到 ~0.05s

依赖：insightface (人脸检测 + 嵌入)
"""
from __future__ import annotations

import logging
import numpy as np
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_face_app = None


def _get_face_app():
    global _face_app
    if _face_app is not None:
        return _face_app

    import os
    try:
        from insightface.app import FaceAnalysis
        _face_app = FaceAnalysis(
            name="buffalo_s",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _face_app.prepare(ctx_id=0, det_size=(320, 320))
        logger.info("[人脸识别] InsightFace Buffalo-S 模型加载完成 (GPU)")
    except Exception:
        try:
            from insightface.app import FaceAnalysis as FaceAnalysisCPU
            _face_app = FaceAnalysisCPU(
                name="buffalo_s",
                providers=["CPUExecutionProvider"],
            )
            _face_app.prepare(ctx_id=-1, det_size=(320, 320))
            logger.info("[人脸识别] InsightFace Buffalo-S 模型加载完成 (CPU)")
        except Exception as e:
            logger.error("[人脸识别] InsightFace 加载失败: %s", e)
            raise
    return _face_app


def _read_image(image_path: str | Path) -> Any | None:
    import cv2
    import numpy as np
    path_str = str(image_path)
    try:
        data = np.fromfile(path_str, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        img = cv2.imread(path_str)
    return img if img is not None else None


def get_face_embedding(image_path: str | Path) -> np.ndarray | None:
    result = _get_face(image_path)
    return result["embedding"] if result else None


def get_face(image_path: str | Path) -> dict[str, Any] | None:
    """返回最大人脸的信息：embedding + bbox"""
    import cv2
    import numpy as np
    path_str = str(image_path)
    try:
        data = np.fromfile(path_str, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        img = cv2.imread(path_str)
    if img is None:
        return None
    h, w = img.shape[:2]
    app = _get_face_app()
    faces = app.get(img)
    if not faces:
        return None
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return {
        "embedding": best.normed_embedding,
        "bbox": [round(float(best.bbox[0]) / w, 4), round(float(best.bbox[1]) / h, 4),
                 round(float(best.bbox[2]) / w, 4), round(float(best.bbox[3]) / h, 4)],
    }


def _get_face(image_path: str | Path) -> dict[str, Any] | None:
    import cv2
    import numpy as np
    path_str = str(image_path)
    try:
        data = np.fromfile(path_str, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        img = cv2.imread(path_str)
    if img is None:
        return None
    h, w = img.shape[:2]
    app = _get_face_app()
    faces = app.get(img)
    if not faces:
        return None
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return {
        "embedding": best.normed_embedding,
        "bbox": [round(float(best.bbox[0]) / w, 4), round(float(best.bbox[1]) / h, 4),
                 round(float(best.bbox[2]) / w, 4), round(float(best.bbox[3]) / h, 4)],
    }


def embedding_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return float(np.dot(emb1, emb2))


# ═══════════════════════════════════════════════════════════════════
# 阶段 1：粗检 — 在采样帧中发现身份变化的区间
# ═══════════════════════════════════════════════════════════════════

def detect_identity_changes(
    frame_paths: list[str],
    frame_times: list[float],
    similarity_threshold: float = 0.55,
) -> list[dict[str, Any]]:
    """粗检身份切换区间。"""
    if len(frame_paths) < 2:
        logger.info("[人脸识别] 帧数<%d，跳过", 2)
        return []

    embeddings = []
    face_flags = []
    for fp in frame_paths:
        emb = get_face_embedding(fp)
        embeddings.append(emb)
        face_flags.append(1 if emb is not None else 0)

    logger.info(
        "[人脸识别] %d帧: 有脸=%d 无脸=%d 相似度阈值=%.2f",
        len(frame_paths), sum(face_flags), len(face_flags) - sum(face_flags),
        similarity_threshold,
    )

    anchor_emb = None
    anchor_idx = -1
    for i, emb in enumerate(embeddings):
        if emb is not None:
            anchor_emb = emb
            anchor_idx = i
            break

    if anchor_emb is None:
        logger.info("[人脸识别] 所有帧都没检测到正面人脸 → 无法判断身份")
        return []

    logger.info("[人脸识别] 锚帧: %.2fs (首张有人脸的帧)", frame_times[anchor_idx])

    changes = []
    last_emb = anchor_emb
    last_idx = anchor_idx

    for i in range(anchor_idx + 1, len(embeddings)):
        cur_emb = embeddings[i]
        if cur_emb is None:
            logger.info("[人脸识别]  %.2fs: 无人脸 → 保持%ss的锚定身份", frame_times[i], frame_times[last_idx])
            continue

        sim = embedding_similarity(last_emb, cur_emb)
        same = sim >= similarity_threshold
        logger.info(
            "[人脸识别]  %.2fs vs %.2fs: 相似度=%.3f %s",
            frame_times[last_idx], frame_times[i], sim,
            "→ 同一人" if same else "→ 可能不同人！",
        )

        if not same:
            # 最后一帧 → 没有下一帧验证
            if i + 1 >= len(embeddings):
                logger.info("[人脸识别]  %.2fs: 无下一帧验证 → 仍标记为切换", frame_times[i])
            # 确认下一帧也不同于旧身份
            elif embeddings[i + 1] is not None:
                next_sim = embedding_similarity(last_emb, embeddings[i + 1])
                if next_sim >= similarity_threshold:
                    logger.info("[人脸识别]  %.2fs: 单帧波动(下一帧相似度=%.3f) → 跳过", frame_times[i], next_sim)
                    continue

            logger.info("[人脸识别]  ✅ 身份切换: %.2fs(角色%d) → %.2fs(角色%d)",
                         frame_times[last_idx], len(changes), frame_times[i], len(changes) + 1)
            changes.append({
                "prev_time": frame_times[last_idx],
                "new_time": frame_times[i],
                "prev_emb": last_emb,
                "new_emb": cur_emb,
                "similarity": round(sim, 3),
            })
            last_emb = cur_emb
            last_idx = i

    if not changes:
        logger.info("[人脸识别] 结论: 未发现身份切换（%d帧间相似度均>%.2f）",
                     len(embeddings) - 1, similarity_threshold)
    else:
        logger.info("[人脸识别] 结论: 发现%d次身份切换", len(changes))
    return changes


def _extract_time(filepath: str) -> float:
    """从文件名提取时间戳"""
    try:
        stem = Path(filepath).stem
        return float(stem.split("_")[-1].replace("s", ""))
    except (ValueError, IndexError):
        return 0.0


# ═══════════════════════════════════════════════════════════════════
# 阶段 2：精确定位 — 在变化区间内二分搜索
# ═══════════════════════════════════════════════════════════════════

def refine_boundary(
    video_path: str | Path,
    t_a: float,
    t_b: float,
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    extract_fn: Callable[[float], str],
    precision: float = 0.05,
    max_iterations: int = 10,
) -> float:
    """二分搜索精确定位身份切换时刻。"""
    if t_b - t_a <= precision:
        return (t_a + t_b) / 2.0

    lo, hi = t_a, t_b
    lo_emb, hi_emb = emb_a, emb_b

    for iteration in range(max_iterations):
        if hi - lo <= precision:
            break

        mid = (lo + hi) / 2.0
        mid_path = extract_fn(mid)
        if mid_path is None:
            break

        mid_emb = get_face_embedding(mid_path)

        if mid_emb is None:
            lo = mid
            continue

        sim_to_a = embedding_similarity(mid_emb, lo_emb)
        sim_to_b = embedding_similarity(mid_emb, hi_emb)

        if sim_to_a >= sim_to_b:
            lo = mid
            lo_emb = mid_emb
        else:
            hi = mid
            hi_emb = mid_emb

    result = (lo + hi) / 2.0
    logger.info("[人脸识别] 精确切点: %.3fs (二分%d次, 精度±%.3fs)", result, iteration + 1, (hi - lo) / 2)
    return result


# ═══════════════════════════════════════════════════════════════════
# 一站式：粗检 + 精确定位 + 切分
# ═══════════════════════════════════════════════════════════════════

def split_by_identity(
    video_path: str | Path,
    segment_start: float,
    segment_end: float,
    frame_paths: list[str],
    frame_times: list[float],
    extract_fn: Callable[[float], str] | None = None,
    similarity_threshold: float = 0.55,
    refine_precision: float = 0.05,
) -> list[dict[str, Any]]:
    """按人脸身份变化切分子片段（粗检 + 精确定位）。"""
    if not frame_paths or len(frame_paths) < 2:
        return [{"start": segment_start, "end": segment_end, "identity_id": 0}]

    # 阶段 1：粗检
    changes = detect_identity_changes(frame_paths, frame_times, similarity_threshold)

    if not changes:
        return [{"start": segment_start, "end": segment_end, "identity_id": 0}]

    # 阶段 2：精确定位每个切换点
    refined_cuts = []
    for ch in changes:
        t_prev = ch["prev_time"]
        t_new = ch["new_time"]
        emb_prev = ch["prev_emb"]
        emb_new = ch["new_emb"]

        if extract_fn and (t_new - t_prev) > refine_precision:
            precise = refine_boundary(
                video_path=video_path,
                t_a=t_prev,
                t_b=t_new,
                emb_a=emb_prev,
                emb_b=emb_new,
                extract_fn=extract_fn,
                precision=refine_precision,
            )
            cut_time = precise
        else:
            cut_time = (t_prev + t_new) / 2.0

        refined_cuts.append(round(cut_time, 3))

    # 构建切点+子片段
    cut_points = [segment_start] + refined_cuts
    if cut_points[-1] < segment_end - 0.3:
        cut_points.append(segment_end)
    else:
        cut_points[-1] = segment_end

    MIN_DUR = 0.2
    raw = []
    for st, en in zip(cut_points, cut_points[1:]):
        if en - st < MIN_DUR and raw:
            raw[-1] = (raw[-1][0], en)
        else:
            raw.append((st, en))

    sub_segments = []
    for i, (st, en) in enumerate(raw):
        sub_segments.append({
            "start": round(st, 3),
            "end": round(en, 3),
            "identity_id": i,
        })

    return sub_segments
