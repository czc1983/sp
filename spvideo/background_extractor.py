"""纯背景提取：从视频镜头中提取干净的背景图。

三种策略（按优先级）：
  1. 直接取无人帧 —— 镜头里有没有人物的帧 → 直接就是纯背景
  2. 中值叠加法 —— 整个镜头一直有人 → 取多帧中位像素，人物被"抹掉"
  3. LaMa 修复 —— 中值法仍有残留 → 用 AI 修复模型补全（需额外安装）

原理：
  人物在画面中移动，背景不变 → 取多帧的中位像素值 →
  每帧人物在不同位置 → 人物像素被中值过滤掉 → 只剩背景
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def extract_background_from_shot(
    video_path: str | Path,
    start: float,
    end: float,
    person_detections: list[dict[str, Any]],
    output_path: str | Path | None = None,
    method: str = "auto",
    num_samples: int = 15,
) -> dict[str, Any]:
    """从一段镜头中提取纯背景图。

    Parameters
    ----------
    video_path
        原视频路径
    start, end
        镜头起止时间（秒）
    person_detections
        该镜头内 YOLO 检测结果 [{time, person_count, frame_path}, ...]
    output_path
        输出图片路径（可选）
    method
        "auto" → 先找无人帧，没有则中值叠加
        "median" → 强制中值叠加
        "first_empty" → 只取第一张无人帧
    num_samples
        中值叠加时的采样帧数

    Returns
    -------
    {
        "background_path": 输出图片路径,
        "method_used": "empty_frame" | "median_blend" | "lama",
        "confidence": 0.0~1.0,
        "notes": ""
    }
    """
    from .ffmpeg_tools import extract_frame

    video_path = Path(video_path)
    duration = end - start

    # ── 策略1：找无人帧 ──────────────────────────────────────────────
    empty_frames = [
        d for d in person_detections
        if d.get("person_count", 1) == 0
    ]

    if empty_frames and method in ("auto", "first_empty"):
        best = empty_frames[len(empty_frames) // 2]  # 取中间那一帧
        frame_path = best.get("frame_path")
        if frame_path and Path(frame_path).exists():
            if output_path:
                _copy_or_save(frame_path, output_path)
            return {
                "background_path": str(frame_path),
                "method_used": "empty_frame",
                "confidence": 0.95,
                "notes": f"使用无人帧（{len(empty_frames)} 帧可选）",
            }

    # ── 策略2：中值叠加 ──────────────────────────────────────────────
    if method in ("auto", "median"):
        result = _median_blend_background(
            video_path, start, end,
            num_samples=num_samples,
            output_path=output_path,
        )
        if result:
            return result

    # ── 策略3：取中间帧兜底 ──────────────────────────────────────────
    mid_time = (start + end) / 2
    fallback_path = output_path or Path(str(video_path).rsplit(".", 1)[0] + f"_bg_{start:.0f}.jpg")
    fallback_path = Path(fallback_path)
    extract_frame(video_path, mid_time, fallback_path)

    return {
        "background_path": str(fallback_path),
        "method_used": "fallback_mid_frame",
        "confidence": 0.15,
        "notes": "无无人帧，中值法失败，取中间帧兜底",
    }


def _median_blend_background(
    video_path: Path,
    start: float,
    end: float,
    num_samples: int = 15,
    output_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """中值叠加法提取背景：多帧取中位像素值，消除移动的人物。

    原理：人物在镜头里移动，每帧位置不同。
         取 N 帧，每个像素位置取中值 → 背景像素稳定（总是同一个值）→ 被选中
         人物像素每帧不同位置 → 被淘汰。
    """
    from .ffmpeg_tools import extract_frame

    try:
        import cv2
    except ImportError:
        logger.warning("opencv 未安装，跳过中值叠加")
        return None

    duration = end - start
    temp_dir = Path(video_path).parent / "_temp_median_bg"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 均匀采样
        frames = []
        for i in range(num_samples):
            t = start + (duration * i / (num_samples - 1)) if num_samples > 1 else start + duration / 2
            t = min(t, end - 0.05)
            frame_path = temp_dir / f"bg_sample_{i:02d}.jpg"
            try:
                extract_frame(video_path, t, frame_path)
                if frame_path.stat().st_size > 100:  # 确保不是空文件
                    frames.append(frame_path)
            except Exception as e:
                logger.debug("  采样帧 %.2f 失败: %s", t, e)

        if len(frames) < 3:
            return None

        # 读取所有帧并统一尺寸
        images = []
        target_size = None
        for fp in frames:
            img = cv2.imread(str(fp))
            if img is not None:
                if target_size is None:
                    target_size = (img.shape[1], img.shape[0])
                img = cv2.resize(img, target_size)
                images.append(img)

        if len(images) < 3:
            return None

        # ── 中值叠加：每个像素位置取中值 ────────────────────────────
        stack = np.stack(images, axis=0)  # (N, H, W, 3)
        median = np.median(stack, axis=0).astype(np.uint8)

        # 保存
        out = output_path or Path(str(video_path).rsplit(".", 1)[0] + f"_bg_{start:.0f}_{end:.0f}.jpg")
        out = Path(out)
        cv2.imwrite(str(out), median)

        return {
            "background_path": str(out),
            "method_used": "median_blend",
            "confidence": 0.60,
            "notes": f"中值叠加（{len(images)}帧采样，{num_samples}帧采集）",
        }

    except Exception as e:
        logger.warning("中值叠加失败: %s", e)
        return None
    finally:
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def batch_extract_backgrounds(
    video_path: str | Path,
    sub_segments: list[dict[str, Any]],
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    """批量提取所有镜头的背景图。

    Parameters
    ----------
    video_path
        原视频路径
    sub_segments
        two_pass_segmentation 返回的 sub_segments
    output_dir
        背景图输出目录

    Returns
    -------
    每个镜头的背景提取结果
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, seg in enumerate(sub_segments):
        bg_path = output_dir / f"bg_{i:03d}_{seg['start']:.0f}_{seg['end']:.0f}.jpg"

        # 构建 person_detections（格式适配）
        detections = [{
            "time": seg["start"],
            "person_count": seg["person_count"],
            "frame_path": seg.get("representative_frame", ""),
        }]

        result = extract_background_from_shot(
            video_path,
            seg["start"],
            seg["end"],
            person_detections=detections,
            output_path=bg_path,
            method="first_empty" if seg["is_pure_background"] else "median",
        )
        result["segment_start"] = seg["start"]
        result["segment_end"] = seg["end"]
        result["is_pure_background_segment"] = seg["is_pure_background"]
        results.append(result)

    return results


def _copy_or_save(src: str, dst: str | Path) -> None:
    """将图片从 src 复制/保存到 dst。"""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        img = Image.open(src)
        img.save(str(dst))
    except Exception:
        import shutil
        shutil.copy2(src, dst)
