from __future__ import annotations

import logging
from typing import Any

from .features import summarize_features
from .models import FrameFeatures, Segment

logger = logging.getLogger(__name__)

TECH_BY_TYPE = {
    "with_human": "有人：优先走人物迁移/driver/口播复刻路线，后续提取 pose、face、hand。",
    "without_human": "无人：优先走产品图生视频、静图推拉、图文动效或素材重拍路线。",
}


def classify_segment(
    segment_id: str,
    start: float,
    end: float,
    frames: list[FrameFeatures],
    yolo_info: dict[str, Any] | None = None,
) -> Segment:
    """分类片段。

    yolo_info 可选，来自 scene_detector 的 YOLO 检测结果：
        {person_count, is_pure_background, main_person_bbox, all_persons}
    传入后优先用 YOLO 结果，不再靠肤色猜测。
    """
    summary = summarize_features(frames)
    detected: list[str] = []
    notes: list[str] = []
    transient_multi_person = bool(yolo_info and yolo_info.get("transient_multi_person"))

    # ── 优先使用 YOLO 检测结果 ──────────────────────────────────────
    if yolo_info and yolo_info.get("person_count", -1) >= 0:
        person_count = yolo_info["person_count"]
        has_human = person_count > 0
        confidence = 0.92 if person_count > 0 else 0.90

        if has_human:
            segment_type = "with_human"
            detected.append(f"YOLO: {person_count}人")
            if person_count >= 2:
                detected.append(f"多人同框({person_count}人)")
                notes.append("多人场景需分别替换角色，注意人物分割质量")
            if transient_multi_person:
                detected.append("短暂多人入镜")
                notes.append("短暂多人入镜，禁止按普通单人片段转绘；请保留原帧或单独处理")
            if yolo_info.get("main_person_bbox"):
                bbox = yolo_info["main_person_bbox"]
                area = yolo_info.get("all_persons", [{}])[0].get("area_ratio", 0) if yolo_info.get("all_persons") else 0
                detected.append(f"主角面积占比:{area:.2%}")
                if area > 0.40:
                    notes.append("人物占比大，特写/近景镜头，口型同步要求高")
                elif area < 0.08:
                    notes.append("人物占比小，远景，动作迁移可能精度下降")
            for p in yolo_info.get("all_persons", []):
                if p.get("confidence", 0) < 0.50:
                    notes.append(f"低置信度人物({p['confidence']:.0%})，建议人工确认")
        else:
            segment_type = "without_human"
            detected.append("YOLO: 无人")
            if yolo_info.get("is_pure_background"):
                detected.append("纯背景")
                notes.append("纯背景镜头——只需替换场景，不需要角色处理")
    else:
        # ── 降级：用像素肤色判断 ─────────────────────────────────────
        blue = summary["blue_ratio"]
        skin = summary["skin_ratio"]
        center_skin = summary["center_skin_ratio"]
        white = summary["white_ratio"]
        edge = summary["edge_score"]

        human_frames = [frame for frame in frames if has_human_presence(frame)]
        human_ratio = len(human_frames) / len(frames) if frames else 0.0
        max_center_skin = max((frame.center_skin_ratio for frame in frames), default=0.0)
        max_skin = max((frame.skin_ratio for frame in frames), default=0.0)
        has_human = (
            human_ratio >= 0.25
            or center_skin > 0.18
            or (max_center_skin > 0.30 and max_skin > 0.18)
        )
        product_like = (
            (center_skin < 0.16 and (white > 0.18 or edge > 0.095))
            or (center_skin < 0.14 and blue > 0.045)
            or (white > 0.28 and edge > 0.075)
        )

        if has_human:
            segment_type = "with_human"
            detected.append("pixel:human_present")
            if blue > 0.08 or white > 0.20 or edge > 0.105:
                detected.append("overlay_or_product_may_coexist")
            confidence = min(0.72, 0.50 + human_ratio * 0.22 + max_center_skin * 0.30 + max_skin * 0.12)
        else:
            segment_type = "without_human"
            detected.append("pixel:no_human")
            if product_like:
                detected.append("product_or_document_candidate")
            if blue > 0.22:
                detected.append("large_blue_layout")
            confidence = min(0.70, 0.52 + (1 - min(1.0, human_ratio)) * 0.18 + white * 0.2 + edge * 0.35)

        notes.append("⚠ 像素肤色判断（未启用YOLO），可能不准")
        if confidence < 0.55:
            notes.append("Needs human review; pixel-level classifier is low confidence.")

    # ── 通用检查 ────────────────────────────────────────────────────
    if 5.8 < (end - start):
        notes.append("片段较长，生成前考虑再细分")
    if segment_type == "with_human":
        notes.append("后续需提取 body pose、hand pose、face landmarks")
    else:
        notes.append("无人片段，不需要人物迁移")

    recommended_tech = TECH_BY_TYPE[segment_type]
    if transient_multi_person:
        recommended_tech = "短暂多人风险段：保留原帧，或为露出人物单独配置角色后再转绘。"

    return Segment(
        segment_id=segment_id,
        start=start,
        end=end,
        segment_type=segment_type,
        confidence=round(confidence, 3),
        detected=detected,
        recommended_tech=recommended_tech,
        needs_ai_driver=segment_type == "with_human",
        needs_manual_check=confidence < 0.62 or transient_multi_person,
        notes=notes,
        person_count=yolo_info.get("person_count", -1) if yolo_info else -1,
        transient_multi_person=transient_multi_person,
    )


def classify_from_sub_segment(
    segment_id: str,
    sub_seg: dict[str, Any],
    frames: list[FrameFeatures],
) -> Segment:
    """从 two_pass_segmentation 的 sub_segment 直接生成 Segment。

    这是推荐入口：直接用 YOLO 结果，不走像素肤色猜测。
    """
    yolo_info = {
        "person_count": sub_seg.get("person_count", -1),
        "is_pure_background": sub_seg.get("is_pure_background", False),
        "main_person_bbox": sub_seg.get("main_person_bbox"),
        "all_persons": sub_seg.get("all_persons", []),
        "transient_multi_person": sub_seg.get("transient_multi_person", False),
    }
    return classify_segment(
        segment_id=segment_id,
        start=sub_seg["start"],
        end=sub_seg["end"],
        frames=frames,
        yolo_info=yolo_info,
    )


def has_human_presence(frame: FrameFeatures) -> bool:
    """像素级人物检测（降级方案）。"""
    strong_center_human = frame.center_skin_ratio >= 0.14 and frame.skin_ratio >= 0.10
    weak_layout_human = (
        frame.center_skin_ratio >= 0.10
        and frame.skin_ratio >= 0.12
        and frame.white_ratio <= 0.19
        and frame.blue_ratio <= 0.04
    )
    return strong_center_human or weak_layout_human
