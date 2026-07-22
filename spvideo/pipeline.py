from __future__ import annotations

import json
import math
import os
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from .classifier import classify_from_sub_segment, classify_segment
from .features import analyze_frame
from .ffmpeg_tools import cut_segment, extract_frame, extract_frames_bulk, probe_video
from .gemini_analyzer import (
    GENERATION_DIR_MAP,
    GENERATION_ROUTE_LABELS,
    SCENE_TYPE_LABELS,
    GeminiClient,
    create_client,
)
from .models import FrameFeatures, Segment, ensure_dir, safe_stem
from .report import make_contact_sheet, make_html_report


CATEGORY_DIRS = {
    "with_human": "01_有人物",
    "without_human": "02_无人物",
}
ALL_CLIPS_DIRNAME = "00_all_mp4_clips"


def create_project_dirs(project_dir: str | Path) -> dict[str, Path]:
    root = ensure_dir(project_dir)
    dirs = {
        "root": root,
        "source": ensure_dir(root / "00_原始视频"),
        "probe": ensure_dir(root / "01_分析探针"),
        "frames": ensure_dir(root / "01_分析探针" / "frames_1fps"),
        "segments": ensure_dir(root / "02_分镜片段"),
        "all_clips": ensure_dir(root / "02_分镜片段" / ALL_CLIPS_DIRNAME),
        "ai_inputs": ensure_dir(root / "03_AI输入素材"),
        "ai_outputs": ensure_dir(root / "04_AI输出成片"),
        "edit": ensure_dir(root / "05_剪辑合成"),
        "export": ensure_dir(root / "06_最终导出"),
    }
    for dirname in CATEGORY_DIRS.values():
        ensure_dir(dirs["segments"] / dirname)
    # Gemini 生成路线目录
    for dirname in GENERATION_DIR_MAP.values():
        ensure_dir(dirs["segments"] / dirname)
    for dirname in (
        "driving_video", "pose_maps", "face_maps", "hand_maps",
        "masks", "background_refs", "character_refs", "product_refs",
    ):
        ensure_dir(dirs["ai_inputs"] / dirname)
    return dirs


def run_segmentation_v2(
    video_path: str | Path,
    project_dir: str | Path,
    sample_interval: float = 1.0,
    min_segment_duration: float = 1.0,
    max_segment_duration: float = 6.0,
    export_video: bool = True,
    extract_backgrounds: bool = False,
    yolo_conf_threshold: float = 0.35,
    device: str | None = None,
    yolo_batch_size: int | None = None,
    gemini_api_key: str | None = None,
    gemini_base_url: str | None = None,
    gemini_model: str | None = None,
    gemini_identity_concurrency: int = 10,
    use_omnishotcut: bool = False,
    use_scene_detect: bool = True,
    use_face_id: bool = True,
    use_visual_model: bool = True,
    use_sam3_finalize: bool = False,
    use_visual_merge: bool | None = None,
    use_pre_director: bool = True,
    pre_director_api_key: str | None = None,
    pre_director_base_url: str | None = None,
    pre_director_model: str | None = None,
) -> dict[str, object]:
    """多层切割管线：OmniShotCut/PySceneDetect + YOLO + 身份校验。

    OmniShotCut 作为候选镜头层；Gemini 作为后置身份层，解决
    "单镜头换角色"的问题。
    """
    from .background_extractor import batch_extract_backgrounds
    from .scene_detector import two_pass_segmentation
    from .sam3_finalizer import finalize_segments_with_sam3
    from .visual_merge import merge_visually_similar_segments
    from .pre_director import (
        analyze_pre_director,
        apply_pre_director_boundaries,
        collapse_transient_person_jitter,
        semantic_scene_for_time,
    )

    video_path = Path(video_path)
    dirs = create_project_dirs(project_dir)
    # ── 1. 探针 ──────────────────────────────────────────────────────
    meta = probe_video(video_path)
    (dirs["source"] / "source_pointer.json").write_text(
        json.dumps({"source_path": str(video_path), "note": "Original video is referenced, not copied."},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (dirs["source"] / "original_meta.json").write_text(
        json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 报告/预导演只需要中频采样；高频人物状态检测由 two-pass 单独负责。
    pre_director_plan: dict[str, object] | None = None
    pre_director_future: Future[dict[str, object]] | None = None
    pre_director_executor: ThreadPoolExecutor | None = None
    if use_pre_director:
        logger.info("第1层：全局预导演已提交后台（上传原视频 URL，与本地抽帧/技术切分并行）...")

        def run_pre_director_task() -> dict[str, object]:
            try:
                return analyze_pre_director(
                    [],
                    video_path=video_path,
                    has_audio=bool(meta.audio_codec),
                    duration=meta.duration,
                    output_path=dirs["probe"] / "pre_director.json",
                    api_key=str(pre_director_api_key or ""),
                    base_url=str(pre_director_base_url or ""),
                    model=str(pre_director_model or ""),
                    on_progress=logger.info,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[预导演] 分析失败，继续技术切分: %s", exc)
                failed: dict[str, object] = {
                    "status": "failed",
                    "error": str(exc),
                    "scenes": [],
                    "boundary_hints": [],
                }
                (dirs["probe"] / "pre_director.json").write_text(
                    json.dumps(failed, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return failed

        pre_director_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pre-director")
        pre_director_future = pre_director_executor.submit(run_pre_director_task)
        logger.info("[并行] 预导演已开始上传/读原视频，本地马上开始抽帧")
    else:
        logger.info("[预导演] 已跳过 — 用户关闭")

    report_floor = 0.1 if meta.duration <= 10 else 0.25 if meta.duration <= 60 else 0.5
    report_sample_interval = max(float(sample_interval), report_floor)
    if report_sample_interval > sample_interval:
        logger.info(
            "[抽帧] 技术检测保持 %.2fs 精度；报告/预导演使用 %.2fs 采样以节省时间",
            sample_interval,
            report_sample_interval,
        )
    frames = sample_frames(video_path, meta.duration, dirs["frames"], report_sample_interval)
    logger.info("[抽帧] 完成: %d 帧, 间隔 %.1fs", len(frames), report_sample_interval)

    # ── 3. 场景检测 ─────────────────────────────────────────────
    # SAM3 needs a YOLO subject box as its point prompt. When the user enables
    # layer 6 alone, still run the two-pass detector as a single full-video shot.
    if use_scene_detect or use_omnishotcut or use_sam3_finalize:
        logger.info("╔═══════════════════════════════════════════")
        if use_omnishotcut and use_scene_detect:
            logger.info("║ 第2层: OmniShotCut 镜头候选")
            logger.info("║ 第3层: PySceneDetect 硬切 + YOLO 人物状态")
        elif use_omnishotcut:
            logger.info("║ 第2层: OmniShotCut 镜头候选")
            logger.info("║ 第3层: YOLO 人物状态")
        else:
            logger.info("║ 第3层: PySceneDetect 硬切 + YOLO 人物状态")
        logger.info("╚═══════════════════════════════════════════")
        seg_result = two_pass_segmentation(
            video_path,
            sample_interval=sample_interval,
            min_scene_duration=min_segment_duration,
            min_sub_duration=min_segment_duration,
            yolo_conf_threshold=yolo_conf_threshold,
            device=device,
            yolo_batch_size=yolo_batch_size,
            use_omnishotcut=use_omnishotcut,
            use_pyscene_detect=use_scene_detect,
        )
        sub_segments = seg_result["sub_segments"]
        stats = seg_result["stats"]
        logger.info(
            "[镜头检测] %d 个镜头 → %d 子片段（%d 纯背景 + %d 有人）",
            stats["total_hard_cuts"],
            stats["total_sub_segments"],
            stats["pure_background_count"],
            stats["person_segment_count"],
        )
        for s in sub_segments:
            logger.info("  ├ 片段 [%.1f-%.1f] person=%s bg=%s",
                         s["start"], s["end"],
                         s.get("person_count", "?"),
                         s.get("is_pure_background", "?"))
        (dirs["probe"] / "two_pass_result.json").write_text(
            json.dumps(seg_result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    else:
        logger.info("[场景检测] 已跳过 — 整段视频作为一个片段")
        sub_segments = [{
            "start": 0.0, "end": meta.duration,
            "person_count": -1, "is_pure_background": False,
            "shot_index": 0,
        }]
        stats = {"total_hard_cuts": 0, "total_sub_segments": 1,
                 "pure_background_count": 0, "person_segment_count": 1}
        seg_result = {
            "hard_cut_scenes": [],
            "omnishotcut": None,
            "sub_segments": sub_segments,
            "background_segments": [],
            "person_segments": sub_segments,
            "stats": stats,
        }

    if pre_director_future is not None:
        wait_started = time.monotonic()
        if not pre_director_future.done():
            logger.info("[并行汇合] 本地技术切分已完成，正在等待全局预导演结果...")
        pre_director_plan = pre_director_future.result()
        logger.info("[并行汇合] 预导演结果已就绪，额外等待 %.1fs", time.monotonic() - wait_started)
        if pre_director_executor is not None:
            pre_director_executor.shutdown(wait=True)

    if pre_director_plan and pre_director_plan.get("boundary_hints"):
        before = len(sub_segments)
        sub_segments = apply_pre_director_boundaries(
            sub_segments,
            list(pre_director_plan.get("boundary_hints") or []),
            min_segment_duration=min_segment_duration,
        )
        sub_segments = collapse_transient_person_jitter(
            sub_segments,
            list(pre_director_plan.get("boundary_hints") or []),
        )
        seg_result["sub_segments"] = sub_segments
        seg_result["stats"]["total_sub_segments"] = len(sub_segments)
        logger.info("[预导演融合] %d -> %d 段；语义边界已加入保护来源", before, len(sub_segments))

    # ── 3.5 人脸身份细分 ─────────────────────────────────────────
    visual_merge_result: dict[str, object] | None = None
    if use_visual_merge is None:
        use_visual_merge = use_omnishotcut
    if use_visual_merge and len(sub_segments) > 1:
        logger.info("第4层：画面相似边界合并（清理 Omni 碎片）...")
        before = len(sub_segments)
        visual_merge_payload = merge_visually_similar_segments(
            video_path,
            sub_segments,
            dirs["probe"] / "visual_merge",
            fps=meta.fps,
        )
        sub_segments = visual_merge_payload["segments"]
        visual_merge_result = visual_merge_payload["result"]
        seg_result["sub_segments"] = sub_segments
        seg_result["stats"]["total_sub_segments"] = len(sub_segments)
        logger.info("[画面合并] %d -> %d 段，合并 %d 个弱边界",
                    before, len(sub_segments), before - len(sub_segments))
    elif use_visual_merge:
        logger.info("[画面合并] 片段数不足，跳过")
    else:
        logger.info("[画面合并] 已跳过 — 未开启")

    if use_face_id:
        logger.info("╔═══════════════════════════════════════════")
        logger.info("║ 第5层: 人脸身份复核 (InsightFace)")
        logger.info("║ ── 逐帧看脸→嵌入向量→比对相似度→发现换人")
        logger.info("╚═══════════════════════════════════════════")
        face_split_count = 0
        try:
            before = len(sub_segments)
            for s in sub_segments:
                pc = s.get("person_count", -1)
                if pc >= 1:
                    logger.info("[人脸识别] 待检片段 [%.1f-%.1f] (%.1fs) person_count=%d",
                                 s["start"], s["end"], s["end"] - s["start"], pc)
                else:
                    logger.info("[人脸识别] 跳过片段 [%.1f-%.1f]: 无人物",
                                 s["start"], s["end"])
            refined = _refine_by_face_identity(
                video_path, sub_segments, temp_dir=dirs["probe"],
                sample_interval=sample_interval,
            )
            face_split_count = sum(1 for s in refined if s.get("identity_id", 0) > 0)
            if face_split_count > 0:
                logger.info("[人脸识别] 切出 %d 个身份边界: %d → %d 段",
                             face_split_count, before, len(refined))
                sub_segments = refined
                for s in sub_segments:
                    logger.info("  ├ 片段 [%.1f-%.1f] identity=%s",
                                 s["start"], s["end"], s.get("identity_id", 0))
            else:
                logger.info("[人脸识别] 未检测到身份切换，保持 %d 段", before)
        except Exception as e:
            logger.warning("[人脸识别] 跳过: %s", e)
    else:
        logger.info("[人脸识别] 已跳过 — 用户关闭")

    # ── 3.6 SAM3 主体轨迹复核 ───────────────────────────────────────
    sam3_result: dict[str, object] | None = None
    if use_sam3_finalize:
        logger.info("第6层：SAM3 主体轨迹复核（仅风险片段）...")
        before = len(sub_segments)
        sam3_payload = finalize_segments_with_sam3(video_path, sub_segments, dirs["probe"] / "sam3_finalize")
        sub_segments = sam3_payload["segments"]
        sam3_result = sam3_payload["result"]
        (dirs["probe"] / "sam3_finalize.json").write_text(
            json.dumps(sam3_result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        if sam3_result.get("skipped"):
            logger.info("[SAM3整合] 已跳过：%s", sam3_result.get("reason", "unknown"))
        elif sam3_result.get("changed"):
            logger.info("[SAM3整合] 已整合切点：%d -> %d 段", before, len(sub_segments))
        else:
            logger.info("[SAM3整合] 未调整切点，保持 %d 段", len(sub_segments))
    else:
        logger.info("[SAM3整合] 已跳过 — 用户关闭")

    # ── 3.7 视觉模型身份仲裁 ───────────────────────────────────────
    if use_visual_model and gemini_api_key:
        logger.info("╔═══════════════════════════════════════════")
        logger.info("║ 补充仲裁: 视觉模型身份复核（仅不确定片段）")
        logger.info("╚═══════════════════════════════════════════")
        before = len(sub_segments)
        gemini_identity_split = _refine_by_gemini_identity(
            sub_segments, frames,
            api_key=gemini_api_key,
            base_url=gemini_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            model=gemini_model or "qwen3.5-omni-plus",
            concurrency=gemini_identity_concurrency,
        )
        if gemini_identity_split > 0:
            logger.info("[视觉模型] 发现 %d 个换人边界: %d → %d 段",
                         gemini_identity_split, before, len(sub_segments))
            for s in sub_segments:
                logger.info("  ├ 片段 [%.1f-%.1f]", s["start"], s["end"])
        else:
            logger.info("[视觉模型] 未发现需要补切的身份边界，保持 %d 段", before)
    elif use_visual_model and not gemini_api_key:
        logger.info("[视觉模型] 未启用（未提供 API Key）")
    else:
        logger.info("[视觉模型] 已跳过 — 用户关闭")

    # ── 3.75 max_segment_duration 强制切分 ────────────────────────────
    if max_segment_duration > 0:
        boundaries = sorted({s["start"] for s in sub_segments} | {s["end"] for s in sub_segments})
        old_count = len(boundaries) - 1
        split_boundaries = split_long_ranges(boundaries, max_segment_duration)
        if len(split_boundaries) > len(boundaries):
            logger.info("[强制切分] %d → %d 段（max=%.1fs）", old_count, len(split_boundaries) - 1, max_segment_duration)
            new_segments: list[dict[str, object]] = []
            for start, end in zip(split_boundaries, split_boundaries[1:]):
                new_seg = dict(sub_segments[0])
                new_seg["start"] = start
                new_seg["end"] = end
                new_seg["start_sources"] = list(sub_segments[0].get("start_sources", []))
                new_seg["end_sources"] = list(sub_segments[0].get("end_sources", []))
                new_seg["start_sources"].append("split_long_range")
                new_seg["end_sources"].append("split_long_range")
                new_segments.append(new_seg)
            sub_segments = new_segments
            seg_result["sub_segments"] = sub_segments
            seg_result["stats"]["total_sub_segments"] = len(sub_segments)

    # ── 3.8 背景提取 ─────────────────────────────────────────────────
    bg_results = []
    if extract_backgrounds:
        bg_dir = ensure_dir(dirs["ai_inputs"] / "background_refs")
        logger.info("提取背景图到 %s...", bg_dir)
        bg_results = batch_extract_backgrounds(video_path, sub_segments, bg_dir)
        (dirs["probe"] / "background_extraction.json").write_text(
            json.dumps(bg_results, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("背景提取完成：%d 张", len(bg_results))

    # ── 4. 构建 Segment 对象 ─────────────────────────────────────────
    logger.info("╔═══════════════════════════════════════════")
    logger.info("║ 最终分段结果（%d 段）", len(sub_segments))
    logger.info("╚═══════════════════════════════════════════")
    for s in sub_segments:
        logger.info("  ├ [%.1f-%.1f] (%.1fs) person=%s bg=%s",
                     s["start"], s["end"], s["end"] - s["start"],
                     s.get("person_count", "?"), s.get("is_pure_background", "?"))

    segments: list[Segment] = []
    logger.info("[后处理] 正在构建片段元数据...")
    for i, sub in enumerate(sub_segments):
        seg_id = f"{i + 1:03d}"

        # 找该片段对应的帧
        seg_frames = [f for f in frames if sub["start"] - 0.1 <= f.time <= sub["end"] + 0.1]
        if not seg_frames:
            # 找最近的一帧
            mid = (sub["start"] + sub["end"]) / 2
            closest = min(frames, key=lambda f: abs(f.time - mid)) if frames else None
            seg_frames = [closest] if closest else []

        segment = classify_from_sub_segment(seg_id, sub, seg_frames)

        semantic_scene = semantic_scene_for_time(
            pre_director_plan,
            (float(sub["start"]) + float(sub["end"])) / 2.0,
        )
        if semantic_scene:
            segment.scene_type = str(semantic_scene.get("scene_type") or "")
            segment.description = str(semantic_scene.get("description") or "")
            segment.has_product = bool(semantic_scene.get("has_product", False))
            segment.main_subject = str(semantic_scene.get("main_subject") or "")
            segment.generation_route = str(semantic_scene.get("generation_route") or "")

        # 传递检测器来源
        segment.start_sources = sub.get("start_sources", [])
        segment.end_sources = sub.get("end_sources", [])

        # 设置代表性帧
        rep = sub.get("representative_frame")
        if rep and Path(rep).exists():
            segment.representative_frame = rep
        elif seg_frames:
            segment.representative_frame = seg_frames[len(seg_frames) // 2].path

        segments.append(segment)

    # ── 5. 裁切片段 ──────────────────────────────────────────────────
    if export_video:
        logger.info("[导出片段] 开始裁切 %d 个 mp4 片段...", len(segments))
        # 重跑同一项目时先移除上次生成的片段，避免旧编号残留在结果目录。
        for old_clip in dirs["all_clips"].glob("*.mp4"):
            old_clip.unlink()
        last_export_report = 0
        for index, segment in enumerate(segments, 1):
            seg_type = "bg" if segment.person_count == 0 else f"p{segment.person_count}"
            filename = f"{segment.segment_id}_{seg_type}_{segment.start:.2f}_{segment.end:.2f}.mp4"
            out_path = dirs["all_clips"] / filename
            try:
                cut_segment(video_path, segment.start, segment.end, out_path)
                segment.output_path = str(out_path)
            except Exception as e:
                logger.warning("裁切片段 %s 失败: %s", segment.segment_id, e)
            pct = int(index / max(1, len(segments)) * 100)
            if pct >= last_export_report + 10 or index == len(segments):
                logger.info("[导出片段] %d%% (%d/%d)", pct, index, len(segments))
                last_export_report = pct
        logger.info("[导出片段] 完成")
    else:
        logger.info("[导出片段] 已跳过 — 用户关闭")

    # ── 6. 写 manifest 和报告 ────────────────────────────────────────
    logger.info("[报告] 正在写 manifest...")
    write_global_manifest(meta, frames, segments, dirs["root"])
    contact_sheet_path = dirs["probe"] / "contact_sheet_segments.jpg"
    report_path = dirs["probe"] / "segmentation_report.html"
    logger.info("[报告] 正在生成接触表...")
    make_contact_sheet(segments, contact_sheet_path)
    logger.info("[报告] 正在生成 HTML 报告...")
    make_html_report(segments, report_path)
    logger.info("[报告] 完成")

    result: dict[str, object] = {
        "project_dir": str(dirs["root"]),
        "clips_dir": str(dirs["all_clips"]),
        "report_path": str(report_path),
        "contact_sheet_path": str(contact_sheet_path),
        "meta": meta.to_dict(),
        "frame_count": len(frames),
        "segment_count": len(segments),
        "segments": [segment.to_dict() for segment in segments],
        "two_pass_stats": stats,
        "background_count": len(bg_results),
        "omnishotcut": seg_result.get("omnishotcut"),
        "sam3_finalize": sam3_result,
        "visual_merge": visual_merge_result,
        "pre_director": pre_director_plan,
    }

    return result


def run_segmentation(
    video_path: str | Path,
    project_dir: str | Path,
    sample_interval: float = 1.0,
    min_segment_duration: float = 2.0,
    max_segment_duration: float = 6.0,
    export_video: bool = True,
    gemini_api_key: str | None = None,
    gemini_base_url: str | None = None,
    gemini_model: str | None = None,
    gemini_identity_concurrency: int = 10,
    use_two_pass: bool = False,
    use_omnishotcut: bool = False,
    yolo_conf_threshold: float = 0.35,
    device: str | None = None,
    yolo_batch_size: int | None = None,
    use_scene_detect: bool = True,
    use_face_id: bool = True,
    use_visual_model: bool = True,
    extract_backgrounds: bool = False,
    use_sam3_finalize: bool = False,
    use_visual_merge: bool | None = None,
    use_pre_director: bool = True,
    pre_director_api_key: str | None = None,
    pre_director_base_url: str | None = None,
    pre_director_model: str | None = None,
) -> dict[str, object]:
    """运行完整分割管线。

    工作模式：
    - use_two_pass=True → PySceneDetect + YOLO 两层检测（推荐）
    - 提供 gemini_api_key → Gemini 先看全帧 → 语义边界裁切
    - 都不提供 → 像素算法检测边界（降级）

    Parameters
    ----------
    use_two_pass : bool
        启用 PySceneDetect + YOLO 两层管线（推荐，切割更精准）
    use_omnishotcut : bool
        在 two-pass 管线前启用 OmniShotCut 候选镜头层。
    yolo_conf_threshold : float
        YOLO 人物检测置信度阈值
    device : str | None
        "cuda" / "cpu" / None(自动检测)
    """
    if use_omnishotcut or use_sam3_finalize or use_pre_director:
        use_two_pass = True

    # ── 路由：两层管线优先 ──────────────────────────────────────────
    if use_two_pass:
        return run_segmentation_v2(
            video_path=video_path,
            project_dir=project_dir,
            sample_interval=sample_interval,
            min_segment_duration=min_segment_duration,
            max_segment_duration=max_segment_duration,
            export_video=export_video,
            extract_backgrounds=extract_backgrounds,
            yolo_conf_threshold=yolo_conf_threshold,
            device=device,
            yolo_batch_size=yolo_batch_size,
            gemini_api_key=gemini_api_key,
            gemini_base_url=gemini_base_url,
            gemini_model=gemini_model,
            gemini_identity_concurrency=gemini_identity_concurrency,
            use_omnishotcut=use_omnishotcut,
            use_scene_detect=use_scene_detect,
            use_face_id=use_face_id,
            use_visual_model=use_visual_model,
            use_sam3_finalize=use_sam3_finalize,
            use_visual_merge=use_visual_merge,
            use_pre_director=use_pre_director,
            pre_director_api_key=pre_director_api_key,
            pre_director_base_url=pre_director_base_url,
            pre_director_model=pre_director_model,
        )

    # ── 以下为旧管线（纯 Gemini 或算法降级） ──────────────────────────
    video_path = Path(video_path)
    dirs = create_project_dirs(project_dir)

    # ── 1. 探针 ──────────────────────────────────────────────────────
    meta = probe_video(video_path)
    (dirs["source"] / "source_pointer.json").write_text(
        json.dumps({"source_path": str(video_path), "note": "Original video is referenced, not copied."}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (dirs["source"] / "original_meta.json").write_text(json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 2. 抽帧 ──────────────────────────────────────────────────────
    frames = sample_frames(video_path, meta.duration, dirs["frames"], sample_interval)

    # ── 3. 边界检测（Gemini 优先 / 算法降级） ────────────────────────
    gemini_analysis: list[dict] | None = None
    gemini_scenes: list[dict] | None = None

    if gemini_api_key:
        logger.info("Gemini 全帧场景分析（%d 帧）...", len(frames))
        try:
            client = create_client(
                api_key=gemini_api_key,
                base_url=gemini_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
                model=gemini_model or "qwen3.5-omni-plus",
            )
            gemini_scenes = client.analyze_full_video(frames)
            if gemini_scenes:
                logger.info("视觉模型 返回 %d 个场景", len(gemini_scenes))
                (dirs["probe"] / "gemini_scenes.json").write_text(
                    json.dumps(gemini_scenes, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except Exception as e:
            logger.error("Gemini 全帧分析失败: %s", e, exc_info=True)
            gemini_scenes = None

    if gemini_scenes:
        # ── 3a. 用 Gemini 的场景边界 ─────────────────────────────────
        segments = build_segments_from_scenes(gemini_scenes, frames)
        gemini_analysis = gemini_scenes
    else:
        # ── 3b. 降级：算法边界检测 ───────────────────────────────────
        logger.info("降级到算法边界检测")
        boundaries = detect_boundaries(frames, meta.duration, min_segment_duration, max_segment_duration)
        segments = build_segments(boundaries, frames)

    # ── 4. 裁切片段 ──────────────────────────────────────────────────
    for segment in segments:
        mid = (segment.start + segment.end) / 2
        closest = min(frames, key=lambda item: abs(item.time - mid)) if frames else None
        if closest:
            segment.representative_frame = closest.path
        if export_video:
            filename = f"{segment.segment_id}_{safe_stem(segment.segment_type)}_{segment.start:.2f}_{segment.end:.2f}.mp4"
            out_path = dirs["all_clips"] / filename
            cut_segment(video_path, segment.start, segment.end, out_path)
            segment.output_path = str(out_path)
            # 按生成路线镜像到对应目录
            route = getattr(segment, 'generation_route', None) or segment.segment_type
            dirname = GENERATION_DIR_MAP.get(route, "99_need_review")
            target_dir = ensure_dir(dirs["segments"] / dirname)
            mirror_clip(out_path, target_dir / filename)

    # ── 5. 写 manifest 和报告 ────────────────────────────────────────
    write_global_manifest(meta, frames, segments, dirs["root"])
    contact_sheet_path = dirs["probe"] / "contact_sheet_segments.jpg"
    report_path = dirs["probe"] / "segmentation_report.html"
    make_contact_sheet(segments, contact_sheet_path)
    make_html_report(segments, report_path, gemini_analysis=gemini_analysis)

    result: dict[str, object] = {
        "project_dir": str(dirs["root"]),
        "clips_dir": str(dirs["all_clips"]),
        "report_path": str(report_path),
        "contact_sheet_path": str(contact_sheet_path),
        "meta": meta.to_dict(),
        "frame_count": len(frames),
        "segment_count": len(segments),
        "segments": [segment.to_dict() for segment in segments],
    }
    if gemini_analysis:
        result["gemini_analysis"] = gemini_analysis

    return result


# ── 以下保持原样（sample_frames / detect_boundaries / …） ─────────────


def sample_frames(video_path: Path, duration: float, frames_dir: Path, sample_interval: float) -> list[FrameFeatures]:
    started_at = time.monotonic()
    features: list[FrameFeatures] = []
    previous_pixels = None
    total = int(math.floor(duration / sample_interval)) + 1
    logger.info(f"[抽帧] 开始: 共 {total} 帧, 间隔 {sample_interval:.1f}s")
    logger.info("[抽帧] 正在一次性解码视频，请稍候...")
    for stale in frames_dir.glob("frame_*.jpg"):
        stale.unlink(missing_ok=True)
    for stale in frames_dir.glob("_bulk_*.jpg"):
        stale.unlink(missing_ok=True)

    bulk_pattern = frames_dir / "_bulk_%06d.jpg"
    try:
        extract_frames_bulk(
            video_path,
            bulk_pattern,
            frames_per_second=1.0 / max(sample_interval, 0.001),
        )
        extracted = sorted(frames_dir.glob("_bulk_*.jpg"))
        if not extracted:
            raise RuntimeError("FFmpeg 未生成采样帧")
        for extra in extracted[total:]:
            extra.unlink(missing_ok=True)
        extracted = extracted[:total]
        logger.info("[抽帧] 视频解码完成，正在分析 %d 张画面...", len(extracted))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[抽帧] 批量解码不可用，回退逐帧模式: %s", exc)
        extracted = []

    last_report = 0
    source_frames = extracted if extracted else [None] * total
    for index, bulk_path in enumerate(source_frames):
        time_seconds = min(duration - 0.05, index * sample_interval)
        if time_seconds < 0:
            time_seconds = 0.0
        frame_path = frames_dir / f"frame_{index + 1:04d}_{time_seconds:.2f}s.jpg"
        if bulk_path is None:
            extract_frame(video_path, time_seconds, frame_path)
        else:
            bulk_path.replace(frame_path)
        frame_features, previous_pixels = analyze_frame(frame_path, time_seconds, previous_pixels)
        features.append(frame_features)
        pct = int((index + 1) / len(source_frames) * 100)
        if pct >= last_report + 10:
            logger.info(f"[抽帧] {pct}% ({index + 1}/{len(source_frames)})")
            last_report = pct

    (frames_dir.parent / "scene_scores.json").write_text(
        json.dumps([item.to_dict() for item in features], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("[抽帧] 全部完成，用时 %.1fs", time.monotonic() - started_at)
    return features


def detect_boundaries(
    frames: list[FrameFeatures],
    duration: float,
    min_segment_duration: float,
    max_segment_duration: float,
) -> list[float]:
    if not frames:
        return [0.0, duration]

    diffs = sorted(frame.diff_prev for frame in frames[1:])
    median = diffs[len(diffs) // 2] if diffs else 0.0
    threshold = max(0.115, median * 2.8)

    boundaries = [0.0]
    last = 0.0
    previous = frames[0]
    for frame in frames[1:]:
        if is_scene_change(previous, frame, threshold) and frame.time - last >= min_segment_duration:
            boundaries.append(round(frame.time, 3))
            last = frame.time
        previous = frame

    if boundaries[-1] < duration:
        boundaries.append(round(duration, 3))

    return split_long_ranges(boundaries, max_segment_duration)


def is_scene_change(previous: FrameFeatures, current: FrameFeatures, visual_threshold: float) -> bool:
    if current.diff_prev >= visual_threshold:
        return True
    if not has_human_presence(previous) == has_human_presence(current):
        center_delta = abs(current.center_skin_ratio - previous.center_skin_ratio)
        skin_delta = abs(current.skin_ratio - previous.skin_ratio)
        return current.diff_prev >= 0.16 and (center_delta >= 0.10 or skin_delta >= 0.12)
    return False


def has_human_presence(frame: FrameFeatures) -> bool:
    strong_center_human = frame.center_skin_ratio >= 0.14 and frame.skin_ratio >= 0.10
    weak_layout_human = (
        frame.center_skin_ratio >= 0.10
        and frame.skin_ratio >= 0.12
        and frame.white_ratio <= 0.19
        and frame.blue_ratio <= 0.04
    )
    return strong_center_human or weak_layout_human


def split_long_ranges(boundaries: list[float], max_segment_duration: float) -> list[float]:
    if max_segment_duration <= 0:
        return boundaries

    result = [boundaries[0]]
    for start, end in zip(boundaries, boundaries[1:]):
        current = start
        while end - current > max_segment_duration:
            current = round(current + max_segment_duration, 3)
            result.append(current)
        if end > result[-1]:
            result.append(end)
    return result


def build_segments(boundaries: list[float], frames: list[FrameFeatures], min_duration: float = 0.3) -> list[Segment]:
    segments: list[Segment] = []
    seq = 0
    for start, end in zip(boundaries, boundaries[1:]):
        if end - start < min_duration:
            continue  # 跳过过短片段
        seq += 1
        selected = [frame for frame in frames if start <= frame.time < end]
        if not selected:
            selected = [min(frames, key=lambda item: abs(item.time - ((start + end) / 2)))] if frames else []
        segment = classify_segment(f"{seq:03d}", start, end, selected)
        segments.append(segment)
    return segments


def build_segments_from_scenes(scenes: list[dict], frames: list[FrameFeatures]) -> list[Segment]:
    """从 视觉模型 返回的场景列表构建 Segment 对象。

    Gemini 的 analyze_full_video 已经给出了边界 + 类型 + 生成路线。
    直接用语义分割结果，不再跑像素算法检测。
    """
    segments: list[Segment] = []
    for idx, scene in enumerate(scenes, start=1):
        start = scene.get("start", 0.0)
        end = scene.get("end", 0.0)
        if end - start < 0.3:
            continue

        has_human = scene.get("has_human", False)
        segment_type = "with_human" if has_human else "without_human"
        scene_type = scene.get("scene_type", "unknown")
        route = scene.get("generation_route", "manual_review")
        needs_review = scene.get("needs_manual_review", True)

        scene_label = SCENE_TYPE_LABELS.get(scene_type, "未知")
        route_label = GENERATION_ROUTE_LABELS.get(route, "人工确认")

        notes = []
        if needs_review:
            notes.append("Gemini 建议人工复审")
        if (end - start) > 5.8:
            notes.append("长片段，生成前考虑再细分")

        seg = Segment(
            segment_id=f"{idx:03d}",
            start=start,
            end=end,
            segment_type=segment_type,
            confidence=0.85 if not needs_review else 0.55,
            detected=[scene_type, f"has_human={has_human}"],
            recommended_tech=f"{scene_label} → {route_label}",
            needs_ai_driver=has_human,
            needs_manual_check=needs_review,
            notes=notes,
            generation_route=route,
            scene_type=scene_type,
            description=scene.get("description", ""),
            has_product=scene.get("has_product", False),
            main_subject=scene.get("main_subject", ""),
        )

        mid = (start + end) / 2
        closest = min(frames, key=lambda item: abs(item.time - mid)) if frames else None
        if closest:
            seg.representative_frame = closest.path

        segments.append(seg)

    return segments


def mirror_clip(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path.unlink()
    try:
        os.link(source_path, target_path)
    except OSError:
        shutil.copy2(source_path, target_path)


def write_category_manifests(segments: list[Segment], segments_dir: Path) -> None:
    for segment_type, dirname in CATEGORY_DIRS.items():
        selected = [segment.to_dict() for segment in segments if segment.segment_type == segment_type]
        out_dir = segments_dir / dirname
        (out_dir / "manifest.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")


def write_global_manifest(meta, frames: list[FrameFeatures], segments: list[Segment], root: Path) -> None:
    data = {
        "version": 1,
        "meta": meta.to_dict() if hasattr(meta, "to_dict") else asdict(meta),
        "segments": [segment.to_dict() for segment in segments],
        "frames": [frame.to_dict() for frame in frames],
    }
    (root / "manifest.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _refine_by_face_identity(
    video_path: Path,
    sub_segments: list[dict],
    temp_dir: Path,
    sample_interval: float = 0.5,
) -> list[dict]:
    """第三层：对单人片段做人脸身份检测，区分不同角色。

    只处理 person_count == 1 的片段（单人但可能身份切换）。
    多人片段、纯背景片段原样返回不动。
    """
    from .face_identity import get_face_embedding, split_by_identity
    from .ffmpeg_tools import extract_frame
    import shutil as _shutil

    work_dir = Path(temp_dir) / "_face_id_temp"
    work_dir.mkdir(parents=True, exist_ok=True)
    refined = []

    try:
        for seg in sub_segments:
            pc = seg.get("person_count", -1)
            if pc != 1:
                refined.append(seg)
                continue

            start = seg["start"]
            end = seg["end"]
            duration = end - start
            if duration < 3.0:
                refined.append(seg)
                continue

            num_samples = max(3, int(duration / sample_interval) + 1)
            frame_paths = []
            frame_times = []

            for s in range(num_samples):
                t = start + min(s * sample_interval, duration - 0.05)
                if t >= end:
                    t = (start + end) / 2
                fp = work_dir / f"face_{start:.1f}_{end:.1f}_{s:02d}.jpg"
                try:
                    extract_frame(video_path, t, fp)
                    frame_paths.append(str(fp))
                    frame_times.append(t)
                except Exception:
                    continue

            if len(frame_paths) < 2:
                refined.append(seg)
                continue
            evidence_paths = [
                frame_paths[index]
                for index in sorted({0, len(frame_paths) // 2, len(frame_paths) - 1})
            ]
            face_evidence_frames = sum(
                1 for path in evidence_paths if get_face_embedding(path) is not None
            )

            # 构造二分搜索回调：对任意时刻抽帧
            def _refine_extract(t: float) -> str | None:
                fp = work_dir / f"refine_{start:.1f}_{t:.3f}.jpg"
                try:
                    extract_frame(video_path, t, fp)
                    return str(fp)
                except Exception:
                    return None

            splits = split_by_identity(
                video_path=video_path,
                segment_start=start, segment_end=end,
                frame_paths=frame_paths, frame_times=frame_times,
                extract_fn=_refine_extract,
                similarity_threshold=0.40,
            )

            if len(splits) <= 1:
                seg["identity_id"] = 0
                seg["face_identity_checked"] = face_evidence_frames >= 2
                seg["face_identity_evidence_frames"] = face_evidence_frames
                refined.append(seg)
                continue

            for split in splits:
                new_seg = dict(seg)
                new_seg["start"] = split["start"]
                new_seg["end"] = split["end"]
                new_seg["identity_id"] = split.get("identity_id", 0)
                new_seg["_face_split"] = True
                new_seg["face_identity_checked"] = face_evidence_frames >= 2
                new_seg["face_identity_evidence_frames"] = face_evidence_frames
                # 标记人脸身份层产生的切分
                new_seg["start_sources"] = list(seg.get("start_sources", []))
                new_seg["end_sources"] = list(seg.get("end_sources", []))
                new_seg["start_sources"].append("face_id")
                new_seg["end_sources"].append("face_id")
                refined.append(new_seg)

    finally:
        if work_dir.exists():
            _shutil.rmtree(work_dir, ignore_errors=True)

    return refined


def _refine_by_gemini_identity(
    sub_segments: list[dict],
    frames: list[FrameFeatures],
    api_key: str,
    base_url: str = "http://152.136.38.202:3000/v1",
    model: str = "qwen3.5-omni-plus",
    concurrency: int = 10,
) -> int:
    """并发检查每个有人片段的主角身份，并应用身份安全边界。"""
    from bisect import bisect_left, bisect_right
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from .gemini_analyzer import create_client

    split_count = 0
    # (片段索引, 旧人物安全结束时间, 新人物安全开始时间)
    # 两个时间之间可能是叠化/转场，直接舍弃以保证每个输出片段身份纯净。
    all_boundaries: list[tuple[int, float, float]] = []
    sorted_frames = sorted(frames, key=lambda frame: frame.time)
    sorted_frame_times = [frame.time for frame in sorted_frames]
    max_workers = min(max(1, int(concurrency)), 32)

    def _run_identity_job(
        job: tuple[int, dict, list[Path], float | None],
    ) -> list[tuple[int, float, float]]:
        seg_idx, seg, paths, yolo_cut = job
        job_boundaries: list[tuple[int, float, float]] = []
        client = None
        try:
            # requests.Session 不跨线程共享，每个并发任务使用独立客户端。
            client = create_client(api_key=api_key, base_url=base_url, model=model)
            _ask_identity_change(
                client, paths, seg_idx, seg, job_boundaries, yolo_cut=yolo_cut,
            )
        except Exception as e:
            logger.warning(
                "[视觉模型] 片段[%.2f-%.2f]身份检查失败: %s",
                seg["start"], seg["end"], e,
            )
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    logger.debug("[视觉模型] 关闭片段客户端失败", exc_info=True)
        return job_boundaries

    futures = []
    logger.info("[视觉模型] 启动流式身份检查，并发上限=%d", max_workers)
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="identity-check",
    ) as executor:
        # 每准备好一个片段就立即提交；请求在后台执行时，主线程继续准备后续片段。
        for i, seg in enumerate(sub_segments):
            pc = seg.get("person_count", -1)
            if pc < 1:
                logger.info("  [视觉模型] 跳过片段[%.1f-%.1f]: 无人物", seg["start"], seg["end"])
                continue
            if not _needs_visual_identity_review(seg):
                logger.info(
                    "  [视觉模型] 跳过片段[%.1f-%.1f]: 已由人脸层稳定确认",
                    seg["start"], seg["end"],
                )
                continue

            left = bisect_left(sorted_frame_times, seg["start"])
            right = bisect_right(sorted_frame_times, seg["end"])
            seg_frames = sorted_frames[left:right]
            if len(seg_frames) < 2:
                logger.info("  [视觉模型] 跳过片段[%.1f-%.1f]: 帧数不足(%d)", seg["start"], seg["end"], len(seg_frames))
                continue

            # 找出 YOLO 检测到的人数变化前最后一帧（切在变化前，不带尾巴）
            yolo_cut_time = None
            all_dets = seg.get("_all_frame_detections", [])
            if all_dets:
                prev_pc = all_dets[0].get("person_count", -1)
                prev_det = all_dets[0]
                for det in all_dets[1:]:
                    cur_pc = det.get("person_count", -1)
                    if prev_pc != cur_pc and cur_pc > 0 and prev_pc > 0:
                        # 切在变化前的安全帧时间（确保第一段不含第二个人）
                        safe_t = prev_det.get("time", 0)
                        yolo_cut_time = safe_t
                        logger.info("  [视觉模型] YOLO 发现人数变化 %d→%d: 安全切点=%.2fs (变化前最后一帧)",
                                     prev_pc, cur_pc, safe_t)
                        break
                    prev_pc = cur_pc
                    prev_det = det

            # 模型只识别图片顺序，不负责猜时间。尽量发送稠密帧；长片段最多
            # 均匀取 24 张，切点精度由这些图片的本地时间戳决定。
            if len(seg_frames) <= 24:
                picks = seg_frames
            else:
                indices = sorted({
                    round(index * (len(seg_frames) - 1) / 23)
                    for index in range(24)
                })
                picks = [seg_frames[index] for index in indices]
            paths = [Path(frame.path) for frame in picks if Path(frame.path).exists()]
            if len(paths) < 2:
                continue

            futures.append(executor.submit(
                _run_identity_job,
                (i, seg, paths, yolo_cut_time),
            ))
            sent_times = [_parse_frame_time(path) for path in paths]
            logger.info(
                "  [视觉模型] 提交片段[%.1f-%.1f]: %d 帧（本地时间 %s）",
                seg["start"], seg["end"], len(paths),
                ", ".join(f"{t:.2f}" for t in sent_times if t is not None),
            )

        if futures:
            logger.info(
                "[视觉模型] 已提交 %d 个身份请求，实际并发=%d",
                len(futures), min(max_workers, len(futures)),
            )
            for completed, future in enumerate(as_completed(futures), 1):
                all_boundaries.extend(future.result())
                logger.info("[视觉模型] 身份检查进度 %d/%d", completed, len(futures))

    if not futures:
        return 0

    # ── 应用身份安全切分 ────────────────────────────────────────────
    if not all_boundaries:
        return 0

    new_segments: list[dict] = []
    for seg_idx, seg in enumerate(sub_segments):
        boundaries = sorted(
            (old_end, new_start)
            for idx, old_end, new_start in all_boundaries
            if idx == seg_idx
        )
        if not boundaries:
            new_segments.append(seg)
            continue

        current = seg["start"]
        produced = 0
        for old_end, new_start in boundaries:
            old_end = min(max(old_end, current), seg["end"])
            new_start = min(max(new_start, old_end), seg["end"])
            if old_end - current >= 0.01:
                new_seg = dict(seg)
                new_seg["start"] = round(current, 3)
                new_seg["end"] = round(old_end, 3)
                new_seg["_gemini_split"] = True
                # 标记 Gemini 角色识别层产生的切分
                new_seg["start_sources"] = list(seg.get("start_sources", []))
                new_seg["end_sources"] = list(seg.get("end_sources", []))
                new_seg["start_sources"].append("gemini")
                new_seg["end_sources"].append("gemini")
                new_segments.append(new_seg)
                produced += 1
            if new_start > old_end:
                logger.info(
                    "视觉模型 丢弃身份不确定区间 [%.3f-%.3f]（%.3fs）",
                    old_end, new_start, new_start - old_end,
                )
            current = new_start

        if seg["end"] - current >= 0.01:
            last_seg = dict(seg)
            last_seg["start"] = round(current, 3)
            last_seg["end"] = round(seg["end"], 3)
            last_seg["_gemini_split"] = True
            # 标记 Gemini 角色识别层产生的切分
            last_seg["start_sources"] = list(seg.get("start_sources", []))
            last_seg["end_sources"] = list(seg.get("end_sources", []))
            last_seg["start_sources"].append("gemini")
            last_seg["end_sources"].append("gemini")
            new_segments.append(last_seg)
            produced += 1

        split_count += len(boundaries)
        logger.info("视觉模型 身份安全切分: %s → %d 段", [seg["start"], seg["end"]], produced)

    sub_segments.clear()
    sub_segments.extend(new_segments)
    return split_count


def _needs_visual_identity_review(segment: dict) -> bool:
    """Reserve remote visual-model calls for segments not settled by local layers."""
    if not segment.get("face_identity_checked"):
        return True
    if int(segment.get("person_count", -1)) != 1:
        return True
    sources = set(segment.get("start_sources") or []) | set(segment.get("end_sources") or [])
    return bool(sources & {"yolo", "yolo_transient_multi", "sam3_track_lost"})


def _parse_frame_time(path: Path) -> float | None:
    """从帧文件名提取时间戳。帧名格式: frame_0001_0.00s.jpg"""
    import re
    m = re.search(r'_(\d+\.\d+)s\.', path.name)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _ask_identity_change(
    client,
    paths: list[Path],
    seg_idx: int,
    seg: dict,
    all_boundaries: list[tuple[int, float, float]],
    hint: str = "",
    yolo_cut: float | None = None,
) -> None:
    """向 Gemini 发送一组帧，询问是否发生角色切换。"""
    from .gemini_analyzer import _IDENTITY_CHECK_PROMPT

    prompt = _IDENTITY_CHECK_PROMPT.format(hint=hint)
    result = client.analyze_segment_keyframes(paths, start=0, end=0, prompt_override=prompt)
    if result is None:
        logger.info("    └ 视觉模型 无返回（跳过）")
        return

    same = result.get("same_person", True)
    reasoning = result.get("reasoning", "")
    switch_time = result.get("switch_time")
    last_old_person_frame = result.get("last_old_person_frame")
    first_new_person_frame = result.get("first_new_person_frame")
    new_person_frame = result.get("new_person_frame")
    switch_after_frame = result.get("switch_after_frame")
    frame_times = [_parse_frame_time(p) for p in paths]
    frame_times = [t for t in frame_times if t is not None]

    logger.info(
        "    └ 视觉模型 回答: same_person=%s reasoning=%s last_old=%s first_new=%s new_person_frame=%s switch_after_frame=%s switch_time=%s",
        same, reasoning, last_old_person_frame, first_new_person_frame,
        new_person_frame, switch_after_frame, switch_time,
    )

    if same:
        return

    # ── 优先用 Gemini 判断出的帧序号，再映射到本地真实帧时间 ─────────
    def _parse_frame_number(value) -> int | None:
        if value is None:
            return None
        try:
            number = int(float(value))
        except (ValueError, TypeError):
            return None
        return number if number > 0 else None

    old_frame = _parse_frame_number(last_old_person_frame)
    new_frame = _parse_frame_number(first_new_person_frame)
    if old_frame is None:
        old_frame = _parse_frame_number(switch_after_frame)
    if new_frame is None:
        new_frame = _parse_frame_number(new_person_frame)
    if old_frame is None and new_frame is not None:
        old_frame = new_frame - 1
    if new_frame is None and old_frame is not None:
        new_frame = old_frame + 1

    if len(frame_times) >= 2 and old_frame is not None and new_frame is not None:
        if 1 <= old_frame < new_frame <= len(frame_times):
            old_end = frame_times[old_frame - 1]
            new_start = frame_times[new_frame - 1]
            if seg["start"] < new_start and old_end < seg["end"]:
                all_boundaries.append((seg_idx, old_end, new_start))
                logger.info(
                    "    └ ✅ 身份安全边界：旧人物到 %.3fs（图%d），新人物从 %.3fs（图%d）开始",
                    old_end, old_frame, new_start, new_frame,
                )
                return
        logger.info(
            "    └ 视觉模型帧序号无效：old=%s new=%s，当前发送 %d 张图",
            old_frame, new_frame, len(frame_times),
        )

    # 模型只看图片，不知道视频时间轴；旧字段 switch_time 只能作为日志参考。
    if switch_time is not None:
        logger.info("    └ 忽略视觉模型 switch_time=%s（模型只接收图片，不用它作为切点）", switch_time)

    # ── 其次用 YOLO 检测到的人物进出时间 ────────────────────────
    if yolo_cut is not None and seg["start"] < yolo_cut < seg["end"]:
        all_boundaries.append((seg_idx, yolo_cut, yolo_cut))
        logger.info("    └ ✅ 将在 %.2fs 处切开（YOLO 人物进出时间）", yolo_cut)
        return
    if yolo_cut is not None:
        logger.info(
            "    └ YOLO 候选切点 %.2fs 不在片段 [%.2f-%.2f] 内，跳过",
            yolo_cut, seg["start"], seg["end"],
        )

    logger.warning("    └ 模型确认换人但未给出有效图片编号；为避免误切，不使用猜测时间")


# 临时 logger（正式代码建议统一配置）
import logging
logger = logging.getLogger(__name__)
