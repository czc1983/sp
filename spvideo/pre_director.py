from __future__ import annotations

import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from .models import FrameFeatures


def analyze_pre_director(
    frames: list[FrameFeatures],
    *,
    video_path: str | Path | None = None,
    has_audio: bool = False,
    duration: float,
    output_path: str | Path,
    api_key: str = "",
    base_url: str = "",
    model: str = "",
    on_progress: Callable[[str], None] | None = None,
    include_asset_manifest: bool = False,
) -> dict[str, Any]:
    log = on_progress or (lambda _message: None)
    plan: dict[str, Any] = {
        "version": 1,
        "status": "disabled" if not api_key else "running",
        "duration": float(duration),
        "generated_at": time.time(),
        "story_summary": "",
        "characters": [],
        "key_actions": [],
        "scenes": [],
        "boundary_hints": [],
        "source": "original_video_sampled_frames",
        "audio_understanding": False,
        "audio_status": "not_requested",
        "transcript": None,
    }
    if include_asset_manifest:
        plan["asset_manifest"] = {"results": []}
        plan["analysis_frame_manifest"] = []
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not api_key:
        plan["status"] = "missing_api_key"
        plan["note"] = "预导演已开启，但没有故事模型 API Key；技术切分继续运行。"
        _write_plan(destination, plan)
        return plan
    if not frames and not video_path:
        plan["status"] = "no_frames"
        plan["note"] = "没有可供预导演分析的采样帧。"
        _write_plan(destination, plan)
        return plan

    from .gemini_analyzer import DEFAULT_BASE_URL, DEFAULT_MODEL, GeminiClient
    from .yuanqi_upload import upload_file_for_url

    log("> [预导演] 正在从原视频反推故事、角色和语义边界...")
    director_frames = _select_director_frames(frames, duration)
    plan["input_frame_count"] = len(frames)
    plan["analysis_frame_count"] = len(director_frames)
    if include_asset_manifest:
        plan["analysis_frame_manifest"] = [
            {
                "index": index,
                "time": round(float(frame.time), 3),
                "path": str(Path(frame.path)),
            }
            for index, frame in enumerate(director_frames, start=1)
        ]
    log(f"> [预导演] 导演采样：{len(frames)} 张检测帧压缩为 {len(director_frames)} 张分析帧")

    visual_client = GeminiClient(
        api_key=api_key,
        base_url=base_url or DEFAULT_BASE_URL,
        model=model or DEFAULT_MODEL,
        request_timeout=240,
    )
    audio_executor: ThreadPoolExecutor | None = None
    audio_future = None
    scenes: list[dict[str, Any]] | None = None
    mode2_plan: dict[str, Any] | None = None

    if video_path:
        try:
            log("> [预导演] 正在上传原视频到元启，获取公网 URL...")
            upload_info = _run_with_heartbeat(
                lambda: upload_file_for_url(video_path, api_key=api_key),
                log=log,
                label="原视频上传",
            )
            video_url = str(upload_info.get("url") or "")
            plan["source"] = "original_video_url"
            plan["video_url"] = video_url
            plan["upload_size"] = upload_info.get("size")
            log("> [预导演] 原视频 URL 已就绪，正在交给千问直接读视频和音频...")
            if include_asset_manifest:
                mode2_plan = _run_with_heartbeat(
                    lambda: visual_client.analyze_mode2_plan(
                        duration=duration,
                        video_url=video_url,
                        frames=director_frames,
                    ),
                    log=log,
                    label="视频 URL 故事与资产理解",
                )
                if isinstance(mode2_plan, dict):
                    scenes = mode2_plan.get("scenes") or []
            else:
                scenes = _run_with_heartbeat(
                    lambda: visual_client.analyze_video_url(video_url, duration=duration),
                    log=log,
                    label="视频 URL 理解",
                )
            if scenes or mode2_plan is not None:
                plan["audio_status"] = "included_in_video" if has_audio else "no_audio_stream"
                plan["audio_understanding"] = bool(has_audio)
        except Exception as exc:  # noqa: BLE001
            plan["video_url_error"] = str(exc)[:500]
            log(f"> [预导演] 视频 URL 预导演失败，回退关键帧/音频兜底：{exc}")

    def transcribe_audio_task() -> dict[str, Any]:
        from .ffmpeg_tools import extract_audio_chunks_for_analysis

        audio_model = _audio_transcription_model(model or DEFAULT_MODEL)
        audio_client = GeminiClient(
            api_key=api_key,
            base_url=base_url or DEFAULT_BASE_URL,
            model=audio_model,
            request_timeout=240,
        )
        try:
            log(f"> [预导演] 音频转写与画面理解并行（{audio_model}）...")
            chunk_seconds = 1200
            audio_paths = extract_audio_chunks_for_analysis(
                video_path,
                destination.parent,
                chunk_seconds=chunk_seconds,
            )
            transcript_parts: list[dict[str, Any] | None] = []
            for index, audio_path in enumerate(audio_paths):
                if len(audio_paths) > 1:
                    log(f"> [预导演] 正在转写音频分块 {index + 1}/{len(audio_paths)}...")
                transcript_parts.append(_run_with_heartbeat(
                    lambda path=audio_path: audio_client.transcribe_audio(path),
                    log=log,
                    label="音频转写",
                ))
            transcript = _merge_transcript_chunks(transcript_parts, chunk_seconds=chunk_seconds)
            completed = sum(1 for item in transcript_parts if item)
            status = (
                "ready" if transcript and completed == len(audio_paths)
                else "partial" if transcript
                else "transcription_failed"
            )
            return {
                "transcript": transcript,
                "status": status,
                "chunk_count": len(audio_paths),
                "model": audio_model,
            }
        finally:
            audio_client.close()

    if scenes or mode2_plan is not None:
        pass
    elif has_audio and video_path:
        audio_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pre-director-audio")
        audio_future = audio_executor.submit(transcribe_audio_task)
    elif not has_audio:
        plan["audio_status"] = "no_audio_stream"

    try:
        if include_asset_manifest:
            if mode2_plan is None:
                mode2_plan = _run_with_heartbeat(
                    lambda: visual_client.analyze_mode2_plan(
                        duration=duration,
                        frames=director_frames,
                        transcript=None,
                    ),
                    log=log,
                    label="全局故事与资产理解",
                )
            if isinstance(mode2_plan, dict):
                scenes = mode2_plan.get("scenes") or []
        else:
            scenes = _run_with_heartbeat(
                lambda: scenes or visual_client.analyze_full_video(
                    director_frames,
                    batch_size=240,
                    overlap=8,
                    transcript=None,
                ),
                log=log,
                label="全局画面理解",
            )
    finally:
        visual_client.close()

    transcript: dict[str, Any] | None = None
    if audio_future is not None:
        try:
            audio_result = audio_future.result()
            transcript = audio_result.get("transcript")
            plan["audio_status"] = audio_result.get("status")
            plan["audio_understanding"] = bool(transcript)
            plan["transcript"] = transcript
            plan["audio_chunk_count"] = audio_result.get("chunk_count", 0)
            plan["audio_model"] = audio_result.get("model", "")
        except Exception as exc:  # noqa: BLE001
            plan["audio_status"] = "extraction_or_transcription_failed"
            plan["audio_error"] = str(exc)[:500]
            log(f"> [预导演] 音频理解失败，继续画面分析：{exc}")
        finally:
            if audio_executor is not None:
                audio_executor.shutdown(wait=True)

    scenes = _normalize_scenes(scenes or [], duration)
    scenes = _enrich_scene_roles_from_transcript(scenes, transcript)
    plan["scenes"] = scenes
    if include_asset_manifest and isinstance(mode2_plan, dict):
        asset_manifest = mode2_plan.get("asset_manifest")
        if isinstance(asset_manifest, dict) and isinstance(asset_manifest.get("results"), list):
            plan["asset_manifest"] = asset_manifest
    plan["boundary_hints"] = _boundary_hints(scenes, duration)
    plan["characters"] = _collect_characters(scenes)
    plan["key_actions"] = _collect_key_actions(scenes)
    visual_summary = "；".join(
        str(scene.get("description") or "").strip()
        for scene in scenes
        if str(scene.get("description") or "").strip()
    )
    audio_summary = str((transcript or {}).get("summary") or "").strip()
    planned_summary = (
        str(mode2_plan.get("story_summary") or "").strip()
        if include_asset_manifest and isinstance(mode2_plan, dict)
        else ""
    )
    plan["story_summary"] = (
        planned_summary
        or "；".join(value for value in (audio_summary, visual_summary) if value)
    )[:2000]
    plan["status"] = "ready" if scenes else "no_semantic_scenes"
    _write_plan(destination, plan)
    log(f"> [预导演] 完成：{len(scenes)} 个语义场景，{len(plan['boundary_hints'])} 个软切点")
    return plan


def apply_pre_director_boundaries(
    segments: list[dict[str, Any]],
    boundary_hints: list[dict[str, Any]],
    *,
    min_segment_duration: float,
    snap_tolerance: float = 0.45,
) -> list[dict[str, Any]]:
    result = [dict(segment) for segment in segments]
    minimum_piece = max(0.3, min(float(min_segment_duration), 1.0))
    for hint in sorted(boundary_hints, key=lambda item: float(item.get("time") or 0)):
        if float(hint.get("confidence") or 0) < 0.65:
            continue
        boundary = float(hint.get("time") or 0)
        existing = _nearest_boundary(result, boundary, snap_tolerance)
        if existing is not None:
            _mark_boundary(result, existing, hint)
            continue
        for index, segment in enumerate(result):
            start = float(segment.get("start") or 0)
            end = float(segment.get("end") or start)
            if not (start + minimum_piece <= boundary <= end - minimum_piece):
                continue
            left = dict(segment)
            right = dict(segment)
            left["end"] = boundary
            right["start"] = boundary
            left["end_sources"] = _append_source(left.get("end_sources"), "pre_director")
            right["start_sources"] = _append_source(right.get("start_sources"), "pre_director")
            left["pre_director_boundary"] = dict(hint)
            right["pre_director_boundary"] = dict(hint)
            result[index:index + 1] = [left, right]
            break
    return result


def collapse_transient_person_jitter(
    segments: list[dict[str, Any]],
    boundary_hints: list[dict[str, Any]],
    *,
    tolerance: float = 0.45,
    max_jitter_duration: float = 0.25,
) -> list[dict[str, Any]]:
    """Collapse a brief 1->many->1 detector spike near a semantic shot boundary."""
    result = [dict(segment) for segment in segments]
    semantic_times = [
        float(hint.get("time") or 0)
        for hint in boundary_hints
        if float(hint.get("confidence") or 0) >= 0.65
    ]
    index = 0
    while index + 2 < len(result):
        left, jitter, right = result[index:index + 3]
        jitter_duration = float(jitter.get("end") or 0) - float(jitter.get("start") or 0)
        outer_count = int(left.get("person_count", -1))
        same_shot = len({left.get("shot_index"), jitter.get("shot_index"), right.get("shot_index")}) == 1
        transient = bool(jitter.get("transient_multi_person"))
        yolo_only = (
            _is_transient_yolo_boundary(left.get("end_sources"), jitter.get("start_sources"))
            and _is_transient_yolo_boundary(jitter.get("end_sources"), right.get("start_sources"))
        )
        near_semantic_edge = any(
            min(
                abs(hint_time - float(left.get("start") or 0)),
                abs(hint_time - float(right.get("end") or 0)),
            ) <= tolerance
            for hint_time in semantic_times
        )
        if (
            same_shot
            and transient
            and yolo_only
            and 0 < jitter_duration <= max_jitter_duration
            and outer_count >= 0
            and outer_count == int(right.get("person_count", -1))
            and outer_count != int(jitter.get("person_count", -1))
            and near_semantic_edge
        ):
            merged = dict(left)
            merged["end"] = right["end"]
            merged["end_sources"] = list(right.get("end_sources") or [])
            merged["person_count"] = outer_count
            merged["transient_multi_person"] = False
            merged["collapsed_person_jitter"] = {
                "start": float(jitter.get("start") or 0),
                "end": float(jitter.get("end") or 0),
                "detected_person_count": int(jitter.get("person_count", -1)),
            }
            detections: list[Any] = []
            for item in (left, jitter, right):
                detections.extend(item.get("_all_frame_detections") or [])
            if detections:
                merged["_all_frame_detections"] = detections
            result[index:index + 3] = [merged]
            if index:
                index -= 1
            continue
        index += 1
    return result


def semantic_scene_for_time(plan: dict[str, Any] | None, time_seconds: float) -> dict[str, Any] | None:
    if not plan:
        return None
    scenes = plan.get("scenes") or []
    matches = [
        scene
        for scene in scenes
        if float(scene.get("start") or 0) <= time_seconds <= float(scene.get("end") or 0)
    ]
    if matches:
        return min(matches, key=lambda scene: float(scene.get("end") or 0) - float(scene.get("start") or 0))
    return None


def _select_director_frames(
    frames: list[FrameFeatures],
    duration: float,
    *,
    max_frames: int = 240,
) -> list[FrameFeatures]:
    if len(frames) <= 3:
        return list(frames)
    target_interval = 0.1 if duration <= 10 else 0.5 if duration <= 60 else 1.0
    target_count = min(max_frames, max(3, int(math.ceil(duration / target_interval)) + 1))
    if len(frames) <= target_count:
        return list(frames)
    indices = sorted({
        min(len(frames) - 1, int(round(index * (len(frames) - 1) / (target_count - 1))))
        for index in range(target_count)
    })
    return [frames[index] for index in indices]


def _audio_transcription_model(selected_model: str) -> str:
    if selected_model.strip().lower() == "qwen3.5-omni-plus":
        return "qwen3.5-omni-flash"
    return selected_model


def _enrich_scene_roles_from_transcript(
    scenes: list[dict[str, Any]],
    transcript: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    utterances = [
        item for item in (transcript or {}).get("utterances", [])
        if isinstance(item, dict) and str(item.get("role_name") or "").strip()
    ]
    result: list[dict[str, Any]] = []
    for raw_scene in scenes:
        scene = dict(raw_scene)
        start = float(scene.get("start") or 0)
        end = float(scene.get("end") or start)
        explicit_roles = {
            str(item.get("role_name") or "").strip()
            for item in utterances
            if min(end, float(item.get("end") or item.get("start") or 0))
            - max(start, float(item.get("start") or 0)) > 0
        }
        details = [
            dict(item) for item in scene.get("character_details", [])
            if isinstance(item, dict)
        ]
        for detail in details:
            if str(detail.get("role_name") or "").strip() not in explicit_roles:
                detail["role_name"] = ""
        if len(details) == 1 and len(explicit_roles) == 1:
            details[0]["role_name"] = next(iter(explicit_roles))
        scene["character_details"] = details
        result.append(scene)
    return result


def _run_with_heartbeat(
    action: Callable[[], Any],
    *,
    log: Callable[[str], None],
    label: str,
    interval: float = 10.0,
) -> Any:
    finished = threading.Event()
    started = time.monotonic()

    def heartbeat() -> None:
        while not finished.wait(interval):
            elapsed = int(time.monotonic() - started)
            log(f"> [预导演] {label}进行中，已等待 {elapsed}s...")

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        return action()
    finally:
        finished.set()


def _normalize_scenes(scenes: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in scenes:
        if not isinstance(raw, dict):
            continue
        start = max(0.0, min(float(raw.get("start") or 0), duration))
        end = max(start, min(float(raw.get("end") or duration), duration))
        if end - start < 0.05:
            continue
        result.append({
            **raw,
            "start": round(start, 3),
            "end": round(end, 3),
            "description": str(raw.get("description") or "").strip(),
            "characters": [str(value).strip() for value in raw.get("characters", []) if str(value).strip()],
            "key_action": str(raw.get("key_action") or "").strip(),
        })
    return sorted(result, key=lambda scene: (scene["start"], scene["end"]))


def _boundary_hints(scenes: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    for left, right in zip(scenes, scenes[1:]):
        boundary = (float(left["end"]) + float(right["start"])) / 2.0
        if boundary <= 0.05 or boundary >= duration - 0.05:
            continue
        right_kind = str(right.get("boundary_kind") or "")
        left_kind = str(left.get("boundary_kind") or "")
        kind = right_kind
        reason = str(right.get("boundary_reason") or "")
        if kind in {"", "video_start", "video_end"} and left_kind not in {"", "video_start", "video_end"}:
            kind = left_kind
            reason = str(left.get("boundary_reason") or reason)
        hints.append({
            "time": round(boundary, 3),
            "confidence": float(right.get("boundary_confidence") or left.get("boundary_confidence") or 0.75),
            "kind": kind or "semantic_scene_change",
            "reason": reason or str(right.get("description") or "语义场景变化"),
        })
    return hints


def _collect_characters(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline: dict[str, dict[str, Any]] = {}
    for scene in scenes:
        details = {
            str(item.get("visual_label") or "").strip(): item
            for item in scene.get("character_details", [])
            if isinstance(item, dict) and str(item.get("visual_label") or "").strip()
        }
        for label in scene.get("characters", []):
            detail = details.get(label, {})
            entry = timeline.setdefault(label, {
                "visual_label": label,
                "role_name": "",
                "role_candidates": [],
                "relationships": [],
                "description": "",
                "confidence": 0.0,
                "time_ranges": [],
            })
            entry["time_ranges"].append([float(scene["start"]), float(scene["end"])])
            confidence = float(detail.get("confidence") or 0)
            for candidate in _string_list(detail.get("role_candidates")):
                if candidate not in entry["role_candidates"]:
                    entry["role_candidates"].append(candidate)
            for relationship in _string_list(detail.get("relationships")):
                if relationship not in entry["relationships"]:
                    entry["relationships"].append(relationship)
            if confidence >= float(entry.get("confidence") or 0):
                entry["role_name"] = str(detail.get("role_name") or "").strip()
                entry["description"] = str(detail.get("description") or "").strip()
                entry["confidence"] = confidence
    return list(timeline.values())


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        values = []
    result: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _collect_key_actions(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "description": scene["key_action"],
            "start": scene["start"],
            "end": scene["end"],
        }
        for scene in scenes
        if scene.get("key_action")
    ]


def _merge_transcript_chunks(
    chunks: list[dict[str, Any] | None],
    *,
    chunk_seconds: float,
) -> dict[str, Any] | None:
    utterances: list[dict[str, Any]] = []
    summaries: list[str] = []
    language = ""
    for index, chunk in enumerate(chunks):
        if not chunk:
            continue
        language = language or str(chunk.get("language") or "")
        summary = str(chunk.get("summary") or "").strip()
        if summary:
            summaries.append(summary)
        offset = index * float(chunk_seconds)
        for raw in chunk.get("utterances") or []:
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            local_start = float(item.get("start") or 0)
            local_end = float(item.get("end") or local_start)
            item["start"] = round(local_start + offset, 3)
            item["end"] = round(local_end + offset, 3)
            utterances.append(item)
    if not utterances and not summaries:
        return None
    return {
        "language": language,
        "summary": "；".join(summaries),
        "utterances": utterances,
    }


def _nearest_boundary(
    segments: list[dict[str, Any]],
    target: float,
    tolerance: float,
) -> float | None:
    candidates: list[tuple[int, float, float]] = []
    for index, segment in enumerate(segments[:-1]):
        boundary = float(segment.get("end") or 0)
        distance = abs(boundary - target)
        if distance > tolerance:
            continue
        sources = set(segment.get("end_sources") or []) | set(segments[index + 1].get("start_sources") or [])
        if sources & {"omnishotcut", "pyscene"}:
            priority = 0
        elif sources and sources <= {"yolo", "yolo_transient_multi"} and "yolo_transient_multi" in sources:
            priority = 2
        else:
            priority = 1
        candidates.append((priority, distance, boundary))
    return min(candidates, default=(0, 0, None))[2]


def _is_transient_yolo_boundary(left_values: Any, right_values: Any) -> bool:
    sources = set(left_values or []) | set(right_values or [])
    return bool("yolo_transient_multi" in sources and sources <= {"yolo", "yolo_transient_multi"})


def _mark_boundary(segments: list[dict[str, Any]], boundary: float, hint: dict[str, Any]) -> None:
    for index in range(len(segments) - 1):
        if abs(float(segments[index].get("end") or 0) - boundary) > 1e-3:
            continue
        segments[index]["end_sources"] = _append_source(segments[index].get("end_sources"), "pre_director")
        segments[index + 1]["start_sources"] = _append_source(segments[index + 1].get("start_sources"), "pre_director")
        segments[index]["pre_director_boundary"] = dict(hint)
        segments[index + 1]["pre_director_boundary"] = dict(hint)
        return


def _append_source(values: Any, source: str) -> list[str]:
    result = [str(value) for value in (values or []) if str(value)]
    if source not in result:
        result.append(source)
    return result


def _write_plan(path: Path, plan: dict[str, Any]) -> None:
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
