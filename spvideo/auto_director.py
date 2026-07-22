from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable


AUTO_DIRECTOR_VERSION = 1
PERSON_SPECIAL_OPTIONS = (
    {"value": "passerby", "label": "路人，不替换"},
    {"value": "unknown", "label": "暂时无法判断"},
)
OBJECT_POLICY_OPTIONS = (
    {"value": "preserve", "label": "保留原物"},
    {"value": "redraw", "label": "跟随角色重绘"},
    {"value": "replace", "label": "替换为新物品"},
    {"value": "remove", "label": "删除"},
    {"value": "unknown", "label": "暂时无法判断"},
)


def load_auto_director(project_dir: str | Path) -> dict[str, Any]:
    path = _plan_path(project_dir)
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def resolve_auto_director_project_root(project_dir: str | Path) -> Path:
    value = str(project_dir or "").strip()
    if not value:
        raise ValueError("missing_project_dir")
    path = Path(value)
    if path.is_file():
        path = path.parent
    candidates = [path, *list(path.parents)[:8]]
    for candidate in candidates:
        if (candidate / "manifest.json").exists():
            return candidate
        if (candidate / "01_分析探针" / "two_pass_result.json").exists():
            return candidate
        if (candidate / "01_probe" / "two_pass_result.json").exists():
            return candidate
    if path.exists() and path.is_dir():
        child = _latest_child_project(path)
        if child is not None:
            return child
    if path.exists() and path.is_dir():
        return path
    raise ValueError(f"project_dir_not_found: {value}")


def _latest_child_project(path: Path) -> Path | None:
    candidates: list[Path] = []
    try:
        children = list(path.iterdir())
    except OSError:
        return None
    for child in children:
        if not child.is_dir():
            continue
        if (child / "manifest.json").exists():
            candidates.append(child)
            continue
        if (child / "01_分析探针" / "two_pass_result.json").exists():
            candidates.append(child)
            continue
        if (child / "01_probe" / "two_pass_result.json").exists():
            candidates.append(child)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def save_auto_director(project_dir: str | Path, plan: dict[str, Any]) -> dict[str, Any]:
    path = _plan_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan


def analyze_auto_director_project(
    project_dir: str | Path,
    *,
    use_story_model: bool = False,
    api_key: str = "",
    base_url: str = "",
    model: str = "",
    scan_faces: bool = True,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    root = resolve_auto_director_project_root(project_dir)
    manifest = _load_analysis_manifest(root)
    if not isinstance(manifest.get("segments"), list):
        raise ValueError("auto_director_segments_not_found")

    from .asset_store import load_asset_store
    import threading

    assets = load_asset_store(root)
    old_plan = load_auto_director(root)
    pre_director_story = _load_pre_director_story(root, manifest)
    log = on_progress or (lambda _message: None)
    face_entities: list[dict[str, Any]] = []
    scan_notes: list[str] = []

    # ── 人脸聚类和视觉故事并行 ──────────────────────────────────────
    story_result: dict[str, Any] = pre_director_story or {"status": "not_requested"}
    story_error: str | None = None

    def _run_face_scan():
        nonlocal face_entities
        if scan_faces:
            log("> [人脸] 正在按分镜代表帧聚类主要人物...")
            entities, note = _scan_face_entities(
                manifest.get("segments", []),
                assets.get("originals", []),
                on_progress=lambda msg: log(f"> [人脸] {msg.removeprefix('> ')}"),
            )
            face_entities = entities
            if note:
                scan_notes.append(note)
        else:
            scan_notes.append("已跳过人脸聚类")

    def _run_story():
        nonlocal story_result, story_error
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        if not key:
            story_result["status"] = "missing_api_key"
            scan_notes.append("未配置视觉故事模型 API Key")
            return
        log(f"> [故事] 正在用 {model or 'Qwen-Omni'} 反推视觉故事...")
        try:
            story_result = _analyze_visual_story(
                manifest,
                api_key=key,
                base_url=base_url,
                model=model,
            )
            log("> [故事] 视觉故事分析完成")
        except Exception as exc:  # noqa: BLE001
            story_result["status"] = "failed"
            story_result["summary"] = ""
            story_error = str(exc)[:500]
            scan_notes.append(f"视觉故事分析失败: {story_error}")

    if pre_director_story:
        scan_notes.append("已复用全局预导演故事、音频和角色时间线")
        _run_face_scan()
    elif use_story_model:
        t1 = threading.Thread(target=_run_face_scan, daemon=True)
        t2 = threading.Thread(target=_run_story, daemon=True)
        t1.start(); t2.start()
        t1.join(); t2.join()
    else:
        _run_face_scan()

    story: dict[str, Any] = {
        "status": "not_requested",
        "summary": "",
        "source": "pre_director_or_representative_frames",
        "audio_understanding": False,
        "characters": [],
        "important_clues": [],
        "key_actions": [],
    }
    if story_result.get("status") != "not_requested":
        story = story_result
        if story_result.get("status") == "failed":
            story["status"] = "failed"
            story["summary"] = f"分析失败: {story_error}" if story_error else "分析失败"

    plan = compose_auto_director_plan(
        project_dir=root,
        manifest=manifest,
        assets=assets,
        face_entities=face_entities,
        story=story,
        old_plan=old_plan,
        scan_notes=scan_notes,
    )
    save_auto_director(root, plan)
    log(
        f"> 自动导演完成：{plan['stats']['question_count']} 个问题，"
        f"待确认 {plan['stats']['pending_count']} 个"
    )
    return plan


def compose_auto_director_plan(
    *,
    project_dir: str | Path,
    manifest: dict[str, Any],
    assets: dict[str, Any],
    face_entities: list[dict[str, Any]] | None = None,
    story: dict[str, Any] | None = None,
    old_plan: dict[str, Any] | None = None,
    scan_notes: list[str] | None = None,
) -> dict[str, Any]:
    root = Path(project_dir)
    segments = [dict(item) for item in manifest.get("segments", []) if isinstance(item, dict)]
    characters = [
        dict(item)
        for item in (assets.get("characters") or {}).values()
        if str(item.get("name") or "").strip()
    ]
    role_names = _unique_nonempty(str(item["name"]).strip() for item in characters)
    directors = assets.get("directors") or {}
    story_value = _normalize_story(story or {})
    for item in story_value.get("characters", []):
        for candidate in _story_character_candidate_names(item, include_visual_label=True):
            if candidate not in role_names:
                role_names.append(candidate)
    entities = [dict(item) for item in (face_entities or [])]
    _merge_story_characters(entities, story_value.get("characters", []), segments)

    covered_segments: set[str] = set()
    questions: list[dict[str, Any]] = []
    person_options = [{"value": name, "label": name} for name in role_names]
    person_options.extend(dict(item) for item in PERSON_SPECIAL_OPTIONS)

    for index, entity in enumerate(entities, 1):
        segment_ids = _clean_segment_ids(entity.get("segment_ids"), segments)
        if not segment_ids:
            continue
        entity["id"] = str(entity.get("id") or f"person_{index:02d}")
        entity["type"] = "person"
        entity["segment_ids"] = segment_ids
        entity["preview_frames"] = _clean_preview_frames(entity.get("preview_frames"), segments, segment_ids)
        entity.setdefault("visual_label", f"人物{_alpha_label(index)}")
        covered_segments.update(segment_ids)
        question_key = f"person_identity:{entity['id']}"
        questions.append({
            "id": _stable_id(question_key),
            "key": question_key,
            "kind": "person_identity",
            "priority": "high" if len(segment_ids) > 1 else "normal",
            "prompt": f"{entity['visual_label']}是谁？",
            "detail": str(entity.get("description") or f"出现在 {len(segment_ids)} 个分镜中"),
            "options": person_options,
            "multiple": False,
            "segment_ids": segment_ids,
            "preview_frames": entity["preview_frames"],
            "entity_id": entity["id"],
            "suggested_answer": str(entity.get("suggested_role") or ""),
            "suggestion_confidence": float(entity.get("suggestion_confidence") or 0.0),
            "candidate_roles": _unique_nonempty([
                str(entity.get("suggested_role") or ""),
                *_string_list(entity.get("role_candidates")),
            ]),
            "status": "pending",
            "answer": None,
            "source": str(entity.get("source") or "face_cluster"),
        })

    for segment in segments:
        segment_id = str(segment.get("segment_id") or "")
        if not segment_id:
            continue
        plan = directors.get(segment_id) or {}
        if _is_multi_person(segment) and plan.get("status") != "ready":
            question_key = f"multi_mapping:{segment_id}"
            questions.append({
                "id": _stable_id(question_key),
                "key": question_key,
                "kind": "multi_role_mapping",
                "priority": "critical",
                "prompt": f"片段 {segment_id} 中的多人分别是谁？",
                "detail": "这是整个片段的多人风险，不是单个人；请进入片段导演，在视频画面上按角色逐个打点标记，避免左右/颜色顺序互换",
                "candidate_roles": role_names,
                "options": [
                    {"value": "manual_director", "label": "进入片段导演"},
                    {"value": "keep_original", "label": "保留原片，不替换"},
                    {"value": "unknown", "label": "暂时无法判断"},
                ],
                "multiple": False,
                "segment_ids": [segment_id],
                "preview_frames": _clean_preview_frames([], segments, [segment_id]),
                "status": "pending",
                "answer": None,
                "source": "segment_risk",
            })

    unresolved = [
        segment for segment in segments
        if _segment_has_human(segment)
        and str(segment.get("segment_id") or "") not in covered_segments
        and not (directors.get(str(segment.get("segment_id") or "")) or {}).get("roles")
    ]
    for segment in unresolved:
        segment_id = str(segment.get("segment_id") or "")
        person_count = int(segment.get("person_count") or 0)
        # 只有真的有人（person_count>0）才问角色问题；-1 表示未检测也跳过
        if person_count <= 0:
            continue
        segment_ids = [segment_id]
        question_key = "segment_roles:" + segment_id
        cluster_count = int(segment.get("person_count") or 0)
        detail_parts = []
        if cluster_count <= 0:
            detail_parts.append("这个镜头没有清晰、可聚类的人脸")
        else:
            detail_parts.append(f"YOLO 检测到 {cluster_count} 人")
        questions.append({
            "id": _stable_id(question_key),
            "key": question_key,
            "kind": "segment_roles",
            "priority": "normal",
            "prompt": f"片段 {segment_id} 中出现了哪些角色？",
            "detail": "；".join(detail_parts) or None,
            "candidate_roles": role_names,
            "options": [
                *[{"value": name, "label": name} for name in role_names],
                {"value": "no_person", "label": "无人物（纯场景/空镜）"},
                {"value": "keep_original", "label": "不替换人物"},
                {"value": "unknown", "label": "暂时无法判断"},
            ],
            "multiple": True,
            "segment_ids": segment_ids,
            "preview_frames": _clean_preview_frames([], segments, segment_ids),
            "status": "pending",
            "answer": None,
            "source": "unresolved_frames",
        })

    for clue_index, clue in enumerate(story_value.get("important_clues", []), 1):
        if str(clue.get("kind") or "").lower() not in {"object", "prop", "item"}:
            continue
        if clue.get("needs_confirmation") is False:
            continue
        description = str(clue.get("description") or f"重要物品{clue_index}").strip()
        segment_ids = _clean_segment_ids(clue.get("segment_ids"), segments)
        question_key = f"object_policy:{description}:{','.join(segment_ids)}"
        questions.append({
            "id": _stable_id(question_key),
            "key": question_key,
            "kind": "object_policy",
            "priority": "high",
            "prompt": str(clue.get("question") or f"“{description}”需要怎么处理？"),
            "detail": str(clue.get("why_important") or "故事模型认为它可能是重要线索"),
            "options": [dict(item) for item in OBJECT_POLICY_OPTIONS],
            "multiple": False,
            "segment_ids": segment_ids,
            "preview_frames": _clean_preview_frames([], segments, segment_ids),
            "clue_id": str(clue.get("id") or f"clue_{clue_index:02d}"),
            "status": "pending",
            "answer": None,
            "source": "visual_story",
        })

    plan: dict[str, Any] = {
        "version": AUTO_DIRECTOR_VERSION,
        "project_dir": str(root),
        "source_path": str((manifest.get("meta") or {}).get("source_path") or ""),
        "status": "ready",
        "generated_at": time.time(),
        "updated_at": time.time(),
        "scan_notes": list(scan_notes or []),
        "story": story_value,
        "entities": entities,
        "questions": _dedupe_questions(questions),
        "segment_decisions": {},
        "object_decisions": {},
        "stats": {},
    }
    _restore_answers(plan, old_plan or {})
    _refresh_stats(plan)
    return plan


def answer_auto_director_question(
    project_dir: str | Path,
    *,
    question_id: str,
    answer: Any,
) -> dict[str, Any]:
    plan = load_auto_director(project_dir)
    if not plan:
        raise ValueError("auto_director_not_analyzed")
    question = next(
        (item for item in plan.get("questions", []) if str(item.get("id") or "") == question_id),
        None,
    )
    if question is None:
        raise ValueError("auto_director_question_not_found")
    normalized = _validate_answer(question, answer)
    question["answer"] = normalized
    question["status"] = "answered"
    question["answered_at"] = time.time()
    _propagate_answer(plan, question, normalized)
    plan["updated_at"] = time.time()
    _refresh_stats(plan)
    return save_auto_director(project_dir, plan)


def _scan_face_entities(
    segments: list[dict[str, Any]],
    originals: list[dict[str, Any]],
    *,
    on_progress: Callable[[str], None],
    similarity_threshold: float = 0.55,
) -> tuple[list[dict[str, Any]], str]:
    try:
        import numpy as np
        from .face_identity import _get_face_app, _read_image
    except Exception as exc:  # noqa: BLE001
        return [], f"人脸聚类不可用: {exc}"

    try:
        import onnxruntime as _ort
        providers = _ort.get_available_providers()
        on_progress(f"> 人脸引擎可用: {', '.join(p for p in providers[:3] if 'CUDA' in p or 'CPU' in p)}")
    except Exception:
        on_progress("> 人脸引擎: 无法检测 onnxruntime")

    clusters: list[dict[str, Any]] = []
    candidates = [item for item in segments if _segment_has_human(item) and _representative_frame(item)]
    max_candidates = min(80, len(candidates))

    app = _get_face_app()
    for index, segment in enumerate(candidates[:max_candidates], 1):
        frame = _representative_frame(segment)
        img = _read_image(frame)
        if img is None:
            continue
        ih, iw = img.shape[:2]
        faces = app.get(img)
        if not faces:
            continue
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        if best.bbox[2] <= best.bbox[0] or best.bbox[3] <= best.bbox[1]:
            continue
        embedding = best.normed_embedding
        bbox = [round(float(best.bbox[0]) / iw, 4), round(float(best.bbox[1]) / ih, 4),
                round(float(best.bbox[2]) / iw, 4), round(float(best.bbox[3]) / ih, 4)]
        best_cluster = None
        best_similarity = -1.0
        for cluster in clusters:
            similarity = float(np.dot(cluster["centroid"], embedding))
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster
        if best_cluster is None or best_similarity < similarity_threshold:
            best_cluster = {
                "centroid": embedding.copy(),
                "embeddings": [embedding],
                "segment_ids": [],
                "preview_frames": [],
            }
            clusters.append(best_cluster)
        else:
            best_cluster["embeddings"].append(embedding)
            centroid = np.mean(best_cluster["embeddings"], axis=0)
            norm = float(np.linalg.norm(centroid)) or 1.0
            best_cluster["centroid"] = centroid / norm
        segment_id = str(segment.get("segment_id") or "")
        if segment_id and segment_id not in best_cluster["segment_ids"]:
            best_cluster["segment_ids"].append(segment_id)
        preview_entry = {"path": frame, "bbox": bbox}
        if not any(p.get("path") == frame for p in best_cluster["preview_frames"]) and len(best_cluster["preview_frames"]) < 4:
            best_cluster["preview_frames"].append(preview_entry)
        if index % 5 == 0:
            on_progress(f"> 已检查 {index}/{max_candidates} 帧，当前聚类 {len(clusters)} 个")

    anchors: list[dict[str, Any]] = []
    for item in originals:
        name = str(item.get("label_name") or "").strip()
        path = str(item.get("cutout_path") or item.get("crop_path") or "").strip()
        if not name or not path or not Path(path).exists():
            continue
        try:
            embedding = get_face_embedding(path)
        except Exception:
            embedding = None
        if embedding is not None:
            anchors.append({"name": name, "embedding": embedding})

    entities: list[dict[str, Any]] = []
    for index, cluster in enumerate(clusters, 1):
        suggested_role = ""
        suggestion_confidence = 0.0
        for anchor in anchors:
            similarity = float(np.dot(cluster["centroid"], anchor["embedding"]))
            if similarity > suggestion_confidence:
                suggestion_confidence = similarity
                suggested_role = anchor["name"]
        if suggestion_confidence < similarity_threshold:
            suggested_role = ""
        entities.append({
            "id": f"face_{index:02d}",
            "type": "person",
            "visual_label": f"人物{_alpha_label(index)}",
            "description": f"人脸聚类覆盖 {len(cluster['segment_ids'])} 个分镜",
            "segment_ids": cluster["segment_ids"],
            "preview_frames": cluster["preview_frames"],
            "suggested_role": suggested_role,
            "suggestion_confidence": round(suggestion_confidence, 3),
            "source": "face_cluster",
        })
    return entities, "" if entities else "代表帧中没有检测到可聚类人脸"


def _analyze_visual_story(
    manifest: dict[str, Any],
    *,
    api_key: str,
    base_url: str,
    model: str,
) -> dict[str, Any]:
    from .gemini_analyzer import DEFAULT_BASE_URL, DEFAULT_MODEL, GeminiClient

    segments = [item for item in manifest.get("segments", []) if _representative_frame(item)]
    selected = _evenly_select(segments, 18)
    if not selected:
        raise ValueError("auto_director_story_frames_not_found")
    frame_paths = [_representative_frame(item) for item in selected]
    timeline = "\n".join(
        f"片段{item.get('segment_id')}，{float(item.get('start') or 0):.2f}s-"
        f"{float(item.get('end') or 0):.2f}s"
        for item in selected
    )

    audio_note = "本次只提供代表帧，没有音频；只能描述外观，不得把猜测当作角色真名"

    prompt = (
        f"你是短剧导演。以下来自同一部短剧，按时间排列。{audio_note}。\n\n"
        f"时间对应：\n{timeline}\n\n"
        "请综合分析并反推：\n"
        "1. 每个出场人物的视觉特征、声音特征、可能的角色名（从对白中推断，如'婆婆''女主'）\n"
        "2. 打人/递东西/拿手机等关键动作\n"
        "3. 后续改编需关注的物品（手机/证据/首饰/包/药物/文件等）\n"
        "输出严格 JSON：\n"
        '{"summary":"2-5句中文","characters":[{"id":"p1","visual_label":"婆婆",'
        '"description":"外观和声音","segment_ids":["001"]}],'
        '"key_actions":[{"description":"","segment_ids":[],"importance":"high"}],'
        '"important_clues":[{"id":"c1","kind":"object","description":"手机","segment_ids":[],'
        '"why_important":"","needs_confirmation":true,"question":""}],'
        '"uncertainties":[]}'
    )

    client = GeminiClient(
        api_key=api_key,
        base_url=base_url or DEFAULT_BASE_URL,
        model=model or DEFAULT_MODEL,
        request_timeout=240,
    )
    try:
        result = client.analyze_segment_keyframes(
            frame_paths,
            start=float(selected[0].get("start") or 0),
            end=float(selected[-1].get("end") or 0),
            retry=1,
            prompt_override=prompt,
        )
    finally:
        client.close()
    if not isinstance(result, dict):
        raise ValueError("auto_director_story_analysis_failed")
    result["status"] = "ready"
    result["source"] = "representative_frames"
    result["audio_understanding"] = False
    result["sampled_segment_ids"] = [str(item.get("segment_id") or "") for item in selected]
    return _normalize_story(result)


def _merge_story_characters(
    entities: list[dict[str, Any]],
    story_characters: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> None:
    for index, character in enumerate(story_characters, 1):
        segment_ids = _clean_segment_ids(character.get("segment_ids"), segments)
        if not segment_ids:
            continue
        best = None
        best_overlap = 0
        story_set = set(segment_ids)
        for entity in entities:
            overlap = len(story_set.intersection(entity.get("segment_ids") or []))
            if overlap > best_overlap:
                best = entity
                best_overlap = overlap
        if best is not None:
            best["visual_label"] = str(character.get("visual_label") or best.get("visual_label") or "")
            best["description"] = str(character.get("description") or best.get("description") or "")
            best["source"] = "face_cluster+visual_story"
            best["role_candidates"] = _unique_nonempty([
                *_string_list(best.get("role_candidates")),
                *_story_character_candidate_names(character, include_visual_label=False),
            ])
            role_name = str(character.get("role_name") or "").strip()
            confidence = float(character.get("confidence") or 0)
            if role_name and confidence >= 0.6:
                best["suggested_role"] = role_name
                best["suggestion_confidence"] = confidence
            for segment_id in segment_ids:
                if segment_id not in best.setdefault("segment_ids", []):
                    best["segment_ids"].append(segment_id)
            continue
        entities.append({
            "id": str(character.get("id") or f"story_person_{index:02d}"),
            "type": "person",
            "visual_label": str(character.get("visual_label") or f"人物{_alpha_label(len(entities) + 1)}"),
            "description": str(character.get("description") or "故事模型识别的人物"),
            "segment_ids": segment_ids,
            "preview_frames": [],
            "source": "visual_story",
            "role_candidates": _story_character_candidate_names(character, include_visual_label=False),
            "suggested_role": str(character.get("role_name") or "") if float(character.get("confidence") or 0) >= 0.6 else "",
            "suggestion_confidence": (
                float(character.get("confidence") or 0)
                if str(character.get("role_name") or "").strip()
                else 0.0
            ),
        })


def _restore_answers(plan: dict[str, Any], old_plan: dict[str, Any]) -> None:
    old_by_key = {
        str(item.get("key") or ""): item
        for item in old_plan.get("questions", [])
        if item.get("key") and item.get("status") == "answered"
    }
    for question in plan.get("questions", []):
        old = old_by_key.get(str(question.get("key") or ""))
        if old is None:
            continue
        try:
            answer = _validate_answer(question, old.get("answer"))
        except ValueError:
            continue
        question["answer"] = answer
        question["status"] = "answered"
        question["answered_at"] = old.get("answered_at") or time.time()
        _propagate_answer(plan, question, answer)


def _propagate_answer(plan: dict[str, Any], question: dict[str, Any], answer: Any) -> None:
    kind = str(question.get("kind") or "")
    segment_ids = [str(value) for value in question.get("segment_ids", []) if str(value)]
    role_values: list[str] = []
    if kind == "person_identity" and isinstance(answer, str):
        entity_id = str(question.get("entity_id") or "")
        for entity in plan.get("entities", []):
            if str(entity.get("id") or "") == entity_id:
                entity["resolved_as"] = answer
        if answer not in {"passerby", "unknown", "keep_original"}:
            role_values = [answer]
    elif kind == "segment_roles" and isinstance(answer, list):
        role_values = [
            value for value in answer
            if value not in {"unknown", "keep_original", "no_person"}
        ]

    for segment_id in segment_ids:
        decision = plan.setdefault("segment_decisions", {}).setdefault(
            segment_id,
            {"suggested_roles": [], "source_questions": []},
        )
        if question["id"] not in decision["source_questions"]:
            decision["source_questions"].append(question["id"])
        for role in role_values:
            if role not in decision["suggested_roles"]:
                decision["suggested_roles"].append(role)
        if kind == "multi_role_mapping":
            decision["policy"] = answer
        elif kind == "segment_roles" and isinstance(answer, list) and "no_person" in answer:
            decision["suggested_roles"] = []
            decision["policy"] = "no_person"
        elif kind == "segment_roles" and isinstance(answer, list) and "keep_original" in answer:
            decision["policy"] = "keep_original"
        elif kind == "person_identity" and answer == "passerby":
            decision.setdefault("ignored_entities", []).append(str(question.get("entity_id") or ""))

    if kind == "object_policy":
        clue_id = str(question.get("clue_id") or question.get("id") or "")
        plan.setdefault("object_decisions", {})[clue_id] = {
            "policy": answer,
            "segment_ids": segment_ids,
            "question_id": question.get("id"),
        }


def _validate_answer(question: dict[str, Any], answer: Any) -> Any:
    allowed = {str(item.get("value") or "") for item in question.get("options", [])}
    if question.get("multiple") or str(question.get("kind") or "") == "multi_role_mapping":
        if not isinstance(answer, list):
            answer = [answer] if answer else []
        values: list[str] = []
        for raw in answer:
            value = str(raw or "").strip()
            if value and value not in values:
                values.append(value)
        if not values:
            raise ValueError("auto_director_answer_required")
        return values
    value = str(answer or "").strip()
    if not value:
        raise ValueError("auto_director_answer_invalid")
    if value not in allowed:
        if str(question.get("kind") or "") != "person_identity":
            raise ValueError("auto_director_answer_invalid")
    return value


def _refresh_stats(plan: dict[str, Any]) -> None:
    questions = plan.get("questions", [])
    answered = sum(1 for item in questions if item.get("status") == "answered")
    critical = sum(
        1 for item in questions
        if item.get("status") != "answered" and item.get("priority") == "critical"
    )
    plan["stats"] = {
        "entity_count": len(plan.get("entities", [])),
        "question_count": len(questions),
        "answered_count": answered,
        "pending_count": len(questions) - answered,
        "critical_pending_count": critical,
        "story_ready": plan.get("story", {}).get("status") == "ready",
    }


def _normalize_story(story: dict[str, Any]) -> dict[str, Any]:
    result = {
        "status": str(story.get("status") or "not_requested"),
        "summary": str(story.get("summary") or story.get("story_summary") or "").strip(),
        "source": str(story.get("source") or "representative_frames"),
        "audio_understanding": bool(story.get("audio_understanding", False)),
        "characters": [dict(item) for item in story.get("characters", []) if isinstance(item, dict)],
        "important_clues": [dict(item) for item in story.get("important_clues", []) if isinstance(item, dict)],
        "key_actions": [dict(item) for item in story.get("key_actions", []) if isinstance(item, dict)],
        "uncertainties": [str(item) for item in story.get("uncertainties", []) if str(item).strip()],
    }
    for key in ("error", "sampled_segment_ids"):
        if key in story:
            result[key] = story[key]
    for key in ("audio_status", "transcript"):
        if key in story:
            result[key] = story[key]
    return result


def _story_character_candidate_names(
    character: dict[str, Any],
    *,
    include_visual_label: bool,
) -> list[str]:
    values: list[str] = [
        str(character.get("role_name") or ""),
        *_string_list(character.get("role_candidates")),
        *_string_list(character.get("candidate_roles")),
        *_string_list(character.get("relationship_roles")),
    ]
    if include_visual_label:
        values.append(str(character.get("visual_label") or ""))
    return _unique_nonempty(values)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        values = []
    return _unique_nonempty(str(item or "").strip() for item in values)


def _unique_nonempty(values: Any) -> list[str]:
    result: list[str] = []
    for raw in values:
        value = str(raw or "").strip()
        if value and value not in result:
            result.append(value)
    return result


def _load_pre_director_story(
    root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    candidates = [
        root / "01_分析探针" / "pre_director.json",
        root / "01_probe" / "pre_director.json",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return None
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(plan, dict) or plan.get("status") != "ready":
        return None

    segments = [item for item in manifest.get("segments", []) if isinstance(item, dict)]
    characters: list[dict[str, Any]] = []
    for index, raw in enumerate(plan.get("characters") or [], 1):
        if not isinstance(raw, dict):
            continue
        time_ranges = raw.get("time_ranges") or []
        segment_ids = [
            str(segment.get("segment_id") or "")
            for segment in segments
            if str(segment.get("segment_id") or "")
            and _segment_overlaps_time_ranges(segment, time_ranges, use_midpoint=True)
        ]
        if not segment_ids:
            continue
        characters.append({
            "id": str(raw.get("id") or f"pre_person_{index:02d}"),
            "visual_label": str(raw.get("visual_label") or f"人物{_alpha_label(index)}"),
            "role_name": str(raw.get("role_name") or ""),
            "role_candidates": _string_list(raw.get("role_candidates")),
            "relationships": _string_list(raw.get("relationships")),
            "description": str(raw.get("description") or "全局预导演识别的人物"),
            "confidence": float(raw.get("confidence") or 0),
            "segment_ids": segment_ids,
            "time_ranges": time_ranges,
        })

    key_actions = []
    for action in plan.get("key_actions") or []:
        if not isinstance(action, dict):
            continue
        mapped = dict(action)
        mapped["segment_ids"] = [
            str(segment.get("segment_id") or "")
            for segment in segments
            if str(segment.get("segment_id") or "")
            and _segment_overlaps_time_ranges(
                segment,
                [[action.get("start"), action.get("end")]],
                use_midpoint=False,
            )
        ]
        key_actions.append(mapped)

    return {
        "status": "ready",
        "summary": str(plan.get("story_summary") or ""),
        "source": "pre_director",
        "audio_understanding": bool(plan.get("audio_understanding", False)),
        "audio_status": str(plan.get("audio_status") or "unknown"),
        "transcript": plan.get("transcript"),
        "characters": characters,
        "important_clues": [],
        "key_actions": key_actions,
        "uncertainties": [],
    }


def _segment_overlaps_time_ranges(
    segment: dict[str, Any],
    ranges: Any,
    *,
    use_midpoint: bool,
) -> bool:
    start = float(segment.get("start") or 0)
    end = float(segment.get("end") or start)
    midpoint = (start + end) / 2.0
    for value in ranges if isinstance(ranges, list) else []:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        range_start = float(value[0] or 0)
        range_end = float(value[1] or range_start)
        if use_midpoint and range_start <= midpoint <= range_end:
            return True
        if not use_midpoint and min(end, range_end) - max(start, range_start) > 0.01:
            return True
    return False


def _clean_segment_ids(values: Any, segments: list[dict[str, Any]]) -> list[str]:
    available = {str(item.get("segment_id") or "") for item in segments}
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for raw in values:
        value = str(raw or "").strip()
        if value in available and value not in result:
            result.append(value)
    return result


def _clean_preview_frames(
    values: Any,
    segments: list[dict[str, Any]],
    segment_ids: list[str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for raw in values if isinstance(values, list) else []:
        if isinstance(raw, dict):
            entry = dict(raw)
        else:
            entry = {"path": str(raw or ""), "bbox": None}
        path = str(entry.get("path") or "").strip()
        if path and Path(path).exists() and path not in seen_paths:
            result.append(entry)
            seen_paths.add(path)
    by_id = {str(item.get("segment_id") or ""): item for item in segments}
    for segment_id in segment_ids:
        frame = _representative_frame(by_id.get(segment_id, {}))
        if frame and Path(frame).exists() and frame not in seen_paths:
            result.append({"path": frame, "bbox": None})
            seen_paths.add(frame)
        if len(result) >= 4:
            break
    return result[:4]


def _segment_has_human(segment: dict[str, Any]) -> bool:
    if str(segment.get("segment_type") or "") in {"with_human", "human_driver", "human_composite"}:
        return True
    if segment.get("needs_ai_driver"):
        return True
    try:
        return int(segment.get("person_count", -1)) != 0
    except (TypeError, ValueError):
        return True


def _is_multi_person(segment: dict[str, Any]) -> bool:
    if segment.get("transient_multi_person"):
        return True
    try:
        return int(segment.get("person_count", -1)) >= 2
    except (TypeError, ValueError):
        return False


def _representative_frame(segment: dict[str, Any]) -> str:
    return str(segment.get("representative_frame") or "").strip()


def _evenly_select(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(items) <= limit:
        return items
    indices = sorted({min(len(items) - 1, int(round(i * (len(items) - 1) / (limit - 1)))) for i in range(limit)})
    return [items[index] for index in indices]


def _dedupe_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    priority_order = {"critical": 0, "high": 1, "normal": 2}
    for question in sorted(questions, key=lambda item: priority_order.get(str(item.get("priority")), 9)):
        key = str(question.get("key") or question.get("id") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(question)
    return result


def _stable_id(value: str) -> str:
    return "q_" + hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _alpha_label(index: int) -> str:
    index = max(1, index)
    label = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        label = chr(65 + remainder) + label
    return label


def _plan_path(project_dir: str | Path) -> Path:
    return resolve_auto_director_project_root(project_dir) / "assets" / "auto_director.json"


def _load_analysis_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest, dict):
            return manifest

    probe_dir = root / "01_分析探针"
    if not probe_dir.exists():
        probe_dir = root / "01_probe"
    two_pass_path = probe_dir / "two_pass_result.json"
    if not two_pass_path.exists():
        raise ValueError("auto_director_result_not_found")
    two_pass = json.loads(two_pass_path.read_text(encoding="utf-8"))
    raw_segments = two_pass.get("sub_segments") or two_pass.get("person_segments") or []
    clips_dir = root / "02_分镜片段" / "00_all_mp4_clips"
    if not clips_dir.exists():
        clips_dir = root / "02_segments" / "00_all_mp4_clips"
    frames = _discover_probe_frames(probe_dir)
    segments: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_segments, 1):
        if not isinstance(raw, dict):
            continue
        segment_id = str(raw.get("segment_id") or f"{index:03d}")
        start = float(raw.get("start") or 0.0)
        end = float(raw.get("end") or start)
        clip = next(clips_dir.glob(f"{segment_id}_*.mp4"), None) if clips_dir.exists() else None
        representative = _nearest_probe_frame(frames, (start + end) / 2.0)
        segments.append({
            **raw,
            "segment_id": segment_id,
            "start": start,
            "end": end,
            "duration": max(0.0, end - start),
            "output_path": str(clip) if clip else str(raw.get("output_path") or ""),
            "representative_frame": representative,
            "segment_type": str(raw.get("segment_type") or (
                "without_human" if raw.get("person_count") == 0 else "with_human"
            )),
        })
    meta = _read_optional_json(root / "00_原始视频" / "original_meta.json")
    if not meta:
        meta = _read_optional_json(root / "00_source" / "original_meta.json")
    return {"version": 1, "meta": meta, "segments": segments, "frames": []}


def _discover_probe_frames(probe_dir: Path) -> list[tuple[float, str]]:
    frames_dir = probe_dir / "frames_1fps"
    if not frames_dir.exists():
        return []
    result: list[tuple[float, str]] = []
    for path in frames_dir.glob("*.jpg"):
        stem = path.stem
        try:
            time_value = float(stem.rsplit("_", 1)[-1].removesuffix("s"))
        except ValueError:
            continue
        result.append((time_value, str(path)))
    return sorted(result)


def _nearest_probe_frame(frames: list[tuple[float, str]], target: float) -> str:
    if not frames:
        return ""
    return min(frames, key=lambda item: abs(item[0] - target))[1]


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}
