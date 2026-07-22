"""Mode 2 visual asset curation with cache-first Qwen Omni review.

This module is intentionally independent from Mode 1. It converts visual
groups into conservative asset candidates; it does not bind or render them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageOps


SCHEMA_VERSION = "mode2.asset_curator.v1"
PROMPT_VERSION = "mode2.asset_curator.prompt.v1"
DEFAULT_CURATOR_MODEL = "qwen3.5-omni-flash"
MIN_ASSET_CONFIDENCE = 0.65
MIN_PROP_CONFIDENCE = 0.75
ALLOWED_KINDS = frozenset({"role", "scene", "prop", "mixed", "ignore"})

_MANIFEST_LOCK = threading.Lock()


def curate_visual_groups(
    visual_groups: Sequence[Mapping[str, Any]],
    *,
    cache_dir: str | Path,
    asset_manifest: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    story_context: str | Mapping[str, Any] | None = None,
    known_roles: Sequence[str | Mapping[str, Any]] | None = None,
    client: Any | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str = DEFAULT_CURATOR_MODEL,
    allow_single_group_fallback: bool = True,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Curate all visual groups while minimizing model calls.

    Resolution order is strict:

    1. Normalize matching entries from ``asset_manifest`` with zero calls.
    2. Reuse exact frame/model/prompt cache entries.
    3. Send every remaining group contact sheet in one batch call.
    4. Retry only missing, malformed, or low-confidence groups one at a time.

    A failed or uncertain result is returned as ``needs_review`` and always has
    ``usable=False``. Callers must not turn such a result into an asset.
    """

    effective_model = str(getattr(client, "model", "") or model).strip()
    groups = [_normalize_group(group) for group in visual_groups]
    _validate_unique_group_ids(groups)
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    role_labels = _known_role_labels(known_roles)
    manifest_entries = _index_manifest_entries(asset_manifest)

    results_by_id: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    cache_keys: dict[str, str] = {}

    for group in groups:
        group_id = group["group_id"]
        try:
            cache_key = build_group_cache_key(
                group["frame_paths"], model=effective_model, prompt_version=PROMPT_VERSION
            )
            cache_keys[group_id] = cache_key
        except (OSError, ValueError) as exc:
            results_by_id[group_id] = _failure_result(group, f"invalid_frames: {exc}")
            continue

        manifest_entry = manifest_entries.get(group_id)
        if manifest_entry is not None:
            result = normalize_curator_result(manifest_entry, group, role_labels=role_labels)
            result["source"] = "pre_director_asset_manifest"
            result["cache_key"] = cache_key
            _save_group_cache(
                cache_root, cache_key, result, manifest_entry, model=effective_model
            )
            results_by_id[group_id] = result
            continue

        if not force_refresh:
            cached = _load_group_cache(cache_root, cache_key, group)
            if cached is not None:
                results_by_id[group_id] = cached
                continue
        pending.append(group)

    if pending:
        model_client, owns_client, client_error = _resolve_client(
            client=client, api_key=api_key, base_url=base_url, model=model
        )
        if model_client is None:
            for group in pending:
                result = _failure_result(group, client_error or "visual_client_unavailable")
                result["cache_key"] = cache_keys[group["group_id"]]
                _save_group_cache(
                    cache_root,
                    cache_keys[group["group_id"]],
                    result,
                    {"error": client_error or "visual_client_unavailable"},
                    model=effective_model,
                )
                results_by_id[group["group_id"]] = result
        else:
            try:
                curated = _curate_pending_groups(
                    pending,
                    cache_root=cache_root,
                    cache_keys=cache_keys,
                    client=model_client,
                    model=effective_model,
                    story_context=story_context,
                    known_roles=known_roles,
                    role_labels=role_labels,
                    allow_single_group_fallback=allow_single_group_fallback,
                )
                results_by_id.update(curated)
            finally:
                if owns_client:
                    close = getattr(model_client, "close", None)
                    if callable(close):
                        close()

    return [results_by_id[group["group_id"]] for group in groups]


def build_group_cache_key(
    frame_paths: Sequence[str | Path],
    *,
    model: str = DEFAULT_CURATOR_MODEL,
    prompt_version: str = PROMPT_VERSION,
) -> str:
    """Build the stable cache key required by the curator contract."""

    if not frame_paths:
        raise ValueError("frame_paths must not be empty")
    frame_hashes = [_sha256_file(Path(path)) for path in frame_paths]
    payload = {
        "frame_hashes": frame_hashes,
        "model": str(model).strip(),
        "prompt_version": str(prompt_version).strip(),
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def normalize_curator_result(
    raw_result: Any,
    group: Mapping[str, Any],
    *,
    role_labels: set[str] | None = None,
) -> dict[str, Any]:
    """Normalize one model/manifest result into the stable public schema."""

    group_data = _normalize_group(group)
    parsed = _parse_json_value(raw_result)
    if not isinstance(parsed, Mapping):
        return _failure_result(group_data, "result_is_not_an_object")

    for wrapper in ("result", "asset", "data"):
        wrapped = parsed.get(wrapper)
        if isinstance(wrapped, Mapping):
            parsed = wrapped
            break

    raw_kind = str(parsed.get("kind") or "").strip().lower()
    kind = raw_kind if raw_kind in ALLOWED_KINDS else "ignore"
    confidence = _confidence(parsed.get("confidence"))
    frame_count = len(group_data["frame_paths"])
    representative_index = _strict_frame_index(
        parsed.get("representative_frame_index"), frame_count
    )
    visible_props = _normalize_visible_props(parsed.get("visible_props"), frame_count)
    matched_role = str(parsed.get("matched_role") or "").strip()
    if matched_role and role_labels and matched_role not in role_labels:
        matched_role = ""

    status = "ready"
    needs_review = False
    usable = kind in {"role", "scene", "prop"}
    validation_errors: list[str] = []

    if raw_kind not in ALLOWED_KINDS:
        validation_errors.append("invalid_kind")
    if kind == "mixed":
        status = "mixed"
        needs_review = True
        usable = False
    elif kind == "ignore":
        status = "ignored" if not validation_errors else "needs_review"
        needs_review = bool(validation_errors)
        usable = False
    elif representative_index is None:
        status = "needs_review"
        needs_review = True
        usable = False
        validation_errors.append("invalid_representative_frame_index")
    elif confidence < MIN_ASSET_CONFIDENCE:
        status = "needs_review"
        needs_review = True
        usable = False
        validation_errors.append("low_confidence")

    if kind == "prop" and confidence < MIN_PROP_CONFIDENCE:
        status = "needs_review"
        needs_review = True
        usable = False
        validation_errors.append("low_prop_confidence")

    representative_path = (
        group_data["frame_paths"][representative_index]
        if representative_index is not None
        else ""
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "group_id": group_data["group_id"],
        "status": status,
        "needs_review": needs_review,
        "usable": usable,
        "kind": kind,
        "name": str(parsed.get("name") or "").strip(),
        "identity": str(parsed.get("identity") or "").strip(),
        "matched_role": matched_role,
        "physical_scene": str(parsed.get("physical_scene") or "").strip(),
        "visible_props": visible_props,
        "representative_frame_index": representative_index,
        "representative_frame_path": representative_path,
        "confidence": confidence,
        "reason": str(parsed.get("reason") or "").strip(),
        "validation_errors": list(dict.fromkeys(validation_errors)),
        "source": "visual_model",
    }
    return result


def _curate_pending_groups(
    groups: Sequence[dict[str, Any]],
    *,
    cache_root: Path,
    cache_keys: Mapping[str, str],
    client: Any,
    model: str,
    story_context: str | Mapping[str, Any] | None,
    known_roles: Sequence[str | Mapping[str, Any]] | None,
    role_labels: set[str],
    allow_single_group_fallback: bool,
) -> dict[str, dict[str, Any]]:
    sheets: list[str] = []
    sheet_errors: dict[str, str] = {}
    callable_groups: list[dict[str, Any]] = []
    for group in groups:
        group_id = group["group_id"]
        sheet_path = cache_root / "contact_sheets" / f"{cache_keys[group_id]}.jpg"
        try:
            _build_contact_sheet(group["frame_paths"], sheet_path, group_id=group_id)
        except (OSError, ValueError) as exc:
            sheet_errors[group_id] = f"contact_sheet_failed: {exc}"
            continue
        sheets.append(str(sheet_path))
        callable_groups.append(group)

    batch_raw: Any = None
    batch_error = ""
    if callable_groups:
        try:
            batch_raw = client.analyze_segment_keyframes(
                sheets,
                retry=0,
                prompt_override=_build_batch_prompt(
                    callable_groups,
                    story_context=story_context,
                    known_roles=known_roles,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            batch_error = f"batch_call_failed: {exc}"

    batch_items = _batch_items_by_group(batch_raw, callable_groups)
    output: dict[str, dict[str, Any]] = {}
    for group in groups:
        group_id = group["group_id"]
        cache_key = cache_keys[group_id]
        raw_item = batch_items.get(group_id)
        if raw_item is not None:
            result = normalize_curator_result(raw_item, group, role_labels=role_labels)
            result["source"] = "visual_model_batch"
        else:
            reason = sheet_errors.get(group_id) or batch_error or "missing_batch_result"
            result = _failure_result(group, reason)

        if allow_single_group_fallback and _should_retry_single(result):
            fallback_raw, fallback_error = _call_single_group(
                client,
                group,
                story_context=story_context,
                known_roles=known_roles,
            )
            if fallback_raw is not None:
                fallback_result = normalize_curator_result(
                    fallback_raw, group, role_labels=role_labels
                )
                fallback_result["source"] = "visual_model_single_fallback"
                result = fallback_result
                raw_item = fallback_raw
            elif fallback_error:
                result = _failure_result(group, fallback_error)
                raw_item = {"error": fallback_error, "batch_response": batch_raw}

        result["cache_key"] = cache_key
        raw_to_save = raw_item if raw_item is not None else {
            "error": result.get("reason") or "curation_failed",
            "batch_response": batch_raw,
        }
        _save_group_cache(cache_root, cache_key, result, raw_to_save, model=model)
        output[group_id] = result
    return output


def _call_single_group(
    client: Any,
    group: Mapping[str, Any],
    *,
    story_context: str | Mapping[str, Any] | None,
    known_roles: Sequence[str | Mapping[str, Any]] | None,
) -> tuple[Any, str]:
    try:
        raw = client.analyze_segment_keyframes(
            list(group["frame_paths"]),
            retry=0,
            prompt_override=_build_single_prompt(
                group, story_context=story_context, known_roles=known_roles
            ),
        )
        if raw is None:
            return None, "single_group_fallback_failed"
        return raw, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"single_group_fallback_failed: {exc}"


def _should_retry_single(result: Mapping[str, Any]) -> bool:
    if str(result.get("status") or "") == "mixed":
        return False
    errors = set(result.get("validation_errors") or [])
    retryable = {
        "invalid_kind",
        "invalid_representative_frame_index",
        "low_confidence",
        "low_prop_confidence",
    }
    return str(result.get("status") or "") == "needs_review" and (
        bool(errors & retryable) or str(result.get("kind") or "") == "ignore"
    )


def _normalize_group(group: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(group, Mapping):
        raise ValueError("each visual group must be an object")
    group_id = str(group.get("group_id") or group.get("id") or "").strip()
    if not group_id:
        raise ValueError("visual group is missing group_id")
    raw_paths = group.get("frame_paths") or group.get("frames") or []
    frame_paths: list[str] = []
    for item in raw_paths:
        if isinstance(item, Mapping):
            value = item.get("path") or item.get("frame_path") or ""
        else:
            value = item
        path = str(value or "").strip()
        if path:
            frame_paths.append(str(Path(path)))
    if not frame_paths:
        raise ValueError(f"visual group {group_id} has no frame paths")
    return {"group_id": group_id, "frame_paths": frame_paths}


def _validate_unique_group_ids(groups: Sequence[Mapping[str, Any]]) -> None:
    ids = [str(group["group_id"]) for group in groups]
    if len(ids) != len(set(ids)):
        raise ValueError("visual group ids must be unique")


def _known_role_labels(
    known_roles: Sequence[str | Mapping[str, Any]] | None,
) -> set[str]:
    labels: set[str] = set()
    for role in known_roles or []:
        if isinstance(role, Mapping):
            for key in ("name", "role_name", "visual_label", "id"):
                value = str(role.get(key) or "").strip()
                if value:
                    labels.add(value)
        else:
            value = str(role or "").strip()
            if value:
                labels.add(value)
    return labels


def _index_manifest_entries(
    manifest: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> dict[str, Mapping[str, Any]]:
    if manifest is None:
        return {}
    if isinstance(manifest, Mapping) and isinstance(manifest.get("asset_manifest"), (Mapping, list)):
        return _index_manifest_entries(manifest["asset_manifest"])

    entries: list[Mapping[str, Any]] = []
    if isinstance(manifest, Mapping):
        for key in ("results", "assets", "groups"):
            value = manifest.get(key)
            if isinstance(value, list):
                entries = [item for item in value if isinstance(item, Mapping)]
                break
        if not entries:
            mapped_entries = []
            for key, value in manifest.items():
                if isinstance(value, Mapping):
                    item = dict(value)
                    item.setdefault("group_id", str(key))
                    mapped_entries.append(item)
            entries = mapped_entries
    elif isinstance(manifest, Sequence) and not isinstance(manifest, (str, bytes)):
        entries = [item for item in manifest if isinstance(item, Mapping)]

    indexed: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        group_id = str(
            entry.get("group_id")
            or entry.get("visual_group_id")
            or entry.get("curator_group_id")
            or ""
        ).strip()
        if group_id:
            indexed[group_id] = entry
    return indexed


def _batch_items_by_group(
    raw_response: Any, groups: Sequence[Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    parsed = _parse_json_value(raw_response)
    if not isinstance(parsed, Mapping):
        return {}
    values = parsed.get("results")
    if not isinstance(values, list):
        if len(groups) == 1:
            return {str(groups[0]["group_id"]): parsed}
        return {}
    output: dict[str, Mapping[str, Any]] = {}
    for item in values:
        if not isinstance(item, Mapping):
            continue
        group_id = str(item.get("group_id") or "").strip()
        if group_id:
            output[group_id] = item
    return output


def _parse_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        return None
    try:
        parsed, _ = json.JSONDecoder().raw_decode(text[min(starts):])
    except json.JSONDecodeError:
        return None
    return parsed


def _strict_frame_index(value: Any, frame_count: int) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 <= value < frame_count else None


def _normalize_visible_props(value: Any, frame_count: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    output: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "").strip()
        confidence = _confidence(item.get("confidence"))
        if not name or confidence < MIN_PROP_CONFIDENCE:
            continue
        indices = item.get("frame_indices") or []
        valid_indices = [
            index
            for index in indices
            if not isinstance(index, bool)
            and isinstance(index, int)
            and 0 <= index < frame_count
        ]
        if not valid_indices:
            continue
        output.append(
            {
                "name": name,
                "frame_indices": list(dict.fromkeys(valid_indices)),
                "confidence": confidence,
            }
        )
    return output


def _confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return round(max(0.0, min(1.0, number)), 6)


def _failure_result(group: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "group_id": str(group.get("group_id") or ""),
        "status": "needs_review",
        "needs_review": True,
        "usable": False,
        "kind": "ignore",
        "name": "",
        "identity": "",
        "matched_role": "",
        "physical_scene": "",
        "visible_props": [],
        "representative_frame_index": None,
        "representative_frame_path": "",
        "confidence": 0.0,
        "reason": str(reason),
        "validation_errors": ["curation_failed"],
        "source": "curation_failure",
    }


def _build_batch_prompt(
    groups: Sequence[Mapping[str, Any]],
    *,
    story_context: str | Mapping[str, Any] | None,
    known_roles: Sequence[str | Mapping[str, Any]] | None,
) -> str:
    group_lines = [
        f"Image {index + 1} is contact sheet for group {group['group_id']} "
        f"with frames F0..F{len(group['frame_paths']) - 1}."
        for index, group in enumerate(groups)
    ]
    return _prompt_header(story_context, known_roles) + "\n" + "\n".join(group_lines) + "\n" + (
        "Return one JSON object with a results array, exactly one item per group. "
        "Use zero-based representative_frame_index relative to F labels. "
        "Never merge different physical spaces into one scene. Do not infer a prop "
        "from story text unless it is visibly present. Use kind only from role, scene, "
        "prop, mixed, ignore. A mixed group must remain mixed. Schema for each item: "
        '{"group_id":"...","kind":"role|scene|prop|mixed|ignore",'
        '"name":"","identity":"","matched_role":"","physical_scene":"",'
        '"visible_props":[{"name":"","frame_indices":[0],"confidence":0.0}],'
        '"representative_frame_index":0,"confidence":0.0,"reason":""}.'
    )


def _build_single_prompt(
    group: Mapping[str, Any],
    *,
    story_context: str | Mapping[str, Any] | None,
    known_roles: Sequence[str | Mapping[str, Any]] | None,
) -> str:
    return _prompt_header(story_context, known_roles) + "\n" + (
        f"Review visual group {group['group_id']}. Uploaded images are F0 through "
        f"F{len(group['frame_paths']) - 1}. Return one JSON object using the same stable "
        "schema. representative_frame_index is a strict zero-based integer. Do not "
        "guess props from text. Return mixed when frames contain different physical "
        "spaces or different asset identities."
    )


def _prompt_header(
    story_context: str | Mapping[str, Any] | None,
    known_roles: Sequence[str | Mapping[str, Any]] | None,
) -> str:
    context = _compact_json(story_context, limit=3000)
    roles = _compact_json(known_roles or [], limit=1500)
    return (
        f"Mode2 visual asset curator. Schema={SCHEMA_VERSION}; prompt={PROMPT_VERSION}. "
        "Judge pixels first; story context is only a clue. A role is one visual person "
        "identity, a scene is one physical space, and a prop must be clearly visible. "
        "Do not invent names or assets. Preserve the source language for names. "
        f"Known roles: {roles}. Story context: {context}."
    )


def _compact_json(value: Any, *, limit: int) -> str:
    if value is None:
        return "none"
    if isinstance(value, str):
        return value[:limit]
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)[:limit]
    except TypeError:
        return str(value)[:limit]


def _build_contact_sheet(
    frame_paths: Sequence[str | Path], destination: Path, *, group_id: str
) -> None:
    images: list[Image.Image] = []
    for frame_path in frame_paths:
        try:
            with Image.open(frame_path) as source:
                images.append(source.convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"cannot read frame {frame_path}: {exc}") from exc
    if not images:
        raise ValueError("no readable frames")

    columns = min(4, len(images))
    rows = math.ceil(len(images) / columns)
    tile_width, tile_height, label_height = 256, 256, 24
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + label_height)), "black")
    draw = ImageDraw.Draw(sheet)
    for index, image in enumerate(images):
        tile = ImageOps.contain(image, (tile_width, tile_height))
        column, row = index % columns, index // columns
        x = column * tile_width + (tile_width - tile.width) // 2
        y = row * (tile_height + label_height) + label_height + (tile_height - tile.height) // 2
        sheet.paste(tile, (x, y))
        draw.text((column * tile_width + 6, row * (tile_height + label_height) + 5), f"F{index}", fill="white")
    draw.text((6, sheet.height - 16), str(group_id)[:40], fill="white")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    sheet.save(temporary, format="JPEG", quality=88)
    os.replace(temporary, destination)


def _resolve_client(
    *, client: Any | None, api_key: str | None, base_url: str | None, model: str
) -> tuple[Any | None, bool, str]:
    if client is not None:
        return client, False, ""
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        return None, False, "missing_api_key"
    try:
        from .gemini_analyzer import DEFAULT_BASE_URL, GeminiClient

        return (
            GeminiClient(
                api_key=key,
                base_url=base_url or DEFAULT_BASE_URL,
                model=model,
                request_timeout=240,
            ),
            True,
            "",
        )
    except Exception as exc:  # noqa: BLE001
        return None, False, f"client_initialization_failed: {exc}"


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"frame does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_artifact_dir(cache_root: Path, cache_key: str) -> Path:
    return cache_root / "groups" / cache_key


def _load_group_cache(
    cache_root: Path, cache_key: str, group: Mapping[str, Any]
) -> dict[str, Any] | None:
    path = _cache_artifact_dir(cache_root, cache_key) / "result.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("prompt_version") != PROMPT_VERSION:
        return None
    data = dict(data)
    data["group_id"] = str(group["group_id"])
    index = _strict_frame_index(data.get("representative_frame_index"), len(group["frame_paths"]))
    data["representative_frame_index"] = index
    data["representative_frame_path"] = group["frame_paths"][index] if index is not None else ""
    data["source"] = "cache"
    data["cache_hit"] = True
    data["cache_key"] = cache_key
    return data


def _save_group_cache(
    cache_root: Path,
    cache_key: str,
    result: Mapping[str, Any],
    raw_response: Any,
    *,
    model: str,
) -> None:
    artifact_dir = _cache_artifact_dir(cache_root, cache_key)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(artifact_dir / "result.json", dict(result))
    _atomic_write_json(artifact_dir / "raw_response.json", raw_response)
    _update_manifest_summary(cache_root, cache_key, result, model=model)


def _update_manifest_summary(
    cache_root: Path, cache_key: str, result: Mapping[str, Any], *, model: str
) -> None:
    manifest_path = cache_root / "manifest.json"
    with _MANIFEST_LOCK:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        if not isinstance(manifest, dict):
            manifest = {}
        entries = manifest.get("entries")
        if not isinstance(entries, dict):
            entries = {}
        previous = entries.get(cache_key) if isinstance(entries.get(cache_key), dict) else {}
        group_ids = list(previous.get("group_ids") or [])
        group_id = str(result.get("group_id") or "")
        if group_id and group_id not in group_ids:
            group_ids.append(group_id)
        entries[cache_key] = {
            "group_ids": group_ids,
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "status": result.get("status"),
            "kind": result.get("kind"),
            "usable": bool(result.get("usable")),
            "result_path": str(_cache_artifact_dir(cache_root, cache_key) / "result.json"),
            "raw_response_path": str(
                _cache_artifact_dir(cache_root, cache_key) / "raw_response.json"
            ),
        }
        manifest.update(
            {
                "schema_version": SCHEMA_VERSION,
                "prompt_version": PROMPT_VERSION,
                "entries": entries,
            }
        )
        _atomic_write_json(manifest_path, manifest)


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = [
    "ALLOWED_KINDS",
    "DEFAULT_CURATOR_MODEL",
    "MIN_ASSET_CONFIDENCE",
    "MIN_PROP_CONFIDENCE",
    "PROMPT_VERSION",
    "SCHEMA_VERSION",
    "build_group_cache_key",
    "curate_visual_groups",
    "normalize_curator_result",
]
