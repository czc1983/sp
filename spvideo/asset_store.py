from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from .models import ensure_dir

_GLOBAL_ROOT = Path(__file__).resolve().parent.parent / ".global_assets"
DIRECTOR_ROLE_COLORS = ("蓝色", "红色", "绿色", "紫色", "青色", "黄色")


def global_character_dirs() -> dict[str, Path]:
    root = ensure_dir(_GLOBAL_ROOT)
    return {"root": root, "characters": ensure_dir(root / "characters")}

_GLOBAL_DIRS = global_character_dirs()


def asset_dirs(project_dir: str | Path) -> dict[str, Path]:
    root = ensure_dir(Path(project_dir) / "assets")
    return {
        "root": root,
        "objects": ensure_dir(root / "objects"),
        "originals": ensure_dir(root / "originals"),
        "tracks": ensure_dir(root / "tracks"),
        "face_samples": ensure_dir(root / "face_samples"),
        "backgrounds": ensure_dir(root / "backgrounds"),
    }


def load_asset_store(project_dir: str | Path) -> dict[str, Any]:
    dirs = asset_dirs(project_dir)
    annotations = _read_json(dirs["root"] / "annotations.json", [])
    directors = _read_json(dirs["root"] / "directors.json", {})
    auto_director = _read_json(dirs["root"] / "auto_director.json", {})
    storyboard_assets = _read_json(dirs["root"] / "storyboard_assets.json", {})
    return {
        "characters": _read_json(_GLOBAL_DIRS["root"] / "characters.json", {}),
        "objects": _read_json(dirs["root"] / "objects.json", {}),
        "originals": _read_json(dirs["root"] / "originals.json", []),
        "annotations": annotations,
        "backgrounds": _read_json(dirs["root"] / "backgrounds.json", {}),
        "directors": _hydrate_director_plans(directors, annotations),
        "auto_director": auto_director,
        "storyboard_assets": storyboard_assets,
    }


def upsert_director_plan(
    project_dir: str | Path,
    *,
    segment_id: str,
    video_path: str,
    roles: list[dict[str, Any]],
    positive_prompt: str = "",
    sam_text: str = "",
) -> dict[str, Any]:
    video_value = str(video_path or "").strip()
    segment_value = str(segment_id or "").strip()
    if not video_value:
        raise ValueError("missing_video_path")

    cleaned_roles: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for raw_role in roles:
        name = str(raw_role.get("name") or "").strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        cleaned_role = {
            "name": name,
            "annotation_id": str(raw_role.get("annotation_id") or "").strip(),
        }
        if raw_role.get("clear_annotation"):
            cleaned_role["clear_annotation"] = True
        cleaned_roles.append(cleaned_role)
    if not cleaned_roles:
        raise ValueError("director_roles_required")
    if len(cleaned_roles) > len(DIRECTOR_ROLE_COLORS):
        raise ValueError(f"director_roles_max_{len(DIRECTOR_ROLE_COLORS)}")

    dirs = asset_dirs(project_dir)
    plans = _read_json(dirs["root"] / "directors.json", {})
    key = _director_key(segment_value, video_value)
    current = plans.get(key, {})
    previous_roles = current.get("roles", [])
    previous_by_name = {
        str(item.get("name") or ""): item
        for item in previous_roles
        if str(item.get("name") or "")
    }
    for order, role in enumerate(cleaned_roles, 1):
        previous = previous_by_name.get(role["name"], {})
        if role.get("clear_annotation"):
            role["annotation_id"] = ""
        elif not role["annotation_id"]:
            role["annotation_id"] = str(previous.get("annotation_id") or "")
        role.pop("clear_annotation", None)
        role["order"] = order
        role["color"] = DIRECTOR_ROLE_COLORS[order - 1]

    item = {
        **current,
        "id": current.get("id") or f"director_{uuid.uuid4().hex[:10]}",
        "segment_id": segment_value,
        "video_path": video_value,
        "mode": "replacement",
        "roles": cleaned_roles,
        "positive_prompt": str(positive_prompt or "").strip()[:4000],
        "sam_text": str(sam_text or "").strip()[:500] or (
            "people" if len(cleaned_roles) > 1 else "person"
        ),
        "updated_at": time.time(),
    }
    item.setdefault("created_at", time.time())
    plans[key] = item
    _write_json(dirs["root"] / "directors.json", plans)
    return get_director_plan(
        project_dir,
        segment_id=segment_value,
        video_path=video_value,
    ) or item


def get_director_plan(
    project_dir: str | Path,
    *,
    segment_id: str = "",
    video_path: str = "",
) -> dict[str, Any] | None:
    dirs = asset_dirs(project_dir)
    plans = _read_json(dirs["root"] / "directors.json", {})
    key = _director_key(str(segment_id or "").strip(), str(video_path or "").strip())
    item = plans.get(key)
    if item is None and video_path:
        item = next(
            (
                plan
                for plan in plans.values()
                if _same_asset_path(plan.get("video_path"), video_path)
            ),
            None,
        )
    if item is None:
        return None
    annotations = _read_json(dirs["root"] / "annotations.json", [])
    return _hydrate_director_plan(item, annotations)


def assign_director_role_annotation(
    project_dir: str | Path,
    *,
    segment_id: str,
    video_path: str,
    role_name: str,
    annotation_id: str,
) -> dict[str, Any]:
    dirs = asset_dirs(project_dir)
    plans = _read_json(dirs["root"] / "directors.json", {})
    key = _director_key(str(segment_id or "").strip(), str(video_path or "").strip())
    item = plans.get(key)
    if item is None:
        raise ValueError("director_plan_not_found")

    matched = False
    for role in item.get("roles", []):
        if str(role.get("name") or "").strip() == str(role_name or "").strip():
            role["annotation_id"] = str(annotation_id or "").strip()
            matched = True
            break
    if not matched:
        raise ValueError("director_role_not_found")

    item["updated_at"] = time.time()
    plans[key] = item
    _write_json(dirs["root"] / "directors.json", plans)
    return get_director_plan(
        project_dir,
        segment_id=segment_id,
        video_path=video_path,
    ) or item


def _hydrate_director_plans(
    plans: dict[str, Any],
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        key: _hydrate_director_plan(item, annotations)
        for key, item in plans.items()
    }


def _hydrate_director_plan(
    item: dict[str, Any],
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {**item}
    video_path = str(result.get("video_path") or "")
    segment_id = str(result.get("segment_id") or "")
    characters = _read_json(_GLOBAL_DIRS["root"] / "characters.json", {})
    character_by_name = {
        str(character.get("name") or "").strip(): character
        for character in characters.values()
        if str(character.get("name") or "").strip()
    }
    annotations_by_id = {
        str(annotation.get("id") or ""): annotation
        for annotation in annotations
        if annotation.get("id")
    }
    hydrated_roles: list[dict[str, Any]] = []
    missing_refs: list[str] = []
    missing_marks: list[str] = []

    for index, raw_role in enumerate(result.get("roles", []), 1):
        role = {**raw_role}
        name = str(role.get("name") or "").strip()
        role["order"] = index
        role["color"] = DIRECTOR_ROLE_COLORS[index - 1]
        character = character_by_name.get(name, {})
        role["target_ref"] = str(character.get("ref_image") or "")
        role["target_extra_refs"] = [
            path
            for path in _clean_path_list(character.get("extra_ref_images") or [])
            if Path(path).exists()
        ]
        if not role["target_ref"] or not Path(role["target_ref"]).exists():
            missing_refs.append(name)

        annotation_id = str(role.get("annotation_id") or "").strip()
        annotation = annotations_by_id.get(annotation_id) if annotation_id else None
        if annotation is None and annotation_id:
            candidates = [
                candidate
                for candidate in annotations
                if candidate.get("type") == "person"
                and str(candidate.get("label_name") or "").strip() == name
                and _same_asset_path(candidate.get("video_path"), video_path)
                and (
                    not segment_id
                    or not candidate.get("segment_id")
                    or str(candidate.get("segment_id")) == segment_id
                )
            ]
            annotation = min(candidates, key=lambda value: float(value.get("time") or 0), default=None)
        if annotation is not None:
            role["annotation_id"] = str(annotation.get("id") or "")
            role["point"] = annotation.get("point")
            role["mark_time"] = float(annotation.get("time") or 0)
            role["track_status"] = str(annotation.get("track_status") or "selected")
            role["track_dir"] = str(annotation.get("track_dir") or "")
        else:
            role["point"] = None
            role["mark_time"] = None
            role["track_status"] = "missing"
            if len(result.get("roles", [])) > 1:
                missing_marks.append(name)
        hydrated_roles.append(role)

    issues: list[str] = []
    if not hydrated_roles:
        issues.append("未选择角色")
    if missing_refs:
        issues.append("缺少目标图: " + "、".join(missing_refs))
    if missing_marks:
        issues.append("缺少身份点: " + "、".join(missing_marks))
    result["roles"] = hydrated_roles
    result["issues"] = issues
    result["status"] = "ready" if not issues else "incomplete"
    return result


def _director_key(segment_id: str, video_path: str) -> str:
    if segment_id:
        return segment_id
    return "video_" + uuid.uuid5(uuid.NAMESPACE_URL, str(video_path).lower()).hex[:16]


def _same_asset_path(left: Any, right: Any) -> bool:
    left_value = str(left or "").strip()
    right_value = str(right or "").strip()
    if not left_value or not right_value:
        return False
    try:
        return Path(left_value).resolve() == Path(right_value).resolve()
    except OSError:
        return left_value.replace("/", "\\").lower() == right_value.replace("/", "\\").lower()


def upsert_background_config(
    project_dir: str | Path,
    *,
    segment_id: str,
    video_path: str,
    mode: str = "keep_original",
    asset_path: str = "",
    fit_mode: str = "cover",
    feather_pixels: int = 9,
    dilate_pixels: int = 6,
) -> dict[str, Any]:
    if mode not in {"keep_original", "replace_static_image", "replace_background_video"}:
        raise ValueError("invalid_background_mode")
    if fit_mode not in {"cover", "contain", "stretch"}:
        raise ValueError("invalid_background_fit_mode")
    if mode != "keep_original":
        asset = Path(asset_path)
        if not asset.exists() or not asset.is_file():
            raise ValueError("background_asset_not_found")

    dirs = asset_dirs(project_dir)
    configs = _read_json(dirs["root"] / "backgrounds.json", {})
    key = segment_id.strip() or _background_video_key(video_path)
    current = configs.get(key, {})
    item = {
        **current,
        "segment_id": segment_id.strip(),
        "video_path": video_path,
        "mode": mode,
        "asset_path": asset_path if mode != "keep_original" else "",
        "fit_mode": fit_mode,
        "feather_pixels": max(0, min(int(feather_pixels), 99)),
        "dilate_pixels": max(0, min(int(dilate_pixels), 40)),
        "status": "ready" if mode != "keep_original" else "disabled",
        "output_path": "",
        "error": "",
        "updated_at": time.time(),
    }
    configs[key] = item
    _write_json(dirs["root"] / "backgrounds.json", configs)
    return item


def get_background_config(
    project_dir: str | Path,
    *,
    segment_id: str = "",
    video_path: str = "",
) -> dict[str, Any] | None:
    dirs = asset_dirs(project_dir)
    configs = _read_json(dirs["root"] / "backgrounds.json", {})
    if segment_id and segment_id in configs:
        return configs[segment_id]
    if video_path:
        direct = configs.get(_background_video_key(video_path))
        if direct:
            return direct
        for item in configs.values():
            try:
                if Path(str(item.get("video_path") or "")).resolve() == Path(video_path).resolve():
                    return item
            except OSError:
                continue
    return None


def update_background_config(
    project_dir: str | Path,
    *,
    segment_id: str = "",
    video_path: str = "",
    **fields: Any,
) -> dict[str, Any] | None:
    dirs = asset_dirs(project_dir)
    configs = _read_json(dirs["root"] / "backgrounds.json", {})
    key = segment_id.strip() or _background_video_key(video_path)
    if key not in configs and video_path:
        key = next(
            (
                candidate
                for candidate, item in configs.items()
                if str(item.get("video_path") or "").lower() == video_path.lower()
            ),
            key,
        )
    if key not in configs:
        return None
    configs[key].update(fields)
    configs[key]["updated_at"] = time.time()
    _write_json(dirs["root"] / "backgrounds.json", configs)
    return configs[key]


def _background_video_key(video_path: str) -> str:
    return "video_" + uuid.uuid5(uuid.NAMESPACE_URL, str(video_path).lower()).hex[:16]


def upsert_character(
    project_dir: str | Path,
    *,
    name: str,
    label: int | None = None,
    ref_image: str = "",
    extra_ref_images: list[str] | None = None,
    character_id: str | None = None,
) -> dict[str, Any]:
    gdirs = _GLOBAL_DIRS
    data = _read_json(gdirs["root"] / "characters.json", {})
    cid = character_id or _find_by_name(data, name) or f"char_{uuid.uuid4().hex[:8]}"

    stored_ref = ref_image
    if ref_image:
        stored_ref = _copy_ref_image(ref_image, gdirs["characters"], cid)

    current = data.get(cid, {})
    stored_extra_refs = current.get("extra_ref_images", [])
    if extra_ref_images is not None:
        stored_extra_refs = [
            _copy_extra_ref_image(path, gdirs["characters"], cid, index)
            for index, path in enumerate(_clean_path_list(extra_ref_images)[:6], 1)
        ]
    data[cid] = {
        **current,
        "id": cid,
        "name": name,
        "label": label,
        "ref_image": stored_ref or current.get("ref_image", ""),
        "extra_ref_images": stored_extra_refs,
        "segments": current.get("segments", []),
        "face_samples": current.get("face_samples", []),
        "tracks": current.get("tracks", []),
        "updated_at": time.time(),
    }
    _write_json(gdirs["root"] / "characters.json", data)
    return data[cid]


def add_annotation(
    project_dir: str | Path,
    *,
    video_path: str,
    time_seconds: float,
    label_id: int,
    label_name: str,
    kind: str,
    point: list[float] | None = None,
    box: list[float] | None = None,
    path: list[list[float]] | None = None,
    segment_id: str = "",
) -> dict[str, Any]:
    dirs = asset_dirs(project_dir)
    annotations = _read_json(dirs["root"] / "annotations.json", [])
    item = {
        "id": f"ann_{uuid.uuid4().hex[:10]}",
        "video_path": video_path,
        "segment_id": segment_id,
        "time": round(float(time_seconds), 3),
        "label_id": int(label_id),
        "label_name": label_name,
        "type": kind,
        "point": point,
        "box": box,
        "path": path,
        "created_at": time.time(),
    }
    annotations.append(item)
    _write_json(dirs["root"] / "annotations.json", annotations)

    if kind == "person":
        upsert_character(project_dir, name=label_name, label=label_id)
    return item


def update_annotation_mask(
    project_dir: str | Path,
    annotation_id: str,
    *,
    mask_path: str,
    mask_type: str,
    mask_box: list[float] | None = None,
) -> dict[str, Any] | None:
    dirs = asset_dirs(project_dir)
    annotations = _read_json(dirs["root"] / "annotations.json", [])
    updated = None
    for item in annotations:
        if item.get("id") == annotation_id:
            item["mask_path"] = mask_path
            item["mask_type"] = mask_type
            item["mask_box"] = mask_box
            item["mask_updated_at"] = time.time()
            updated = item
            break
    if updated is not None:
        _write_json(dirs["root"] / "annotations.json", annotations)
    return updated


def update_annotation(
    project_dir: str | Path,
    annotation_id: str,
    **fields: Any,
) -> dict[str, Any] | None:
    dirs = asset_dirs(project_dir)
    annotations = _read_json(dirs["root"] / "annotations.json", [])
    updated = None
    for item in annotations:
        if item.get("id") == annotation_id:
            item.update(fields)
            item["updated_at"] = time.time()
            updated = item
            break
    if updated is not None:
        _write_json(dirs["root"] / "annotations.json", annotations)
    return updated


def add_original_asset(
    project_dir: str | Path,
    *,
    annotation_id: str,
    label_id: int,
    label_name: str,
    kind: str,
    asset_name: str = "",
    role: str = "source_to_keep_or_replace",
    video_path: str,
    time_seconds: float,
    crop_path: str,
    context_path: str = "",
    cutout_path: str = "",
    mask_path: str = "",
    box: list[float] | None = None,
) -> dict[str, Any]:
    dirs = asset_dirs(project_dir)
    originals = _read_json(dirs["root"] / "originals.json", [])
    existing = next((item for item in originals if item.get("annotation_id") == annotation_id), None)
    item = {
        **(existing or {}),
        "id": (existing or {}).get("id") or f"orig_{uuid.uuid4().hex[:10]}",
        "annotation_id": annotation_id,
        "asset_name": asset_name or label_name,
        "label_id": int(label_id),
        "label_name": label_name,
        "type": kind,
        "video_path": video_path,
        "time": round(float(time_seconds), 3),
        "crop_path": crop_path,
        "context_path": context_path,
        "cutout_path": cutout_path or crop_path,
        "mask_path": mask_path,
        "box": box,
        "role": role or "source_to_keep_or_replace",
        "updated_at": time.time(),
    }
    if existing:
        originals = [item if old.get("annotation_id") == annotation_id else old for old in originals]
    else:
        item["created_at"] = time.time()
        originals.append(item)
    _write_json(dirs["root"] / "originals.json", originals)
    return item


def delete_original_asset(project_dir: str | Path, asset_id: str) -> bool:
    dirs = asset_dirs(project_dir)
    originals = _read_json(dirs["root"] / "originals.json", [])
    target = next(
        (
            item
            for item in originals
            if item.get("id") == asset_id
            or item.get("annotation_id") == asset_id
            or item.get("crop_path") == asset_id
        ),
        None,
    )
    if not target:
        return False

    originals = [item for item in originals if item is not target]
    _write_json(dirs["root"] / "originals.json", originals)

    for key in ("crop_path", "cutout_path", "context_path", "mask_path"):
        path = target.get(key)
        if not path:
            continue
        try:
            file_path = Path(path)
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
        except OSError:
            pass

    annotation_id = str(target.get("annotation_id") or "")
    if annotation_id:
        annotations = _read_json(dirs["root"] / "annotations.json", [])
        annotations = [item for item in annotations if item.get("id") != annotation_id]
        _write_json(dirs["root"] / "annotations.json", annotations)
    return True


def _copy_ref_image(ref_image: str, target_dir: Path, item_id: str) -> str:
    src = Path(ref_image)
    if not src.exists() or not src.is_file():
        return ref_image
    suffix = src.suffix.lower() or ".png"
    dst = target_dir / f"{item_id}{suffix}"
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return str(dst)


def _copy_extra_ref_image(ref_image: str, target_dir: Path, item_id: str, index: int) -> str:
    src = Path(ref_image)
    if not src.exists() or not src.is_file():
        return ref_image
    suffix = src.suffix.lower() or ".png"
    dst = target_dir / f"{item_id}_extra_{index:02d}{suffix}"
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return str(dst)


def _clean_path_list(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        path = str(value or "").strip()
        if not path:
            continue
        key = path.replace("/", "\\").lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(path)
    return cleaned


def _find_by_name(data: dict[str, Any], name: str) -> str | None:
    for key, value in data.items():
        if value.get("name") == name:
            return key
    return None


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def delete_character(project_dir: str | Path, character_id: str) -> bool:
    data = _read_json(_GLOBAL_DIRS["root"] / "characters.json", {})
    if character_id not in data:
        return False
    deleted_name = data.get(character_id, {}).get("name", "")
    del data[character_id]
    _write_json(_GLOBAL_DIRS["root"] / "characters.json", data)

    dirs = asset_dirs(project_dir)
    originals = _read_json(dirs["root"] / "originals.json", [])
    for original in [item for item in originals if item.get("label_name") == deleted_name]:
        delete_original_asset(project_dir, str(original.get("id") or ""))

    annotations = _read_json(dirs["root"] / "annotations.json", [])
    annotations = [item for item in annotations if item.get("label_name") != deleted_name]
    _write_json(dirs["root"] / "annotations.json", annotations)
    return True
