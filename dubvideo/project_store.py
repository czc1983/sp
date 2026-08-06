from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from .settings_store import ensure_runtime_dirs, load_settings


def _paths() -> dict[str, Path]:
    settings = load_settings()
    ensure_runtime_dirs(settings)
    raw = settings.get("foreign_dub", {}).get("paths", {})
    return {
        "project_root": Path(str(raw.get("project_root"))),
        "upload_root": Path(str(raw.get("upload_root"))),
        "export_root": Path(str(raw.get("export_root"))),
    }


def create_project(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    roots = _paths()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    project_id = f"dub_{stamp}_{uuid.uuid4().hex[:8]}"
    root = roots["project_root"] / project_id
    for name in ("input", "audio/tts_segments", "subtitles", "scripts", "render", "export", "logs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    defaults = load_settings().get("foreign_dub", {}).get("defaults", {})
    project = {
        "project_id": project_id,
        "name": str(payload.get("name") or project_id),
        "created_at": time.time(),
        "updated_at": time.time(),
        "source_language": payload.get("source_language") or defaults.get("source_language", "auto"),
        "target_language": payload.get("target_language") or defaults.get("target_language", "en"),
        "status": "created",
        "paths": {"project_dir": str(root)},
        "source_segments": [],
        "source_reviewed": False,
        "segments": [],
        "outputs": {},
    }
    save_project(project)
    return project


def project_path(project_id: str) -> Path:
    safe = "".join(ch for ch in str(project_id or "") if ch.isalnum() or ch in ("-", "_"))
    if not safe:
        raise ValueError("missing_project_id")
    return _paths()["project_root"] / safe


def load_project(project_id: str) -> dict[str, Any]:
    path = project_path(project_id) / "project.json"
    if not path.exists():
        raise FileNotFoundError("project_not_found")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("invalid_project")
    return data


def save_project(project: dict[str, Any]) -> dict[str, Any]:
    project["updated_at"] = time.time()
    root = Path(str(project.get("paths", {}).get("project_dir") or project_path(str(project.get("project_id")))))
    root.mkdir(parents=True, exist_ok=True)
    (root / "project.json").write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
    return project


def invalidate_downstream_outputs(project: dict[str, Any]) -> dict[str, Any]:
    project["segments"] = []
    outputs = dict(project.get("outputs") or {})
    for key in (
        "translated_srt",
        "tts_manifest",
        "translated_full_audio",
        "lipsync_driver_audio",
        "raw_lipsync_video",
        "lipsync_video",
        "lipsync_prompt_id",
    ):
        outputs.pop(key, None)
    project["outputs"] = outputs
    return project


def list_projects() -> list[dict[str, Any]]:
    root = _paths()["project_root"]
    projects: list[dict[str, Any]] = []
    for path in root.glob("dub_*"):
        meta = path / "project.json"
        if not meta.exists():
            continue
        try:
            data = json.loads(meta.read_text(encoding="utf-8-sig"))
            projects.append({
                "project_id": data.get("project_id"),
                "name": data.get("name"),
                "status": data.get("status"),
                "target_language": data.get("target_language"),
                "has_source_video": bool(data.get("paths", {}).get("source_video")),
                "source_segment_count": len(data.get("source_segments") or []),
                "translated_segment_count": len(data.get("segments") or []),
                "updated_at": data.get("updated_at"),
                "project_dir": str(path),
            })
        except Exception:
            continue
    return sorted(projects, key=lambda item: float(item.get("updated_at") or 0), reverse=True)


def attach_source_video(project_id: str, source_path: str | Path) -> dict[str, Any]:
    project = load_project(project_id)
    root = Path(str(project["paths"]["project_dir"]))
    src = Path(source_path)
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"source_video_not_found: {src}")
    target = root / "input" / ("original" + src.suffix.lower())
    shutil.copy2(src, target)
    project.setdefault("paths", {})["source_video"] = str(target)
    project["source_segments"] = []
    project["source_reviewed"] = False
    project = invalidate_downstream_outputs(project)
    outputs = dict(project.get("outputs") or {})
    for key in (
        "source_audio",
        "source_srt",
        "ocr_manifest",
    ):
        outputs.pop(key, None)
    project["outputs"] = outputs
    project["status"] = "video_ready"
    return save_project(project)
