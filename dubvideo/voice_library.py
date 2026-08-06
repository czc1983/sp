from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .settings_store import load_settings


def collect_voice_library() -> list[dict[str, Any]]:
    """扫描所有项目，汇总复刻过的音色（voice_id），供跨项目复用。

    每个音色返回：voice_id、出现过的说话人标签、来源项目、可试听的采样音频路径、最近更新时间。
    """
    settings = load_settings()
    project_root = Path(str(settings["foreign_dub"]["paths"]["project_root"]))
    voices: dict[str, dict[str, Any]] = {}
    if not project_root.exists():
        return []
    for meta in sorted(project_root.glob("dub_*/project.json")):
        try:
            project = json.loads(meta.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if not isinstance(project, dict):
            continue
        project_id = str(project.get("project_id") or meta.parent.name)
        project_dir = Path(str(project.get("paths", {}).get("project_dir") or meta.parent))
        samples = _speaker_samples(project_dir)
        speaker_voices = project.get("speaker_voices") or {}
        if not isinstance(speaker_voices, dict):
            continue
        for speaker, raw_voice_id in speaker_voices.items():
            voice_id = str(raw_voice_id or "").strip()
            if not voice_id:
                continue
            entry = voices.setdefault(voice_id, {
                "voice_id": voice_id,
                "labels": [],
                "projects": [],
                "sample_path": "",
                "updated_at": 0.0,
            })
            label = str(speaker or "").strip()
            if label and label not in entry["labels"]:
                entry["labels"].append(label)
            if project_id not in entry["projects"]:
                entry["projects"].append(project_id)
            updated = float(project.get("updated_at") or meta.stat().st_mtime)
            sample = samples.get(label, "")
            if updated >= float(entry["updated_at"] or 0):
                entry["updated_at"] = updated
                if sample and Path(sample).is_file():
                    entry["sample_path"] = sample
    return sorted(voices.values(), key=lambda item: float(item.get("updated_at") or 0), reverse=True)


def _speaker_samples(project_dir: Path) -> dict[str, str]:
    manifest_path = project_dir / "audio" / "voice_clones.json"
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    samples: dict[str, str] = {}
    for item in manifest.get("cloned") or []:
        if not isinstance(item, dict):
            continue
        speaker = str(item.get("speaker") or "").strip()
        sample_path = str(item.get("sample_path") or "").strip()
        if speaker and sample_path:
            samples[speaker] = sample_path
    return samples
