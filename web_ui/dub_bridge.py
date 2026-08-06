"""外语对口型（翻译）工作台桥接层。

把 E:/fy/foreign_lipsync 的独立服务合并进 SP 主服务：
- 页面：GET /foreign_lipsync（同源 iframe 嵌入 SP 工作台「翻译」页）
- 接口：/api/dub/...（ASR / OCR / 翻译 / TTS / 音色复刻 / 对口型 / 混音）
- 媒体：复用主服务已有的 GET /media?path=

业务实现都在顶层包 dubvideo 里，这里只做路由分发。
"""

from __future__ import annotations

import cgi  # noqa: W4901  (Python 3.12 仍可用，与 fy 原服务一致)
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote

SP_ROOT = Path(__file__).resolve().parent.parent
if str(SP_ROOT) not in sys.path:
    sys.path.insert(0, str(SP_ROOT))

from dubvideo.asr_client import AsrSettings, DashScopeFileTransAsrClient, OpenAICompatibleAsrClient  # noqa: E402
from dubvideo.comfy_provider import ComfyLipSyncProvider  # noqa: E402
from dubvideo.ffmpeg_tools import ffmpeg_status  # noqa: E402
from dubvideo.jobs import cancel_job, create_job, get_job, run_background  # noqa: E402
from dubvideo.ocr_client import OcrSettings, QwenOcrClient  # noqa: E402
from dubvideo.pipeline import (  # noqa: E402
    run_asr_job,
    run_final_mix_job,
    run_lipsync_job,
    run_ocr_job,
    run_translate_job,
    run_tts_job,
    run_voice_clone_job,
)
from dubvideo.project_store import (  # noqa: E402
    attach_source_video,
    create_project,
    invalidate_downstream_outputs,
    list_projects,
    load_project,
    save_project,
)
from dubvideo.settings_store import (  # noqa: E402
    APP_ROOT,
    ensure_runtime_dirs,
    import_legacy_minimax_config,
    import_sp_translate_config,
    load_settings,
    public_settings,
    save_settings,
    status_summary,
)
from dubvideo.subtitle_tools import parse_srt, save_srt, segments_to_srt  # noqa: E402
from dubvideo.translate_client import OpenAICompatibleTranslator, TranslateSettings  # noqa: E402
from dubvideo.tts_client_minimax import MiniMaxTtsClient, MiniMaxTtsSettings  # noqa: E402
from dubvideo.voice_clone_client import MiniMaxVoiceCloneClient, VoiceCloneSettings, generate_voice_id  # noqa: E402
from dubvideo.voice_library import collect_voice_library  # noqa: E402

DASHBOARD = Path(__file__).resolve().parent / "foreign_lipsync_dashboard.html"
SERVER_STARTED_AT = time.time()

ensure_runtime_dirs()

DUB_PAGE_PATHS = {"/foreign_lipsync", "/foreign_lipsync_dashboard.html"}


def handle_dub_get(handler: Any, parsed: Any) -> bool:
    """处理 GET 请求；命中 dub 路由返回 True，否则返回 False。"""
    path = parsed.path
    if path in DUB_PAGE_PATHS:
        handler._send_file(DASHBOARD)
        return True
    if path == "/api/dub/server-status":
        settings = load_settings()
        handler._send_json({
            "pid": os.getpid(),
            "started_at": SERVER_STARTED_AT,
            "app_root": str(APP_ROOT),
            "settings_path": str(APP_ROOT / ".dub_config" / "settings.json"),
            "provider_status": status_summary(settings),
            "ffmpeg": ffmpeg_status(str(settings["foreign_dub"]["paths"].get("ffmpeg") or "")),
        })
        return True
    if path == "/api/dub/settings":
        handler._send_json({
            "settings": public_settings(),
            "provider_status": status_summary(),
        })
        return True
    if path == "/api/dub/voice-library":
        handler._send_json({"voices": collect_voice_library()})
        return True
    if path == "/api/dub/projects":
        handler._send_json({"projects": list_projects()})
        return True
    if path.startswith("/api/dub/projects/"):
        project_id = path.rsplit("/", 1)[-1]
        handler._send_json(load_project(project_id))
        return True
    if path.startswith("/api/dub/jobs/"):
        job_id = path.rsplit("/", 1)[-1]
        job = get_job(job_id)
        if not job:
            handler._send_json({"error": "job_not_found"}, status=404)
        else:
            handler._send_json(job)
        return True
    return False


def handle_dub_post(handler: Any, parsed: Any) -> bool:
    """处理 POST 请求；命中 dub 路由返回 True，否则返回 False。"""
    path = parsed.path
    if not path.startswith("/api/dub/"):
        return False
    if path == "/api/dub/settings":
        payload = handler._read_json()
        settings = save_settings(payload.get("settings") if "settings" in payload else payload)
        handler._send_json({"settings": public_settings(settings), "provider_status": status_summary(settings)})
        return True
    if path == "/api/dub/settings/test-provider":
        handler._send_json(test_provider(handler._read_json()))
        return True
    if path == "/api/dub/settings/import-minimax-legacy":
        payload = handler._read_json()
        result = import_legacy_minimax_config(payload.get("source_dir"))
        settings = result.pop("settings")
        handler._send_json({
            **result,
            "settings": public_settings(settings),
            "provider_status": status_summary(settings),
        })
        return True
    if path == "/api/dub/settings/import-sp-translate":
        payload = handler._read_json()
        result = import_sp_translate_config(payload.get("sp_root"))
        settings = result.pop("settings")
        handler._send_json({
            **result,
            "settings": public_settings(settings),
            "provider_status": status_summary(settings),
        })
        return True
    if path == "/api/dub/pick-path":
        payload = handler._read_json()
        handler._send_json({"path": pick_path(str(payload.get("kind") or "directory"))})
        return True
    if path == "/api/dub/open-path":
        payload = handler._read_json()
        target = Path(str(payload.get("path") or ""))
        if not target.exists():
            raise FileNotFoundError(f"path_not_found: {target}")
        os.startfile(str(target))  # type: ignore[attr-defined]
        handler._send_json({"ok": True})
        return True
    if path == "/api/dub/create-project":
        handler._send_json(create_project(handler._read_json()))
        return True
    if path == "/api/dub/save-project":
        handler._send_json(save_project(handler._read_json()))
        return True
    if path == "/api/dub/import-srt":
        handler._send_json(import_srt(handler._read_json()))
        return True
    if path == "/api/dub/export-srt":
        handler._send_json(export_srt(handler._read_json()))
        return True
    if path == "/api/dub/upload-video":
        handler._send_json(_handle_upload_video(handler))
        return True
    if path == "/api/dub/asr":
        payload = handler._read_json()
        job = create_job("dub_asr", payload)
        run_background(job, run_asr_job)
        handler._send_json({"job_id": job["id"], "job": job})
        return True
    if path == "/api/dub/ocr":
        payload = handler._read_json()
        job = create_job("dub_ocr", payload)
        run_background(job, run_ocr_job)
        handler._send_json({"job_id": job["id"], "job": job})
        return True
    if path == "/api/dub/tts":
        payload = handler._read_json()
        job = create_job("dub_tts", payload)
        run_background(job, run_tts_job)
        handler._send_json({"job_id": job["id"], "job": job})
        return True
    if path == "/api/dub/clone-voices":
        payload = handler._read_json()
        job = create_job("dub_voice_clone", payload)
        run_background(job, run_voice_clone_job)
        handler._send_json({"job_id": job["id"], "job": job})
        return True
    if path == "/api/dub/clone-voice-file":
        handler._send_json(_handle_clone_voice_file(handler))
        return True
    if path == "/api/dub/translate":
        payload = handler._read_json()
        job = create_job("dub_translate", payload)
        run_background(job, run_translate_job)
        handler._send_json({"job_id": job["id"], "job": job})
        return True
    if path == "/api/dub/lipsync":
        payload = handler._read_json()
        job = create_job("dub_lipsync", payload)
        run_background(job, run_lipsync_job)
        handler._send_json({"job_id": job["id"], "job": job})
        return True
    if path == "/api/dub/final-mix":
        payload = handler._read_json()
        job = create_job("dub_final_mix", payload)
        run_background(job, run_final_mix_job)
        handler._send_json({"job_id": job["id"], "job": job})
        return True
    if path.startswith("/api/dub/jobs/") and path.endswith("/cancel"):
        job_id = path.rsplit("/", 2)[-2]
        handler._send_json({"cancelled": cancel_job(job_id)})
        return True
    handler._send_json({"error": "not_found"}, status=404)
    return True


def _read_multipart(handler: Any) -> cgi.FieldStorage:
    return cgi.FieldStorage(
        fp=handler.rfile,
        headers=handler.headers,
        environ={
            "REQUEST_METHOD": "POST",
            "CONTENT_TYPE": handler.headers.get("Content-Type", ""),
        },
    )


def _handle_upload_video(handler: Any) -> dict[str, Any]:
    form = _read_multipart(handler)
    project_id = str(form.getfirst("project_id") or "").strip()
    if not project_id:
        project = create_project({})
        project_id = str(project["project_id"])
    file_item = form["file"] if "file" in form else None
    if file_item is None or not getattr(file_item, "filename", ""):
        raise ValueError("missing_upload_file")
    settings = load_settings()
    upload_root = Path(str(settings["foreign_dub"]["paths"]["upload_root"]))
    upload_root.mkdir(parents=True, exist_ok=True)
    suffix = Path(str(file_item.filename)).suffix.lower() or ".mp4"
    upload_path = upload_root / f"{project_id}_{int(time.time())}{suffix}"
    with upload_path.open("wb") as out:
        shutil.copyfileobj(file_item.file, out)
    project = attach_source_video(project_id, upload_path)
    return {"project": project, "uploaded_path": str(upload_path)}


def _handle_clone_voice_file(handler: Any) -> dict[str, Any]:
    form = _read_multipart(handler)
    project_id = str(form.getfirst("project_id") or "").strip()
    speaker = str(form.getfirst("speaker") or "").strip()
    if not project_id or not speaker:
        raise ValueError("missing_project_or_speaker")
    file_item = form["file"] if "file" in form else None
    if file_item is None or not getattr(file_item, "filename", ""):
        raise ValueError("missing_upload_file")
    suffix = Path(str(file_item.filename)).suffix.lower()
    if suffix not in (".mp3", ".wav", ".m4a"):
        raise ValueError("仅支持 mp3 / wav / m4a 音频文件")
    project = load_project(project_id)
    root = Path(str(project["paths"]["project_dir"]))
    sample_dir = root / "audio" / "voice_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", speaker) or "speaker"
    sample_path = sample_dir / f"upload_{safe_name}{suffix}"
    with sample_path.open("wb") as out:
        shutil.copyfileobj(file_item.file, out)

    settings = load_settings()
    clone_settings = VoiceCloneSettings.from_tts_settings(settings["foreign_dub"]["tts_minimax"])
    if not clone_settings.api_key:
        raise ValueError("MiniMax API Key 未配置，无法复刻音色。")
    client = MiniMaxVoiceCloneClient()
    file_id = client.upload_file(sample_path, clone_settings)
    voice_id = generate_voice_id()
    client.clone_voice(file_id, voice_id, clone_settings)

    speaker_voices = {
        str(k): str(v)
        for k, v in (project.get("speaker_voices") or {}).items()
        if str(v or "").strip()
    }
    speaker_voices[speaker] = voice_id
    project["speaker_voices"] = speaker_voices

    manifest_path = root / "audio" / "voice_clones.json"
    previous_cloned: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            for item in previous.get("cloned") or []:
                if isinstance(item, dict) and item.get("speaker"):
                    previous_cloned[str(item["speaker"])] = item
        except Exception:  # noqa: BLE001
            previous_cloned = {}
    previous_cloned[speaker] = {
        "speaker": speaker,
        "voice_id": voice_id,
        "sample_path": str(sample_path),
        "sample_seconds": 0,
    }
    manifest_path.write_text(
        json.dumps({"speaker_voices": speaker_voices, "cloned": list(previous_cloned.values()), "skipped": [], "errors": []}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_project(project)
    return {"project": project, "voice_id": voice_id}


def test_provider(payload: dict[str, Any]) -> dict[str, Any]:
    provider = str(payload.get("provider") or "").strip()
    settings = load_settings()
    root = settings["foreign_dub"]
    if provider == "comfy":
        comfy = root["lipsync_comfy"]
        client = ComfyLipSyncProvider(str(payload.get("base_url") or comfy.get("base_url") or ""))
        result = client.test_connection()
        if result.get("ok"):
            inventory = client.fetch_inventory(APP_ROOT / ".dub_config" / "server_inventory", refresh=False)
            result["profiles"] = inventory.get("profiles")
            result["summary"] = inventory.get("summary")
        return result
    if provider == "minimax":
        data = dict(root["tts_minimax"])
        if isinstance(payload.get("settings"), dict):
            data.update({key: value for key, value in payload["settings"].items() if not (key == "api_key" and value == "__KEEP__")})
        return MiniMaxTtsClient().test_connection(MiniMaxTtsSettings.from_dict(data))
    if provider == "asr":
        data = dict(root["asr"])
        if isinstance(payload.get("settings"), dict):
            data.update({key: value for key, value in payload["settings"].items() if not (key == "api_key" and value == "__KEEP__")})
        asr_settings = AsrSettings.from_dict(data)
        if asr_settings.provider.strip().lower() == "dashscope_filetrans" or "filetrans" in asr_settings.model:
            return DashScopeFileTransAsrClient().test_connection(asr_settings)
        return OpenAICompatibleAsrClient().test_connection(asr_settings)
    if provider == "ocr":
        data = dict(root.get("ocr") or {})
        if isinstance(payload.get("settings"), dict):
            data.update({key: value for key, value in payload["settings"].items() if not (key == "api_key" and value == "__KEEP__")})
        return QwenOcrClient().test_connection(OcrSettings.from_dict(data), _ocr_test_image())
    if provider == "translate":
        data = dict(root["translate"])
        if isinstance(payload.get("settings"), dict):
            data.update({key: value for key, value in payload["settings"].items() if not (key == "api_key" and value == "__KEEP__")})
        return OpenAICompatibleTranslator().test_connection(TranslateSettings.from_dict(data))
    if provider == "ffmpeg":
        return ffmpeg_status(str(root["paths"].get("ffmpeg") or ""))
    return {"ok": False, "error": f"unknown_provider: {provider}"}


def _ocr_test_image() -> Path:
    path = APP_ROOT / ".dub_config" / "ocr_test.png"
    if path.exists():
        return path
    try:
        from PIL import Image, ImageDraw, ImageFont

        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (900, 220), "black")
        draw = ImageDraw.Draw(image)
        font = None
        for candidate in (
            Path(r"C:\Windows\Fonts\msyh.ttc"),
            Path(r"C:\Windows\Fonts\simhei.ttf"),
            Path(r"C:\Windows\Fonts\simsun.ttc"),
        ):
            if candidate.exists():
                font = ImageFont.truetype(str(candidate), 58)
                break
        draw.text((120, 75), "画面字幕测试", fill="white", font=font)
        image.save(path)
    except Exception:
        path.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
            b"\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
        )
    return path


def import_srt(payload: dict[str, Any]) -> dict[str, Any]:
    project_id = str(payload.get("project_id") or "").strip()
    project = load_project(project_id) if project_id else create_project(payload)
    content = str(payload.get("content") or "")
    source_path = str(payload.get("path") or "").strip()
    if not content and source_path:
        content = Path(source_path).read_text(encoding="utf-8-sig")
    if not content.strip():
        raise ValueError("missing_srt_content")
    language = str(payload.get("language") or project.get("target_language") or "en")
    segments = parse_srt(content, language=language)
    if not segments:
        raise ValueError("srt_has_no_segments")
    root = Path(str(project["paths"]["project_dir"]))
    kind = str(payload.get("kind") or "translated").strip() or "translated"
    srt_name = "source.srt" if kind == "source" else "translated.srt"
    srt_path = root / "subtitles" / srt_name
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path.write_text(content, encoding="utf-8")
    segment_dicts = [segment.to_dict() for segment in segments]
    if kind == "source":
        project = invalidate_downstream_outputs(project)
        project["source_language"] = language
        project["source_segments"] = segment_dicts
        project["source_reviewed"] = False
        project["source_revision_reason"] = "import_srt"
        project["status"] = "source_script_ready"
    else:
        project["target_language"] = language
        project["segments"] = segment_dicts
        project["status"] = "script_ready"
    project.setdefault("outputs", {})[f"{kind}_srt"] = str(srt_path)
    project = save_project(project)
    return {
        "project": project,
        "segments": segment_dicts,
        "path": str(srt_path),
        "count": len(segments),
    }


def export_srt(payload: dict[str, Any]) -> dict[str, Any]:
    project_id = str(payload.get("project_id") or "").strip()
    if not project_id:
        raise ValueError("missing_project_id")
    project = load_project(project_id)
    kind = str(payload.get("kind") or "translated").strip() or "translated"
    segments = project.get("source_segments" if kind == "source" else "segments") or []
    if not segments:
        raise ValueError("missing_source_segments" if kind == "source" else "missing_segments")
    root = Path(str(project["paths"]["project_dir"]))
    srt_name = "source.srt" if kind == "source" else "translated.srt"
    srt_path = save_srt(root / "subtitles" / srt_name, segments)
    project.setdefault("outputs", {})[f"{kind}_srt"] = str(srt_path)
    project = save_project(project)
    return {
        "project": project,
        "path": str(srt_path),
        "content": segments_to_srt(segments),
        "count": len(segments),
    }


def pick_path(kind: str) -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if kind == "file":
            return filedialog.askopenfilename() or ""
        return filedialog.askdirectory() or ""
    finally:
        root.destroy()
