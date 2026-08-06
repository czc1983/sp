from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = APP_ROOT / ".dub_config"
SETTINGS_PATH = CONFIG_ROOT / "settings.json"


DEFAULT_SETTINGS: dict[str, Any] = {
    "foreign_dub": {
        "providers": {
            "asr": "dashscope_filetrans",
            "ocr": "openai_compatible_vision",
            "translate": "manual",
            "tts": "minimax",
            "lipsync": "comfy",
        },
        "asr": {
            "provider": "dashscope_filetrans",
            "base_url": "",
            "api_key": "",
            "model": "qwen3-asr-flash-filetrans",
            "max_tokens": 8192,
            "timeout_seconds": 1800,
            "poll_interval_seconds": 2,
            "language": "auto",
            "enable_itn": False,
            "enable_words": True,
            "diarization_enabled": True,
            "speaker_count": 0,
        },
        "ocr": {
            "provider": "openai_compatible_vision",
            "base_url": "",
            "api_key": "",
            "model": "qwen-vl-ocr-latest",
            "crop_bottom_ratio": 0.5,
            "frames_per_segment": 6,
            "request_interval_ms": 300,
            "timeout_seconds": 120,
        },
        "translate": {
            "provider": "openai_compatible",
            "base_url": "",
            "api_key": "",
            "model": "qwen-plus",
            "temperature": 0.2,
            "max_tokens": 4096,
            "batch_size": 20,
            "request_interval_ms": 500,
        },
        "tts_minimax": {
            "api_url": "https://api.minimaxi.com/v1/t2a_v2",
            "api_key": "",
            "group_id": "",
            "model": "speech-2.8-turbo",
            "voice_id": "",
            "speed": 1.0,
            "volume": 1.0,
            "pitch": 0,
            "audio_format": "mp3",
            "sample_rate": 32000,
            "bitrate": 128000,
            "channel": 1,
            "language_boost": "auto",
            "request_interval_ms": 500,
            "max_retries": 3,
            "rate_limit_wait_seconds": 65,
            "speaker_voice_map": {},
            "voice_options": [],
            "emotion": "",
            "auto_emotion": True,
        },
        "lipsync_comfy": {
            "base_url": "https://8188-cpod-1sqfx2anig0i.pod.compshare.cn",
            "profile": "kling_audio",
            "workflow": "",
            "model": "",
            "timeout_seconds": 5400,
        },
        "lipsync_cloud": {
            "provider": "",
            "base_url": "",
            "api_key": "",
            "model": "videoretalk",
            "poll_interval_seconds": 5,
            "timeout_seconds": 3600,
        },
        "paths": {
            "ffmpeg": "ffmpeg",
            "ffprobe": "ffprobe",
            "project_root": str(APP_ROOT / ".dub_projects"),
            "upload_root": str(APP_ROOT / ".dub_uploads"),
            "job_root": str(APP_ROOT / ".dub_jobs"),
            "export_root": str(APP_ROOT / "exports"),
            "sp_root": "E:/sp",
        },
        "defaults": {
            "source_language": "auto",
            "target_language": "en",
            "keep_background_audio": True,
            "generate_subtitles": True,
            "auto_fit_translation_duration": True,
        },
        "sp_integration": {
            "enable_send_to_sp": False,
            "send_mode": "foreign_original",
        },
    }
}

SECRET_KEYS = {"api_key"}


def _deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def ensure_runtime_dirs(settings: dict[str, Any] | None = None) -> None:
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    data = settings or load_settings()
    paths = data.get("foreign_dub", {}).get("paths", {})
    for key in ("project_root", "upload_root", "job_root", "export_root"):
        raw = paths.get(key)
        if raw:
            Path(str(raw)).mkdir(parents=True, exist_ok=True)


def load_settings() -> dict[str, Any]:
    data: dict[str, Any] = {}
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8-sig"))
        except Exception:
            data = {}
    settings = _deep_merge(DEFAULT_SETTINGS, data if isinstance(data, dict) else {})
    _upgrade_legacy_settings(settings)
    return settings


def _upgrade_legacy_settings(settings: dict[str, Any]) -> None:
    root = settings.setdefault("foreign_dub", {})
    asr = root.setdefault("asr", {})
    if str(asr.get("provider") or "").strip().lower() == "manual" and not str(asr.get("model") or "").strip():
        asr["provider"] = "dashscope_filetrans"
        asr["model"] = "qwen3-asr-flash-filetrans"
    if not str(asr.get("model") or "").strip():
        asr["model"] = "qwen3-asr-flash-filetrans"
    if str(asr.get("model") or "").strip() == "qwen3.5-omni-flash":
        asr["provider"] = "dashscope_filetrans"
        asr["model"] = "qwen3-asr-flash-filetrans"
    asr.setdefault("max_tokens", 8192)
    asr.setdefault("timeout_seconds", 1800)
    asr.setdefault("poll_interval_seconds", 2)
    asr.setdefault("language", "auto")
    asr.setdefault("enable_itn", False)
    asr["enable_words"] = bool(asr.get("enable_words", True))
    asr["diarization_enabled"] = bool(asr.get("diarization_enabled", True))
    asr["speaker_count"] = int(asr.get("speaker_count") or 0)
    ocr = root.setdefault("ocr", {})
    ocr.setdefault("provider", "openai_compatible_vision")
    ocr.setdefault("model", "qwen-vl-ocr-latest")
    ocr.setdefault("crop_bottom_ratio", 0.5)
    ocr.setdefault("frames_per_segment", 6)
    ocr.setdefault("request_interval_ms", 300)
    ocr.setdefault("timeout_seconds", 120)
    translate = root.setdefault("translate", {})
    if str(translate.get("provider") or "").strip().lower() == "manual" and not str(translate.get("model") or "").strip():
        translate["provider"] = "openai_compatible"
        translate["model"] = "qwen-plus"
    if not str(translate.get("model") or "").strip():
        translate["model"] = "qwen-plus"
    if (
        str(translate.get("model") or "").strip().lower() == "gpt-5.5"
        and "aliyuncs.com" in str(translate.get("base_url") or "").lower()
    ):
        translate["model"] = "qwen-plus"
    if str(asr.get("provider") or "").strip().lower() != "manual":
        if not str(asr.get("base_url") or "").strip() and str(translate.get("base_url") or "").strip():
            asr["base_url"] = translate.get("base_url")
        if not str(asr.get("api_key") or "").strip() and str(translate.get("api_key") or "").strip():
            asr["api_key"] = translate.get("api_key")
    if not str(ocr.get("base_url") or "").strip() and str(translate.get("base_url") or "").strip():
        ocr["base_url"] = translate.get("base_url")
    if not str(ocr.get("api_key") or "").strip() and str(translate.get("api_key") or "").strip():
        ocr["api_key"] = translate.get("api_key")
    providers = root.setdefault("providers", {})
    if providers.get("asr") in {"manual", "openai_compatible_audio"} and asr.get("provider") == "dashscope_filetrans":
        providers["asr"] = "dashscope_filetrans"
    if not providers.get("ocr") and ocr.get("provider"):
        providers["ocr"] = "openai_compatible_vision"
    if providers.get("translate") == "manual" and translate.get("provider") == "openai_compatible":
        providers["translate"] = "openai_compatible"


def save_settings(incoming: dict[str, Any]) -> dict[str, Any]:
    current = load_settings()
    clean = _preserve_secrets(current, _strip_public_secret_meta(incoming))
    merged = _deep_merge(current, clean)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    ensure_runtime_dirs(merged)
    return merged


def _preserve_secrets(current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    cleaned = copy.deepcopy(incoming)

    def walk(now: Any, old: Any) -> Any:
        if isinstance(now, dict):
            for key, value in list(now.items()):
                if key in SECRET_KEYS and (value is None or str(value) in {"", "__KEEP__"}):
                    if isinstance(old, dict):
                        now[key] = old.get(key, "")
                else:
                    now[key] = walk(value, old.get(key) if isinstance(old, dict) else {})
        return now

    return walk(cleaned, current)


def _strip_public_secret_meta(incoming: dict[str, Any]) -> dict[str, Any]:
    cleaned = copy.deepcopy(incoming)

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            for key in list(node.keys()):
                if key.endswith("_set") or key.endswith("_preview"):
                    node.pop(key, None)
                else:
                    walk(node[key])
        elif isinstance(node, list):
            for item in node:
                walk(item)
        return node

    return walk(cleaned)


def public_settings(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    data = copy.deepcopy(settings or load_settings())

    def mask(node: Any, path: tuple[str, ...] = ()) -> Any:
        if isinstance(node, dict):
            for key, value in list(node.items()):
                if key in SECRET_KEYS:
                    raw = str(value or "")
                    reveal = path in {("foreign_dub", "asr"), ("foreign_dub", "ocr"), ("foreign_dub", "translate")}
                    node[key] = raw if reveal else ""
                    node[f"{key}_set"] = bool(raw)
                    node[f"{key}_preview"] = _preview_secret(raw)
                else:
                    mask(value, path + (str(key),))
        elif isinstance(node, list):
            for item in node:
                mask(item, path)
        return node

    return mask(data)


def _preview_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 10:
        return value[:2] + "***"
    return value[:6] + "..." + value[-4:]


def status_summary(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    data = settings or load_settings()
    root = data.get("foreign_dub", {})
    tts = root.get("tts_minimax", {})
    comfy = root.get("lipsync_comfy", {})
    asr = root.get("asr", {})
    translate = root.get("translate", {})
    asr_provider = str(asr.get("provider") or "").strip().lower()
    translate_provider = str(translate.get("provider") or "").strip().lower()
    return {
        "asr": (
            "manual" if asr_provider == "manual"
            else "configured" if asr.get("base_url") and asr.get("api_key") and asr.get("model")
            else "missing"
        ),
        "translate": (
            "manual" if translate_provider == "manual"
            else "configured" if translate.get("base_url") and translate.get("api_key") and translate.get("model")
            else "missing"
        ),
        "ocr": "configured" if root.get("ocr", {}).get("base_url") and root.get("ocr", {}).get("api_key") and root.get("ocr", {}).get("model") else "missing",
        "tts": "configured" if tts.get("api_key") else "missing",
        "lipsync": "configured" if comfy.get("base_url") else "missing",
    }


def import_legacy_minimax_config(source_dir: str | Path | None = None) -> dict[str, Any]:
    base = Path(source_dir) if source_dir else Path(r"C:\Users\Administrator\Desktop\20")
    config_path = base / "minimax_tts_config.json"
    voices_path = base / "minimax_voices.json"
    if not config_path.exists():
        raise FileNotFoundError(f"legacy_minimax_config_not_found: {config_path}")
    raw_config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw_config, dict):
        raise ValueError("legacy_minimax_config_invalid")

    allowed = {
        "api_url",
        "api_key",
        "model",
        "voice_id",
        "speed",
        "volume",
        "pitch",
        "audio_format",
        "sample_rate",
        "bitrate",
        "channel",
        "request_interval_ms",
        "max_retries",
        "rate_limit_wait_seconds",
    }
    tts_update = {key: raw_config[key] for key in allowed if key in raw_config}
    tts_update["language_boost"] = "auto"

    voice_options: list[dict[str, Any]] = []
    if voices_path.exists():
        try:
            raw_voices = json.loads(voices_path.read_text(encoding="utf-8-sig"))
            values = raw_voices.get("voices", []) if isinstance(raw_voices, dict) else raw_voices
            if isinstance(values, list):
                for item in values:
                    if not isinstance(item, dict):
                        continue
                    voice_id = str(item.get("voice_id") or "").strip()
                    if not voice_id:
                        continue
                    voice_options.append({
                        "voice_id": voice_id,
                        "name": str(item.get("name") or voice_id),
                        "language": str(item.get("language") or ""),
                        "emotion": str(item.get("emotion") or ""),
                        "custom": bool(item.get("custom")),
                    })
        except Exception:
            voice_options = []
    if voice_options:
        tts_update["voice_options"] = voice_options

    settings = save_settings({"foreign_dub": {"tts_minimax": tts_update}})
    return {
        "settings": settings,
        "source_dir": str(base),
        "config_path": str(config_path),
        "voice_count": len(voice_options),
        "imported_fields": sorted(key for key in tts_update if key != "api_key"),
        "api_key_imported": bool(str(raw_config.get("api_key") or "")),
    }


def import_sp_translate_config(sp_root: str | Path | None = None) -> dict[str, Any]:
    base = Path(sp_root) if sp_root else Path(r"E:\sp")
    dashboard_path = base / "web_ui" / "story_generate_dashboard.html"
    if not dashboard_path.exists():
        raise FileNotFoundError(f"sp_dashboard_not_found: {dashboard_path}")
    html = dashboard_path.read_text(encoding="utf-8-sig")
    base_url = _js_const(html, "DEFAULT_PRE_DIRECTOR_BASE_URL")
    api_key = _js_const(html, "DEFAULT_PRE_DIRECTOR_API_KEY")
    raw_model = _js_const(html, "DEFAULT_ASSISTANT_MODEL") or "qwen-plus"
    model = "qwen-plus" if raw_model.strip().lower() == "gpt-5.5" and "aliyuncs.com" in base_url.lower() else raw_model
    if not base_url or not api_key:
        raise ValueError("sp_translate_config_missing_base_url_or_key")
    settings = save_settings({
        "foreign_dub": {
            "providers": {
                "asr": "dashscope_filetrans",
                "ocr": "openai_compatible_vision",
                "translate": "openai_compatible",
            },
            "asr": {
                "provider": "dashscope_filetrans",
                "base_url": base_url,
                "api_key": api_key,
                "model": "qwen3-asr-flash-filetrans",
                "max_tokens": 8192,
                "timeout_seconds": 1800,
                "poll_interval_seconds": 2,
                "language": "auto",
                "enable_itn": False,
                "enable_words": True,
            },
            "ocr": {
                "provider": "openai_compatible_vision",
                "base_url": base_url,
                "api_key": api_key,
                "model": "qwen-vl-ocr-latest",
                "crop_bottom_ratio": 0.5,
                "frames_per_segment": 6,
            },
            "translate": {
                "provider": "openai_compatible",
                "base_url": base_url,
                "api_key": api_key,
                "model": model,
            },
        }
    })
    return {
        "settings": settings,
        "sp_root": str(base),
        "source": str(dashboard_path),
        "base_url": base_url,
        "model": model,
        "api_key_imported": bool(api_key),
    }


def _js_const(source: str, name: str) -> str:
    pattern = re.compile(rf"const\s+{re.escape(name)}\s*=\s*(['\"])(.*?)\1\s*;", re.DOTALL)
    match = pattern.search(source)
    return str(match.group(2)).strip() if match else ""
