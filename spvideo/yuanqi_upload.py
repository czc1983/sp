"""Upload local media to Yuanqi and return a public URL."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests


DEFAULT_YUANQI_BASE_URL = "https://ai.szyqsc.cn"


class YuanqiUploadError(RuntimeError):
    pass


def resolve_yuanqi_api_key(fallback: str = "") -> str:
    return (
        os.environ.get("YUANQI_UPLOAD_API_KEY")
        or os.environ.get("YUANQI_API_KEY")
        or os.environ.get("WAN22_API_KEY")
        or fallback
        or ""
    ).strip()


def upload_file_for_url(
    file_path: str | Path,
    *,
    api_key: str = "",
    base_url: str = "",
    timeout: int = 600,
) -> dict[str, Any]:
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise YuanqiUploadError(f"upload_file_not_found: {path}")
    key = resolve_yuanqi_api_key(api_key)
    if not key:
        raise YuanqiUploadError("yuanqi_upload_api_key_missing")
    auth = key if key.lower().startswith("bearer ") else f"Bearer {key}"
    url = f"{(base_url or os.environ.get('YUANQI_UPLOAD_BASE_URL') or DEFAULT_YUANQI_BASE_URL).rstrip('/')}/api/upload"
    headers = {"Authorization": auth}
    with path.open("rb") as handle:
        response = requests.post(
            url,
            headers=headers,
            files={"file": (path.name, handle, _guess_content_type(path))},
            timeout=timeout,
        )
    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        detail = ""
        try:
            detail = str(response.text or "")[:800]
        except Exception:  # noqa: BLE001
            detail = ""
        raise YuanqiUploadError(f"yuanqi_upload_failed: {exc} {detail}".strip()) from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise YuanqiUploadError("yuanqi_upload_non_json_response") from exc
    public_url = _extract_url(data)
    if not public_url:
        raise YuanqiUploadError(f"yuanqi_upload_url_missing: {str(data)[:800]}")
    return {
        "url": public_url,
        "response": data,
        "size": path.stat().st_size,
        "filename": path.name,
    }


def _extract_url(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    candidates = [
        data.get("url"),
        data.get("video_url"),
        data.get("result_url"),
    ]
    nested = data.get("data")
    if isinstance(nested, dict):
        candidates.extend([
            nested.get("url"),
            nested.get("video_url"),
            nested.get("result_url"),
        ])
    for value in candidates:
        text = str(value or "").strip()
        if text.startswith(("http://", "https://")):
            return text
    return ""


def _guess_content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".mp4":
        return "video/mp4"
    if suffix == ".mov":
        return "video/quicktime"
    if suffix == ".mkv":
        return "video/x-matroska"
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix == ".wav":
        return "audio/wav"
    return "application/octet-stream"
