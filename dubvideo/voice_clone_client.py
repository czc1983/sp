from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class VoiceCloneSettings:
    api_base: str = "https://api.minimaxi.com"
    api_key: str = ""
    need_noise_reduction: bool = True
    need_volume_normalization: bool = True

    @classmethod
    def from_tts_settings(cls, data: dict[str, Any]) -> "VoiceCloneSettings":
        api_url = str(data.get("api_url") or "https://api.minimaxi.com/v1/t2a_v2").rstrip("/")
        api_base = re.sub(r"/v1/t2a_v2$", "", api_url) or "https://api.minimaxi.com"
        return cls(
            api_base=api_base,
            api_key=str(data.get("api_key") or ""),
            need_noise_reduction=bool(data.get("clone_need_noise_reduction", True)),
            need_volume_normalization=bool(data.get("clone_need_volume_normalization", True)),
        )


def generate_voice_id(prefix: str = "dubspk") -> str:
    """MiniMax 自定义 voice_id：首字符必须是英文字母，[8,256] 位，仅字母数字 - _，不能以 - _ 结尾。"""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class MiniMaxVoiceCloneClient:
    def upload_file(self, audio_path: str | Path, settings: VoiceCloneSettings, *, purpose: str = "voice_clone") -> int:
        path = Path(audio_path)
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"clone_audio_missing: {path}")
        if not settings.api_key:
            raise ValueError("minimax_api_key_missing")
        with path.open("rb") as file:
            response = requests.post(
                f"{settings.api_base}/v1/files/upload",
                headers={"Authorization": f"Bearer {settings.api_key}"},
                data={"purpose": purpose},
                files={"file": (path.name, file, _audio_mime(path))},
                timeout=300,
            )
        response.raise_for_status()
        data = response.json()
        _raise_for_base_resp(data, "upload")
        file_id = (data.get("file") or {}).get("file_id")
        if file_id in (None, ""):
            raise RuntimeError(f"MiniMax 文件上传未返回 file_id: {json.dumps(data, ensure_ascii=False)[:500]}")
        return int(file_id)

    def clone_voice(
        self,
        file_id: int,
        voice_id: str,
        settings: VoiceCloneSettings,
    ) -> dict[str, Any]:
        if not settings.api_key:
            raise ValueError("minimax_api_key_missing")
        payload = {
            "file_id": int(file_id),
            "voice_id": voice_id,
            "need_noise_reduction": bool(settings.need_noise_reduction),
            "need_volume_normalization": bool(settings.need_volume_normalization),
            "aigc_watermark": False,
        }
        response = requests.post(
            f"{settings.api_base}/v1/voice_clone",
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=600,
        )
        response.raise_for_status()
        data = response.json()
        _raise_for_base_resp(data, "voice_clone")
        return data


def _raise_for_base_resp(data: dict[str, Any], stage: str) -> None:
    base_resp = data.get("base_resp") or {}
    status_code = base_resp.get("status_code", 0)
    if status_code not in (0, "0", None):
        raise RuntimeError(f"MiniMax {stage} 失败: {base_resp.get('status_msg') or base_resp}")


def _audio_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
    }.get(suffix, "application/octet-stream")
