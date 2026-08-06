from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .models import TtsSegment


@dataclass
class MiniMaxTtsSettings:
    api_url: str
    api_key: str
    model: str = "speech-2.8-turbo"
    voice_id: str = ""
    speed: float = 1.0
    volume: float = 1.0
    pitch: int = 0
    audio_format: str = "mp3"
    sample_rate: int = 32000
    bitrate: int = 128000
    channel: int = 1
    language_boost: str = "auto"
    request_interval_ms: int = 500
    max_retries: int = 3
    rate_limit_wait_seconds: int = 65
    speaker_voice_map: dict[str, str] | None = None
    emotion: str = ""
    auto_emotion: bool = True
    emotion_prosody: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MiniMaxTtsSettings":
        return cls(
            api_url=str(data.get("api_url") or "https://api.minimaxi.com/v1/t2a_v2"),
            api_key=str(data.get("api_key") or ""),
            model=str(data.get("model") or "speech-2.8-turbo"),
            voice_id=str(data.get("voice_id") or ""),
            speed=float(data.get("speed") or 1.0),
            volume=float(data.get("volume") or 1.0),
            pitch=int(data.get("pitch") or 0),
            audio_format=str(data.get("audio_format") or "mp3"),
            sample_rate=int(data.get("sample_rate") or 32000),
            bitrate=int(data.get("bitrate") or 128000),
            channel=int(data.get("channel") or 1),
            language_boost=str(data.get("language_boost") or "auto"),
            request_interval_ms=int(data.get("request_interval_ms") or 500),
            max_retries=int(data.get("max_retries") or 3),
            rate_limit_wait_seconds=int(data.get("rate_limit_wait_seconds") or 65),
            speaker_voice_map=dict(data.get("speaker_voice_map") or {}),
            emotion=str(data.get("emotion") or ""),
            auto_emotion=bool(data.get("auto_emotion", True)),
            emotion_prosody=bool(data.get("emotion_prosody", True)),
        )


LANGUAGE_BOOST = {
    "zh": "Chinese",
    "cn": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "jp": "Japanese",
    "ko": "Korean",
    "kr": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
}

# 情绪 → 韵律微调（语速倍率、音量倍率、音高偏移）。
# MiniMax 的 emotion 参数只是轻度倾向，强情绪需要韵律参数配合才能顶上去。
# 改动映射后 bump PROSODY_VERSION，让旧缓存自动失效重合成。
PROSODY_VERSION = "v1_20260806"
EMOTION_PROSODY: dict[str, tuple[float, float, int]] = {
    "happy":     (1.05, 1.10, 1),
    "sad":       (0.92, 0.92, -1),
    "angry":     (1.08, 1.20, 1),
    "fearful":   (1.10, 1.05, 2),
    "disgusted": (0.95, 1.00, -1),
    "surprise":  (1.10, 1.12, 2),
    "neutral":   (1.00, 1.00, 0),
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _apply_emotion_prosody(settings: MiniMaxTtsSettings, emotion: str) -> tuple[float, float, int]:
    """按情绪映射微调语速/音量/音高，返回实际生效的三参数。"""
    speed, volume, pitch = settings.speed, settings.volume, settings.pitch
    if emotion and settings.emotion_prosody and emotion in EMOTION_PROSODY:
        mult_speed, mult_vol, pitch_shift = EMOTION_PROSODY[emotion]
        speed = _clamp(speed * mult_speed, 0.5, 2.0)
        volume = _clamp(volume * mult_vol, 0.1, 10.0)
        pitch = int(_clamp(pitch + pitch_shift, -12, 12))
    return speed, volume, pitch


class MiniMaxTtsClient:
    def test_connection(self, settings: MiniMaxTtsSettings) -> dict[str, Any]:
        return {
            "ok": bool(settings.api_url and settings.api_key),
            "provider": "minimax",
            "configured": bool(settings.api_key),
            "model": settings.model,
            "message": "MiniMax 已配置" if settings.api_key else "MiniMax API Key 未配置",
        }

    def synthesize_text(
        self,
        text: str,
        settings: MiniMaxTtsSettings,
        *,
        language: str = "en",
        voice_id: str = "",
        emotion: str = "",
    ) -> bytes:
        text = text.strip()
        if not text:
            raise ValueError("tts_text_empty")
        if not settings.api_key:
            raise ValueError("minimax_api_key_missing")
        voice = voice_id or settings.voice_id
        if not voice:
            raise ValueError("minimax_voice_id_missing")
        language_boost = settings.language_boost
        if language_boost.lower() == "auto":
            language_boost = LANGUAGE_BOOST.get(language.lower(), "")

        speed, volume, pitch = _apply_emotion_prosody(settings, emotion)

        payload: dict[str, Any] = {
            "model": settings.model,
            "text": text,
            "stream": False,
            "voice_setting": {
                "voice_id": voice,
                "speed": speed,
                "vol": volume,
                "pitch": pitch,
            },
            "audio_setting": {
                "sample_rate": settings.sample_rate,
                "bitrate": settings.bitrate,
                "format": settings.audio_format,
                "channel": settings.channel,
            },
            "output_format": "hex",
        }
        if language_boost:
            payload["language_boost"] = language_boost
        if emotion:
            payload["voice_setting"]["emotion"] = emotion

        request = urllib.request.Request(
            settings.api_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"MiniMax TTS HTTP {error.code}: {body}") from error

        base_resp = data.get("base_resp") or {}
        status_code = base_resp.get("status_code", 0)
        if status_code not in (0, "0", None):
            hint = ""
            if str(status_code) == "1008":
                hint = "（MiniMax 账户余额不足，请充值后重新生成配音；已生成的分段会自动复用，不重复扣费）"
            raise RuntimeError(f"MiniMax TTS failed: {base_resp}{hint}")
        audio_hex = (data.get("data") or {}).get("audio")
        if not audio_hex:
            raise RuntimeError(f"MiniMax TTS returned no audio: {data}")
        return bytes.fromhex(audio_hex)

    def synthesize_segments(
        self,
        segments: list[TtsSegment],
        output_dir: Path,
        settings: MiniMaxTtsSettings,
        progress: Callable[[str, int], None] | None = None,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        manifest = _load_manifest(manifest_path)
        manifest.update({
            "version": "1.0",
            "provider": "minimax",
            "model": settings.model,
            "audio_format": settings.audio_format,
        })
        existing = {
            str(item.get("segment_id") or item.get("id")): item
            for item in manifest.get("segments", [])
            if isinstance(item, dict)
        }
        total = len(segments)
        last_request_started = 0.0
        for index, segment in enumerate(segments, start=1):
            sid = segment.segment_id
            voice_id = (settings.speaker_voice_map or {}).get(segment.speaker) or settings.voice_id
            filename = f"{sid}.{settings.audio_format}"
            audio_path = output_dir / filename
            signature = _cache_signature(settings, voice_id, segment.language, segment.emotion)
            cached = existing.get(sid)
            if _can_reuse(cached, segment.text, audio_path, signature):
                if progress:
                    progress(f"复用已有 TTS: {sid} ({index}/{total})", int(index / max(total, 1) * 100))
            else:
                if settings.request_interval_ms > 0 and last_request_started > 0:
                    wait_ms = settings.request_interval_ms - int((time.monotonic() - last_request_started) * 1000)
                    if wait_ms > 0:
                        time.sleep(wait_ms / 1000)
                if progress:
                    label = f"生成 TTS: {sid} ({index}/{total})"
                    if segment.emotion and settings.emotion_prosody and segment.emotion in EMOTION_PROSODY and segment.emotion != "neutral":
                        p_speed, p_vol, p_pitch = _apply_emotion_prosody(settings, segment.emotion)
                        label += f" [{segment.emotion} → 语速{p_speed:.2f} 音量{p_vol:.2f} 音高{p_pitch:+d}]"
                    progress(label, int((index - 1) / max(total, 1) * 100))
                last_request_started = time.monotonic()
                audio = self._with_retry(segment, settings, voice_id)
                audio_path.write_bytes(audio)
            existing[sid] = {
                "segment_id": sid,
                "start": segment.start,
                "end": segment.end,
                "target_seconds": round(segment.target_seconds, 3),
                "text": segment.text,
                "speaker": segment.speaker,
                "language": segment.language,
                "role": segment.role,
                "audio_file": filename,
                "bytes": audio_path.stat().st_size if audio_path.exists() else 0,
                "emotion": segment.emotion or "",
                "prosody_applied": {
                    "speed": round(_apply_emotion_prosody(settings, segment.emotion)[0], 3),
                    "volume": round(_apply_emotion_prosody(settings, segment.emotion)[1], 3),
                    "pitch": _apply_emotion_prosody(settings, segment.emotion)[2],
                },
                "cache_signature": signature,
            }
            manifest["segments"] = [existing[key] for key in sorted(existing)]
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"manifest_path": str(manifest_path), "segments": manifest.get("segments", [])}

    def _with_retry(self, segment: TtsSegment, settings: MiniMaxTtsSettings, voice_id: str) -> bytes:
        last_error: Exception | None = None
        for attempt in range(1, max(1, settings.max_retries) + 1):
            try:
                return self.synthesize_text(segment.text, settings, language=segment.language, voice_id=voice_id, emotion=segment.emotion)
            except Exception as exc:
                last_error = exc
                # 模型可能不支持 emotion 参数：带情绪失败时先降级为不带情绪再试一次
                if segment.emotion and ("emotion" in str(exc).lower() or "invalid" in str(exc).lower() or "参数" in str(exc)):
                    try:
                        return self.synthesize_text(segment.text, settings, language=segment.language, voice_id=voice_id)
                    except Exception as fallback_exc:
                        last_error = fallback_exc
                if not _is_rate_limit_error(str(last_error)) or attempt >= settings.max_retries:
                    raise RuntimeError(str(last_error))
                time.sleep(max(1, settings.rate_limit_wait_seconds))
        raise RuntimeError(str(last_error) if last_error else "MiniMax TTS failed")


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"segments": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else {"segments": []}
    except Exception:
        return {"segments": []}


def _cache_signature(settings: MiniMaxTtsSettings, voice_id: str, language: str, emotion: str = "") -> dict[str, Any]:
    boost = settings.language_boost
    if boost.lower() == "auto":
        boost = LANGUAGE_BOOST.get(language.lower(), "")
    return {
        "model": settings.model,
        "voice_id": voice_id,
        "speed": settings.speed,
        "volume": settings.volume,
        "pitch": settings.pitch,
        "audio_format": settings.audio_format,
        "sample_rate": settings.sample_rate,
        "bitrate": settings.bitrate,
        "channel": settings.channel,
        "language_boost": boost,
        "emotion": emotion or "",
        "prosody": PROSODY_VERSION if (emotion and settings.emotion_prosody) else "",
    }


def _can_reuse(cached: dict[str, Any] | None, text: str, audio_path: Path, signature: dict[str, Any]) -> bool:
    if not cached or not audio_path.exists() or audio_path.stat().st_size <= 0:
        return False
    return cached.get("text") == text and cached.get("cache_signature") == signature


def _is_rate_limit_error(message: str) -> bool:
    text = message.lower()
    return "rate limit" in text or "rpm" in text or "1002" in text
