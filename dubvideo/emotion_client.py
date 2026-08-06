from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# MiniMax t2a_v2 支持的情绪标签
MINIMAX_EMOTIONS = ("happy", "sad", "angry", "fearful", "disgusted", "surprise", "neutral")

# 模型自由回答 → MiniMax 情绪标签
_EMOTION_ALIASES: dict[str, str] = {
    "happy": "happy", "joy": "happy", "joyful": "happy", "excited": "happy", "cheerful": "happy",
    "开心": "happy", "高兴": "happy", "快乐": "happy", "喜悦": "happy", "兴奋": "happy", "欢快": "happy",
    "sad": "sad", "sadness": "sad", "sorrow": "sad", "grief": "sad", "crying": "sad", "tearful": "sad",
    "伤心": "sad", "悲伤": "sad", "难过": "sad", "哀伤": "sad", "悲痛": "sad", "委屈": "sad", "哭": "sad", "哭泣": "sad", "哽咽": "sad",
    "angry": "angry", "anger": "angry", "furious": "angry", "rage": "angry", "mad": "angry", "shouting": "angry", "yelling": "angry",
    "愤怒": "angry", "生气": "angry", "暴怒": "angry", "怒吼": "angry", "气愤": "angry", "咆哮": "angry", "大喊": "angry", "喊叫": "angry",
    "fearful": "fearful", "fear": "fearful", "afraid": "fearful", "scared": "fearful", "terrified": "fearful", "nervous": "fearful",
    "害怕": "fearful", "恐惧": "fearful", "惊恐": "fearful", "紧张": "fearful", "惊慌": "fearful",
    "disgusted": "disgusted", "disgust": "disgusted", "contempt": "disgusted",
    "厌恶": "disgusted", "恶心": "disgusted", "嫌弃": "disgusted", "鄙视": "disgusted",
    "surprise": "surprise", "surprised": "surprise", "amazed": "surprise", "shocked": "surprise", "astonished": "surprise",
    "惊讶": "surprise", "吃惊": "surprise", "震惊": "surprise", "意外": "surprise",
    "neutral": "neutral", "calm": "neutral", "flat": "neutral", "normal": "neutral",
    "平静": "neutral", "中性": "neutral", "正常": "neutral", "冷静": "neutral", "平淡": "neutral",
}

_PROMPT = (
    "听这段语音，只根据说话者的语气、音高、音量和能量判断情绪与强度，不要根据台词的文字内容判断。"
    "严格用「情绪:强度」的格式回答，不要解释。"
    "情绪只能选一个：happy(开心)、sad(伤心)、angry(愤怒/大喊)、fearful(害怕)、disgusted(厌恶)、surprise(惊讶)、neutral(平静)。"
    "强度只能选一个：low(轻声/压抑/克制)、medium(正常说话)、high(大喊/吼叫/爆发/哭喊)。"
    "示例：angry:high、sad:low、neutral:medium。"
    "判断依据：音量很大、音高拉高、爆发力强、吼叫或哭喊 → high；音量正常、语速平稳 → medium；轻声细语、压低声音 → low。"
)

_INTENSITY_ALIASES: dict[str, str] = {
    "low": "low", "weak": "low", "mild": "low", "soft": "low", "quiet": "low",
    "medium": "medium", "moderate": "medium", "normal": "medium",
    "high": "high", "strong": "high", "intense": "high", "extreme": "high",
    "轻": "low", "弱": "low", "低": "low", "轻声": "low",
    "中": "medium", "中等": "medium",
    "强": "high", "高": "high", "爆发": "high",
}


@dataclass
class EmotionSettings:
    base_url: str = ""
    api_key: str = ""
    model: str = "qwen3-omni-flash"
    timeout_seconds: int = 60

    @classmethod
    def from_translate_settings(cls, data: dict[str, Any], model: str = "") -> "EmotionSettings":
        return cls(
            base_url=str(data.get("base_url") or ""),
            api_key=str(data.get("api_key") or ""),
            model=model or "qwen3-omni-flash",
        )

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)


class QwenAudioEmotionClient:
    """用千问 Audio（OpenAI 兼容接口）识别短音频片段的主要情绪。"""

    def analyze(self, audio_path: str | Path, settings: EmotionSettings) -> str:
        """返回 MiniMax 情绪标签；识别不出时返回空串。"""
        return self.analyze_detailed(audio_path, settings)[0]

    def analyze_detailed(self, audio_path: str | Path, settings: EmotionSettings) -> tuple[str, str]:
        """返回 (情绪标签, 强度 low/medium/high)；识别不出时强度为空串。"""
        path = Path(audio_path)
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"emotion_audio_missing: {path}")
        if not settings.configured:
            raise ValueError("emotion_model_not_configured")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4"}.get(path.suffix.lower(), "audio/mpeg")
        payload = {
            "model": settings.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": f"data:{mime};base64,{encoded}", "format": path.suffix.lstrip(".").lower() or "mp3"}},
                    {"type": "text", "text": _PROMPT},
                ],
            }],
            "max_tokens": 24,
            "temperature": 0.0,
        }
        base = settings.base_url.rstrip("/")
        url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=settings.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"emotion HTTP {error.code}: {body[:300]}") from error
        content = str((((data.get("choices") or [{}])[0]).get("message") or {}).get("content") or "")
        if isinstance(content, list):
            content = " ".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
        return normalize_emotion_with_intensity(content)


def normalize_emotion_with_intensity(text: str) -> tuple[str, str]:
    """解析「情绪:强度」格式的回答，返回 (情绪标签, 强度)。"""
    emotion = normalize_emotion(text)
    cleaned = re.sub(r"[^a-zA-Z一-鿿]+", " ", str(text or "")).strip().lower()
    intensity = ""
    if cleaned:
        for token in cleaned.split():
            alias = _INTENSITY_ALIASES.get(token)
            if alias:
                intensity = alias
                break
        if not intensity:
            for key, alias in _INTENSITY_ALIASES.items():
                if len(key) >= 2 and key in cleaned:
                    intensity = alias
                    break
    return emotion, intensity


def normalize_emotion(text: str) -> str:
    """把模型的自由回答映射到 MiniMax 情绪标签；映射不上返回空串。"""
    cleaned = re.sub(r"[^a-zA-Z一-鿿]+", " ", str(text or "")).strip().lower()
    if not cleaned:
        return ""
    for token in cleaned.split():
        if token in MINIMAX_EMOTIONS:
            return token
        alias = _EMOTION_ALIASES.get(token)
        if alias:
            return alias
    for key, alias in _EMOTION_ALIASES.items():
        if key in cleaned:
            return alias
    return ""
