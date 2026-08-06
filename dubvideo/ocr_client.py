from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class OcrSettings:
    provider: str = "openai_compatible_vision"
    base_url: str = ""
    api_key: str = ""
    model: str = "qwen-vl-ocr-latest"
    crop_bottom_ratio: float = 0.5
    frames_per_segment: int = 6
    request_interval_ms: int = 300
    timeout_seconds: int = 120

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OcrSettings":
        return cls(
            provider=str(data.get("provider") or "openai_compatible_vision"),
            base_url=str(data.get("base_url") or ""),
            api_key=str(data.get("api_key") or ""),
            model=str(data.get("model") or "qwen-vl-ocr-latest"),
            crop_bottom_ratio=float(data.get("crop_bottom_ratio") or 0.5),
            frames_per_segment=int(data.get("frames_per_segment") or 6),
            request_interval_ms=int(data.get("request_interval_ms") or 300),
            timeout_seconds=int(data.get("timeout_seconds") or 120),
        )


class QwenOcrClient:
    def test_connection(self, settings: OcrSettings, image_path: str | Path) -> dict[str, Any]:
        configured = bool(settings.base_url and settings.api_key and settings.model)
        result = {
            "ok": configured,
            "provider": settings.provider,
            "configured": configured,
            "base_url": settings.base_url,
            "model": settings.model,
        }
        if not configured:
            result["message"] = "OCR 模型未配置完整"
            return result
        try:
            text = self.recognize_image(Path(image_path), settings, prompt="只输出图片中的文字，不要解释。")
            result.update({"ok": True, "message": f"OCR 模型可用：{settings.model}", "sample": text[:120]})
            return result
        except Exception as exc:  # noqa: BLE001
            result.update({"ok": False, "message": f"OCR 模型不可用：{settings.model}", "error": str(exc)})
            return result

    def recognize_image(self, image_path: str | Path, settings: OcrSettings, *, prompt: str = "") -> str:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"ocr_image_not_found: {path}")
        if not settings.base_url or not settings.api_key or not settings.model:
            raise ValueError("ocr_model_not_configured")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        payload = {
            "model": settings.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                        {
                            "type": "text",
                            "text": prompt or "只输出画面中的字幕文字；如果没有字幕，输出空字符串。不要解释。",
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 160,
        }
        data = self._post_chat_completions(settings, payload)
        content = str(data.get("choices", [{}])[0].get("message", {}).get("content", ""))
        return clean_ocr_text(content)

    def _post_chat_completions(self, settings: OcrSettings, payload: dict[str, Any]) -> dict[str, Any]:
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
            with urllib.request.urlopen(request, timeout=max(10, settings.timeout_seconds)) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OCR 模型 HTTP {error.code}: {body}") from error


def clean_ocr_text(value: str) -> str:
    text = str(value or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:text|json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    text = text.strip().strip("\"'“”‘’")
    text = re.sub(r"\s+", "", text)
    empty_values = {"", "0", "无", "无字幕", "没有字幕", "空字符串", "null", "None", "none"}
    if text in empty_values:
        return ""
    no_text_markers = (
        "无法识别",
        "无法提供",
        "请求的帮助",
        "抱歉",
        "看不清",
        "模糊",
        "没有清晰字幕",
        "没有看到字幕",
        "未检测到字幕",
        "图片中没有",
    )
    if any(marker in text for marker in no_text_markers):
        return ""
    if not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", text):
        return ""
    return text


def merge_ocr_texts(values: list[str]) -> str:
    merged = ""
    for raw in values:
        text = clean_ocr_text(raw)
        if not text:
            continue
        if not merged:
            merged = text
            continue
        if text in merged:
            continue
        if merged in text:
            merged = text
            continue
        overlap = _suffix_prefix_overlap(merged, text)
        if overlap:
            merged += text[overlap:]
        else:
            merged += text
    return merged


def choose_source_text(asr_text: str, ocr_text: str) -> tuple[str, str]:
    asr = str(asr_text or "").strip()
    ocr = clean_ocr_text(ocr_text)
    if not ocr:
        return asr, "asr"
    if not asr:
        return ocr, "ocr"
    asr_norm = _normalize_compare_text(asr)
    ocr_norm = _normalize_compare_text(ocr)
    if not ocr_norm:
        return asr, "asr"
    if not asr_norm:
        return ocr, "ocr"
    overlap = _lcs_length(asr_norm, ocr_norm)
    overlap_ratio = overlap / max(1, min(len(asr_norm), len(ocr_norm)))
    ocr_is_shorter = len(ocr_norm) < len(asr_norm) * 0.92
    ocr_is_longer = len(ocr_norm) > len(asr_norm) * 1.15
    if overlap_ratio < 0.45:
        return ocr, "ocr_asr_conflict"
    if ocr_is_shorter or ocr_is_longer:
        return ocr, "ocr_needs_review"
    return ocr, "ocr"


def _suffix_prefix_overlap(left: str, right: str) -> int:
    max_len = min(len(left), len(right))
    for size in range(max_len, 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def _normalize_compare_text(value: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", str(value or "")).lower()


def _lcs_length(left: str, right: str) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_ch in left:
        current = [0]
        for index, right_ch in enumerate(right, start=1):
            if left_ch == right_ch:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def throttle(last_started: float, interval_ms: int) -> float:
    if interval_ms > 0 and last_started > 0:
        wait_ms = interval_ms - int((time.monotonic() - last_started) * 1000)
        if wait_ms > 0:
            time.sleep(wait_ms / 1000)
    return time.monotonic()
