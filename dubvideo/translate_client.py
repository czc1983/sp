from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class TranslateSettings:
    provider: str = "openai_compatible"
    base_url: str = ""
    api_key: str = ""
    model: str = "qwen-plus"
    temperature: float = 0.2
    max_tokens: int = 4096
    batch_size: int = 20
    request_interval_ms: int = 500

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranslateSettings":
        return cls(
            provider=str(data.get("provider") or "openai_compatible"),
            base_url=str(data.get("base_url") or ""),
            api_key=str(data.get("api_key") or ""),
            model=str(data.get("model") or "qwen-plus"),
            temperature=float(data.get("temperature") if data.get("temperature") is not None else 0.2),
            max_tokens=int(data.get("max_tokens") or 4096),
            batch_size=int(data.get("batch_size") or 20),
            request_interval_ms=int(data.get("request_interval_ms") or 500),
        )


LANGUAGE_NAMES = {
    "zh": "Chinese",
    "zh-CN": "Simplified Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pt-BR": "Brazilian Portuguese",
}

CHINESE_SURNAME_CHARS = set(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
    "戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐"
    "费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平"
    "黄和穆萧尹邢"
)

PINYIN_FALLBACK = {
    "听": "Ting",
    "然": "Ran",
    "邢": "Xing",
    "冽": "Lie",
    "羽": "Yu",
    "秦": "Qin",
    "丽": "Li",
}

LEADING_NAME_FOLLOWERS = set("你我他她它这那快别给把真怎到是叫说放来去在就也都还再")
NON_NAME_SHORT_TEXTS = {
    "不要",
    "再说",
    "醒了",
    "假的",
    "老公",
    "老婆",
    "结婚证",
    "叫声",
}


class OpenAICompatibleTranslator:
    def test_connection(self, settings: TranslateSettings) -> dict[str, Any]:
        configured = bool(settings.base_url and settings.api_key and settings.model)
        result = {
            "ok": configured,
            "provider": settings.provider,
            "configured": configured,
            "base_url": settings.base_url,
            "model": settings.model,
        }
        if not configured:
            result["message"] = "翻译模型未配置完整"
            return result
        try:
            data = self._post_chat_completions(settings, {
                "model": settings.model,
                "messages": [
                    {"role": "system", "content": "Return strict JSON only."},
                    {"role": "user", "content": "Return {\"ok\":true}."},
                ],
                "temperature": 0,
                "max_tokens": 32,
                "response_format": {"type": "json_object"},
            })
            content = str(data.get("choices", [{}])[0].get("message", {}).get("content", ""))
            result.update({"ok": True, "message": f"翻译模型可用：{settings.model}", "sample": content[:120]})
            return result
        except Exception as exc:  # noqa: BLE001
            result.update({"ok": False, "message": f"翻译模型不可用：{settings.model}", "error": str(exc)})
            return result

    def translate_segments(
        self,
        segments: list[dict[str, Any]],
        settings: TranslateSettings,
        *,
        source_language: str = "auto",
        target_language: str = "en",
        progress: Callable[[str, int], None] | None = None,
    ) -> list[dict[str, Any]]:
        if not segments:
            raise ValueError("missing_source_segments")
        if not settings.base_url or not settings.api_key or not settings.model:
            raise ValueError("translate_model_not_configured")

        name_glossary = _build_name_glossary(segments, target_language)
        translated_by_id: dict[str, str] = {}
        batches = list(_chunks(segments, max(1, settings.batch_size)))
        total = len(batches)
        last_request_started = 0.0
        for index, batch in enumerate(batches, start=1):
            if settings.request_interval_ms > 0 and last_request_started > 0:
                wait_ms = settings.request_interval_ms - int((time.monotonic() - last_request_started) * 1000)
                if wait_ms > 0:
                    time.sleep(wait_ms / 1000)
            if progress:
                progress(f"翻译字幕批次 {index}/{total}", int((index - 1) / max(total, 1) * 90))
            last_request_started = time.monotonic()
            result = self._translate_batch(
                batch,
                settings,
                source_language=source_language,
                target_language=target_language,
                name_glossary=name_glossary,
            )
            translated_by_id.update(result)

        output: list[dict[str, Any]] = []
        for raw in segments:
            item = dict(raw)
            sid = str(item.get("segment_id") or item.get("id") or "")
            item["segment_id"] = sid
            translated_text = translated_by_id.get(sid, str(item.get("text") or ""))
            item["text"] = _protect_required_names(
                str(item.get("text") or item.get("source_text") or ""),
                translated_text,
                name_glossary,
                target_language,
            )
            item["language"] = target_language
            output.append(item)
        if progress:
            progress("翻译字幕完成", 95)
        return output

    def _translate_batch(
        self,
        segments: list[dict[str, Any]],
        settings: TranslateSettings,
        *,
        source_language: str,
        target_language: str,
        name_glossary: list[dict[str, str]],
    ) -> dict[str, str]:
        source_name = LANGUAGE_NAMES.get(source_language, source_language or "auto")
        target_name = LANGUAGE_NAMES.get(target_language, target_language or "English")
        compact = [
            {
                "segment_id": str(item.get("segment_id") or item.get("id") or f"seg_{index:03d}"),
                "text": str(item.get("text") or item.get("source_text") or ""),
                "target_seconds": round(
                    float(item.get("target_seconds") or (float(item.get("end") or 0) - float(item.get("start") or 0)) or 0),
                    2,
                ),
                "role": str(item.get("role") or item.get("segment_type") or "dialogue"),
                "name_hints": _names_in_text(
                    str(item.get("text") or item.get("source_text") or ""),
                    name_glossary,
                ),
            }
            for index, item in enumerate(segments, start=1)
        ]
        prompt = (
            "You are a professional subtitle translator for video dubbing and lip sync.\n"
            f"Translate the subtitle text from {source_name} to {target_name}.\n"
            "Keep meaning natural, concise, spoken, and suitable for TTS.\n"
            "Each segment has target_seconds; keep the translation short enough to speak naturally within that time.\n"
            "Never omit person names, even when the segment is short. Preserve names from name_hints.\n"
            "For Chinese person names, use the target form from the glossary exactly.\n"
            "If a subtitle begins with a name without punctuation, treat it as a vocative and keep it at the beginning, e.g. 'Ting Ran, ...'.\n"
            "If role is narration, keep it natural for voiceover. If role is dialogue, keep it lip-sync friendly.\n"
            "Do not change segment_id. Do not add explanations.\n"
            "Name glossary:\n"
            + json.dumps(name_glossary, ensure_ascii=False)
            + "\n"
            "Return strict JSON only in this schema:\n"
            "{\"segments\":[{\"segment_id\":\"1\",\"text\":\"translated text\"}]}\n\n"
            "Segments:\n"
            + json.dumps(compact, ensure_ascii=False)
        )
        payload = {
            "model": settings.model,
            "messages": [
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens,
            "response_format": {"type": "json_object"},
        }
        data = self._post_chat_completions(settings, payload)
        content = str(data["choices"][0]["message"]["content"])
        parsed = _load_json_content(content)
        values = parsed.get("segments") if isinstance(parsed, dict) else []
        if not isinstance(values, list):
            raise RuntimeError(f"translate_response_invalid: {content[:500]}")
        result: dict[str, str] = {}
        for item in values:
            if isinstance(item, dict):
                sid = str(item.get("segment_id") or "").strip()
                text = str(item.get("text") or "").strip()
                if sid and text:
                    result[sid] = text
        return result

    def _post_chat_completions(self, settings: TranslateSettings, payload: dict[str, Any]) -> dict[str, Any]:
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
            with urllib.request.urlopen(request, timeout=240) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"翻译模型 HTTP {error.code}: {body}") from error


def _build_name_glossary(segments: list[dict[str, Any]], target_language: str) -> list[dict[str, str]]:
    texts = [str(item.get("text") or item.get("source_text") or "").strip() for item in segments]
    cjk_texts = [_cjk_only(text) for text in texts if text]
    isolated = {
        text
        for text in cjk_texts
        if 2 <= len(text) <= 3 and text not in NON_NAME_SHORT_TEXTS
    }
    candidates: set[str] = set()
    for text in cjk_texts:
        if len(text) <= 2:
            continue
        if len(text) > 2:
            leading = text[:2]
            follower = text[2:3]
            if (
                leading not in NON_NAME_SHORT_TEXTS
                and (leading in isolated or (follower in LEADING_NAME_FOLLOWERS and leading[0] in CHINESE_SURNAME_CHARS))
            ):
                candidates.add(leading)
        if len(text) > 3:
            leading = text[:3]
            if leading in isolated and leading not in NON_NAME_SHORT_TEXTS:
                candidates.add(leading)
    for name in isolated:
        if sum(1 for text in cjk_texts if name in text) >= 2:
            candidates.add(name)
    values = sorted(candidates, key=lambda value: (-len(value), value))
    return [
        {"source": value, "target": _romanize_chinese_name(value, target_language)}
        for value in values
    ]


def _cjk_only(value: str) -> str:
    return "".join(re.findall(r"[\u4e00-\u9fff]", str(value or "")))


def _romanize_chinese_name(value: str, target_language: str) -> str:
    if target_language in {"zh", "zh-CN"}:
        return value
    try:
        from pypinyin import lazy_pinyin  # type: ignore

        parts = lazy_pinyin(value)
        if parts:
            return " ".join(part.capitalize() for part in parts)
    except Exception:
        pass
    parts = [PINYIN_FALLBACK.get(char, "") for char in value]
    if all(parts):
        return " ".join(parts)
    return value


def _names_in_text(text: str, name_glossary: list[dict[str, str]]) -> list[dict[str, str]]:
    source = str(text or "")
    return [item for item in name_glossary if item.get("source") and str(item["source"]) in source]


def _protect_required_names(
    source_text: str,
    translated_text: str,
    name_glossary: list[dict[str, str]],
    target_language: str,
) -> str:
    text = str(translated_text or "").strip()
    if not text:
        return text
    source_cjk = _cjk_only(source_text)
    for item in name_glossary:
        source_name = str(item.get("source") or "")
        target_name = str(item.get("target") or source_name)
        if not source_name or not target_name:
            continue
        if not source_cjk.startswith(source_name):
            continue
        if _translation_contains_name(text, source_name, target_name):
            continue
        separator = "" if target_language in {"zh", "zh-CN", "ja", "ko"} else ", "
        return f"{target_name}{separator}{text}"
    return text


def _translation_contains_name(text: str, source_name: str, target_name: str) -> bool:
    normalized_text = _normalize_name_compare(text)
    return (
        _normalize_name_compare(source_name) in normalized_text
        or _normalize_name_compare(target_name) in normalized_text
    )


def _normalize_name_compare(value: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", str(value or "")).lower()


def _chunks(values: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [values[index:index + size] for index in range(0, len(values), size)]


def _load_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("translate_json_must_be_object")
    return data
