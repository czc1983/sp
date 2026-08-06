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

import requests


@dataclass
class AsrSettings:
    provider: str = "dashscope_filetrans"
    base_url: str = ""
    api_key: str = ""
    model: str = "qwen3-asr-flash-filetrans"
    max_tokens: int = 8192
    timeout_seconds: int = 1800
    poll_interval_seconds: float = 2.0
    language: str = "auto"
    enable_itn: bool = False
    enable_words: bool = True
    diarization_enabled: bool = False
    speaker_count: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AsrSettings":
        return cls(
            provider=str(data.get("provider") or "dashscope_filetrans"),
            base_url=str(data.get("base_url") or ""),
            api_key=str(data.get("api_key") or ""),
            model=str(data.get("model") or "qwen3-asr-flash-filetrans"),
            max_tokens=int(data.get("max_tokens") or 8192),
            timeout_seconds=int(data.get("timeout_seconds") or 1800),
            poll_interval_seconds=float(data.get("poll_interval_seconds") or 2.0),
            language=str(data.get("language") or "auto"),
            enable_itn=bool(data.get("enable_itn")),
            enable_words=bool(data.get("enable_words", True)),
            diarization_enabled=bool(data.get("diarization_enabled")),
            speaker_count=int(data.get("speaker_count") or 0),
        )


DIARIZATION_FALLBACK_MODEL = "fun-asr"


def supports_diarization(model: str) -> bool:
    """qwen3-asr 系列不支持说话人分离；fun-asr / paraformer 支持。"""
    name = str(model or "").strip().lower()
    if not name:
        return False
    if "qwen3-asr" in name or "qwen-asr" in name:
        return False
    return "fun-asr" in name or "paraformer" in name


class DashScopeFileTransAsrClient:
    def test_connection(self, settings: AsrSettings) -> dict[str, Any]:
        configured = bool(settings.base_url and settings.api_key and settings.model)
        return {
            "ok": configured,
            "provider": settings.provider,
            "configured": configured,
            "base_url": settings.base_url,
            "model": settings.model,
            "message": "ASR 文件转写已配置" if configured else "ASR 文件转写未配置完整",
        }

    def transcribe_audio(
        self,
        audio_path: str | Path,
        settings: AsrSettings,
        progress: Any | None = None,
    ) -> dict[str, Any]:
        path = Path(audio_path)
        if not path.exists() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"asr_audio_missing: {path}")
        if not settings.base_url or not settings.api_key or not settings.model:
            raise ValueError("ASR 文件转写未配置完整")

        def emit(message: str, percent: int) -> None:
            if progress:
                progress(message, percent)

        emit("正在申请百炼临时上传地址", 22)
        oss_url = self._upload_to_dashscope(path, settings)
        emit("正在提交 qwen3-asr-flash-filetrans 任务", 35)
        task_id = self._submit_filetrans(oss_url, settings)
        emit(f"ASR 任务已提交：{task_id}", 42)
        result = self._poll_filetrans(task_id, settings, progress=progress)
        return _normalize_filetrans_result(result)

    def _upload_to_dashscope(self, path: Path, settings: AsrSettings) -> str:
        response = requests.get(
            "https://dashscope.aliyuncs.com/api/v1/uploads",
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
            },
            params={"action": "getPolicy", "model": settings.model},
            timeout=30,
        )
        response.raise_for_status()
        policy = response.json().get("data") or {}
        upload_dir = str(policy.get("upload_dir") or "").strip("/")
        if not upload_dir:
            raise RuntimeError(f"dashscope_upload_policy_invalid: {response.text[:500]}")
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", path.name) or "source_asr.mp3"
        key = f"{upload_dir}/{int(time.time())}_{safe_name}"
        with path.open("rb") as file:
            files = {
                "OSSAccessKeyId": (None, policy.get("oss_access_key_id")),
                "Signature": (None, policy.get("signature")),
                "policy": (None, policy.get("policy")),
                "x-oss-object-acl": (None, policy.get("x_oss_object_acl")),
                "x-oss-forbid-overwrite": (None, policy.get("x_oss_forbid_overwrite")),
                "key": (None, key),
                "success_action_status": (None, "200"),
                "file": (safe_name, file, _audio_mime(path)),
            }
            upload = requests.post(str(policy.get("upload_host") or ""), files=files, timeout=300)
        upload.raise_for_status()
        return f"oss://{key}"

    def _submit_filetrans(self, oss_url: str, settings: AsrSettings) -> str:
        parameters: dict[str, Any] = {
            "channel_id": [0],
            "enable_itn": settings.enable_itn,
            "enable_words": settings.enable_words,
        }
        if settings.diarization_enabled:
            parameters["diarization_enabled"] = True
            if settings.speaker_count >= 2:
                parameters["speaker_count"] = settings.speaker_count
        input_payload: dict[str, Any]
        if supports_diarization(settings.model):
            # fun-asr / paraformer 录音文件识别使用 file_urls 数组
            input_payload = {"file_urls": [oss_url]}
        else:
            input_payload = {"file_url": oss_url}
        payload = {
            "model": settings.model,
            "input": input_payload,
            "parameters": parameters,
        }
        if settings.language and settings.language != "auto" and "fun-asr" not in settings.model.lower():
            payload["parameters"]["language_hints"] = [settings.language]
        response = requests.post(
            f"{_dashscope_api_base(settings.base_url)}/services/audio/asr/transcription",
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
                "X-DashScope-Async": "enable",
                "X-DashScope-OssResourceResolve": "enable",
            },
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        task_id = str(data.get("output", {}).get("task_id") or "")
        if not task_id:
            raise RuntimeError(f"dashscope_asr_task_missing: {response.text[:500]}")
        return task_id

    def _poll_filetrans(
        self,
        task_id: str,
        settings: AsrSettings,
        progress: Any | None = None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        last_percent = 45
        while True:
            response = requests.get(
                f"{_dashscope_api_base(settings.base_url)}/tasks/{task_id}",
                headers={
                    "Authorization": f"Bearer {settings.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            output = data.get("output") or {}
            status = str(output.get("task_status") or "")
            if status == "SUCCEEDED":
                result_url = _extract_transcription_url(output)
                if not result_url:
                    raise RuntimeError(f"dashscope_asr_result_url_missing: {response.text[:500]}")
                result_response = requests.get(result_url, timeout=120)
                result_response.raise_for_status()
                return result_response.json()
            if status in {"FAILED", "UNKNOWN"}:
                raise RuntimeError(f"dashscope_asr_failed: {response.text[:1000]}")
            elapsed = time.monotonic() - started
            if elapsed > max(30, settings.timeout_seconds):
                raise TimeoutError(f"dashscope_asr_timeout: {task_id}")
            last_percent = min(84, last_percent + 2)
            if progress:
                progress(f"ASR 任务状态：{status or 'PENDING'}", last_percent)
            time.sleep(max(0.5, settings.poll_interval_seconds))


class OpenAICompatibleAsrClient:
    def test_connection(self, settings: AsrSettings) -> dict[str, Any]:
        configured = bool(settings.base_url and settings.api_key and settings.model)
        return {
            "ok": configured,
            "provider": settings.provider,
            "configured": configured,
            "base_url": settings.base_url,
            "model": settings.model,
            "message": "ASR 模型已配置" if configured else "ASR 模型未配置完整",
        }

    def transcribe_audio(self, audio_path: str | Path, settings: AsrSettings) -> dict[str, Any]:
        path = Path(audio_path)
        if not path.exists() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"asr_audio_missing: {path}")
        if path.stat().st_size > 7 * 1024 * 1024:
            raise ValueError("ASR 音频超过 7MB，请先用短视频测试；长视频分块后续接入。")
        if not settings.base_url or not settings.api_key or not settings.model:
            raise ValueError("ASR 模型未配置完整")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        prompt = (
            "你是视频字幕转写助手。请完整听取音频，按时间顺序输出所有可听见的对白。"
            "不要翻译，只转写原文。没有对白的声音事件可以忽略。"
            "时间戳单位为秒，尽量精确到 0.1 秒。"
            "输出严格 JSON："
            '{"language":"zh","segments":[{"start":0.0,"end":1.2,"speaker":"说话人1","text":"原文台词"}]}'
        )
        media_content = {
            "type": "input_audio",
            "input_audio": {
                "data": f"data:;base64,{encoded}",
                "format": path.suffix.lower().lstrip(".") or "mp3",
            },
        }
        payload = {
            "model": settings.model,
            "messages": [{"role": "user", "content": [media_content, {"type": "text", "text": prompt}]}],
            "max_tokens": settings.max_tokens,
            "response_format": {"type": "json_object"},
        }
        data = self._post_chat_completions(settings, payload)
        content = str(data["choices"][0]["message"]["content"])
        parsed = _load_json_content(content)
        utterances = parsed.get("segments") or parsed.get("utterances") or []
        if not isinstance(utterances, list):
            raise RuntimeError(f"ASR 返回格式不正确: {content[:500]}")
        segments: list[dict[str, Any]] = []
        for index, raw in enumerate(utterances, start=1):
            if not isinstance(raw, dict):
                continue
            text = str(raw.get("text") or "").strip()
            if not text:
                continue
            start = _time_seconds(raw.get("start"))
            end = max(start + 0.1, _time_seconds(raw.get("end"), fallback=start + 2.0))
            segments.append({
                "segment_id": f"seg_{index:03d}",
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": str(raw.get("speaker") or "说话人1"),
                "text": text,
                "language": str(parsed.get("language") or ""),
            })
        if not segments:
            raise RuntimeError("ASR 未识别到字幕段")
        return {
            "language": str(parsed.get("language") or ""),
            "summary": str(parsed.get("summary") or ""),
            "segments": segments,
        }

    def _post_chat_completions(self, settings: AsrSettings, payload: dict[str, Any]) -> dict[str, Any]:
        base = settings.base_url.rstrip("/")
        url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        request_payload = dict(payload)
        use_stream = "qwen" in settings.model.lower() and "omni" in settings.model.lower()
        if use_stream:
            request_payload["stream"] = True
            request_payload["stream_options"] = {"include_usage": True}
            request_payload.setdefault("modalities", ["text"])
        request = urllib.request.Request(
            url,
            data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                raw = response.read().decode("utf-8", errors="replace")
                content_type = str(response.headers.get("Content-Type") or "").lower()
                if "text/event-stream" in content_type or raw.lstrip().startswith("data:"):
                    return _stream_to_chat_response(raw)
                return json.loads(raw)
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ASR 模型 HTTP {error.code}: {body}") from error


def _time_seconds(value: Any, fallback: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    text = str(value or "").strip()
    if not text:
        return max(0.0, float(fallback))
    if ":" not in text:
        try:
            return max(0.0, float(text))
        except ValueError:
            return max(0.0, float(fallback))
    parts = [float(part.replace(",", ".")) for part in text.split(":")]
    total = 0.0
    for part in parts:
        total = total * 60 + part
    return max(0.0, total)


def _extract_transcription_url(output: dict[str, Any]) -> str:
    """qwen3 filetrans 结果在 output.result.transcription_url；fun-asr/paraformer 在 output.transcription_url 或 output.results[0]。"""
    candidates: list[Any] = [
        (output.get("result") or {}).get("transcription_url"),
        output.get("transcription_url"),
    ]
    results = output.get("results") or []
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict):
                candidates.append(item.get("transcription_url"))
    for candidate in candidates:
        url = str(candidate or "").strip()
        if url:
            return url
    return ""


def _normalize_filetrans_result(data: dict[str, Any]) -> dict[str, Any]:
    transcripts = data.get("transcripts") or []
    segments: list[dict[str, Any]] = []
    language = ""
    for transcript in transcripts:
        if not isinstance(transcript, dict):
            continue
        channel_id = transcript.get("channel_id")
        channel_speaker = _friendly_channel_speaker(channel_id)
        sentences = transcript.get("sentences") or []
        if isinstance(sentences, list) and sentences:
            for sentence in sentences:
                if not isinstance(sentence, dict):
                    continue
                if not language:
                    language = str(sentence.get("language") or "")
                speaker = _sentence_speaker(sentence, channel_speaker)
                words = sentence.get("words") or []
                if isinstance(words, list) and words:
                    for segment in _segments_from_words(words, speaker=speaker, language=language):
                        segment["segment_id"] = f"seg_{len(segments) + 1:03d}"
                        segments.append(segment)
                    continue
                text = str(sentence.get("text") or "").strip()
                if not text:
                    continue
                start = _ms_to_seconds(sentence.get("begin_time"))
                end = max(start + 0.1, _ms_to_seconds(sentence.get("end_time"), fallback=(start + 2.0) * 1000))
                segments.append({
                    "segment_id": f"seg_{len(segments) + 1:03d}",
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "speaker": speaker,
                    "text": text,
                    "language": language,
                })
            continue
        text = str(transcript.get("text") or "").strip()
        if text:
            segments.append({
                "segment_id": f"seg_{len(segments) + 1:03d}",
                "start": 0.0,
                "end": 2.0,
                "speaker": channel_speaker,
                "text": text,
                "language": language,
            })
    if not segments:
        raise RuntimeError("ASR 没有识别到字幕段。")
    return {
        "language": language,
        "summary": "",
        "segments": sorted(segments, key=lambda item: (float(item.get("start") or 0), float(item.get("end") or 0))),
        "raw": data,
    }


def _sentence_speaker(sentence: dict[str, Any], fallback: str) -> str:
    """开启说话人分离后，每句带 speaker_id（从 0 开始）。"""
    speaker_id = sentence.get("speaker_id")
    if speaker_id in (None, ""):
        return fallback
    try:
        return f"说话人{int(speaker_id) + 1}"
    except (TypeError, ValueError):
        return fallback


def _segments_from_words(words: list[Any], *, speaker: str, language: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    chunk: list[str] = []
    chunk_start: float | None = None
    chunk_end = 0.0
    previous_end: float | None = None

    def flush() -> None:
        nonlocal chunk, chunk_start, chunk_end, previous_end
        text = "".join(chunk).strip()
        if text and chunk_start is not None:
            segments.append({
                "segment_id": "",
                "start": round(chunk_start, 3),
                "end": round(max(chunk_start + 0.1, chunk_end), 3),
                "speaker": speaker,
                "text": text,
                "language": language,
            })
        chunk = []
        chunk_start = None
        chunk_end = 0.0
        previous_end = None

    for item in words:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        punctuation = str(item.get("punctuation") or "")
        start = _ms_to_seconds(item.get("begin_time"), fallback=chunk_end * 1000)
        end = max(start + 0.05, _ms_to_seconds(item.get("end_time"), fallback=(start + 0.1) * 1000))
        if previous_end is not None and start - previous_end >= 1.4 and chunk:
            flush()
        if chunk_start is None:
            chunk_start = start
        chunk.append(text + punctuation)
        chunk_end = max(chunk_end, end)
        previous_end = end

        duration = chunk_end - (chunk_start if chunk_start is not None else chunk_end)
        char_count = len("".join(chunk))
        is_strong_punctuation = punctuation in {"。", "！", "？", "!", "?", "；", ";"}
        is_soft_break = punctuation in {"，", ",", "、", "：", ":"} and (duration >= 2.4 or char_count >= 12)
        is_too_long = duration >= 5.2 or char_count >= 24
        if is_strong_punctuation or is_soft_break or is_too_long:
            flush()
    flush()
    return segments


def _ms_to_seconds(value: Any, fallback: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, float(value) / 1000.0)
    text = str(value or "").strip()
    if not text:
        return max(0.0, float(fallback) / 1000.0)
    try:
        return max(0.0, float(text) / 1000.0)
    except ValueError:
        return max(0.0, float(fallback) / 1000.0)


def _dashscope_api_base(base_url: str) -> str:
    base = str(base_url or "").rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    if base.endswith("/compatible-mode/v1"):
        return base[: -len("/compatible-mode/v1")] + "/api/v1"
    if base.endswith("/api/v1"):
        return base
    return base + "/api/v1"


def _audio_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
    }.get(suffix, "application/octet-stream")


def _friendly_channel_speaker(channel_id: Any) -> str:
    if channel_id in (None, ""):
        return "说话人1"
    try:
        return f"说话人{int(channel_id) + 1}"
    except (TypeError, ValueError):
        return "说话人1"


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
        raise ValueError("asr_json_must_be_object")
    return data


def _stream_to_chat_response(raw: str) -> dict[str, Any]:
    parts: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        chunk = json.loads(data)
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str):
                parts.append(content)
    if not parts:
        raise RuntimeError("ASR stream response is empty")
    return {"choices": [{"message": {"content": "".join(parts)}}]}
