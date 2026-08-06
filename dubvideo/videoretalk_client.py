from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import requests

from .asr_client import _dashscope_api_base

UPLOAD_POLICY_URL = "https://dashscope.aliyuncs.com/api/v1/uploads"
SUBMIT_PATH = "/services/aigc/image2video/video-synthesis/"
PIXVERSE_SUBMIT_PATH = "/services/aigc/video-generation/video-synthesis"


class VideoRetalkClient:
    """阿里百炼声动人像 VideoRetalk：人物视频 + 人声音频 → 口型同步视频（异步任务）。"""

    def run(
        self,
        *,
        video_path: str | Path,
        audio_path: str | Path,
        output_path: str | Path,
        api_key: str,
        base_url: str,
        model: str = "videoretalk",
        video_extension: bool = False,
        timeout_seconds: int = 3600,
        poll_interval_seconds: float = 5.0,
        progress: Callable[[str, int], None] | None = None,
    ) -> dict[str, Any]:
        video = Path(video_path)
        audio = Path(audio_path)
        if not video.is_file():
            raise FileNotFoundError(f"videoretalk_video_missing: {video}")
        if not audio.is_file():
            raise FileNotFoundError(f"videoretalk_audio_missing: {audio}")
        if not api_key:
            raise ValueError("videoretalk_api_key_missing")
        api_base = _dashscope_api_base(base_url)

        def emit(message: str, percent: int) -> None:
            if progress:
                progress(message, percent)

        label = "PixVerse 对口型" if "pixverse" in model.lower() else "VideoRetalk 口型"
        emit("上传原视频到百炼临时存储", 12)
        video_url = upload_to_dashscope(video, api_key, model)
        emit("上传配音音频到百炼临时存储", 24)
        audio_url = upload_to_dashscope(audio, api_key, model)
        emit(f"提交{label}任务", 30)
        task_id = self._submit(api_base, api_key, model, video_url, audio_url, video_extension)
        emit(f"{label}任务已提交：{task_id}", 36)
        result = self._poll(api_base, api_key, task_id, timeout_seconds, poll_interval_seconds, progress)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        emit("下载口型结果视频", 92)
        response = requests.get(result["video_url"], timeout=600)
        response.raise_for_status()
        output.write_bytes(response.content)
        return {
            "ok": True,
            "task_id": task_id,
            "output_video_path": str(output),
            "usage": result.get("usage") or {},
        }

    def _submit(
        self,
        api_base: str,
        api_key: str,
        model: str,
        video_url: str,
        audio_url: str,
        video_extension: bool,
    ) -> str:
        if "pixverse" in model.lower():
            payload = {
                "model": model,
                "input": {
                    "media": [
                        {"type": "video_url", "url": video_url},
                        {"type": "audio_url", "url": audio_url},
                    ]
                },
                "parameters": {"watermark": False},
            }
            submit_path = PIXVERSE_SUBMIT_PATH
        else:
            payload = {
                "model": model,
                "input": {
                    "video_url": video_url,
                    "audio_url": audio_url,
                    "ref_image_url": "",
                },
                "parameters": {"video_extension": bool(video_extension)},
            }
            submit_path = SUBMIT_PATH
        response = requests.post(
            f"{api_base}{submit_path}",
            headers={
                "Authorization": f"Bearer {api_key}",
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
            raise RuntimeError(f"videoretalk_task_missing: {response.text[:500]}")
        return task_id

    def _poll(
        self,
        api_base: str,
        api_key: str,
        task_id: str,
        timeout_seconds: int,
        poll_interval_seconds: float,
        progress: Callable[[str, int], None] | None,
    ) -> dict[str, Any]:
        started = time.monotonic()
        last_percent = 40
        while True:
            response = requests.get(
                f"{api_base}/tasks/{task_id}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            output = data.get("output") or {}
            status = str(output.get("task_status") or "")
            if status == "SUCCEEDED":
                video_url = str(output.get("video_url") or "")
                if not video_url:
                    raise RuntimeError(f"videoretalk_result_url_missing: {response.text[:500]}")
                return {"video_url": video_url, "usage": data.get("usage") or {}}
            if status in {"FAILED", "UNKNOWN", "CANCELED"}:
                message = output.get("message") or output.get("code") or response.text[:800]
                raise RuntimeError(f"videoretalk_failed: {message}")
            elapsed = time.monotonic() - started
            if elapsed > max(60, timeout_seconds):
                raise TimeoutError(f"videoretalk_timeout: {task_id}")
            last_percent = min(88, last_percent + 1)
            if progress:
                progress(f"云端口型任务状态：{status or 'PENDING'}（{int(elapsed)}s）", last_percent)
            time.sleep(max(2.0, poll_interval_seconds))


def upload_to_dashscope(path: Path, api_key: str, model: str) -> str:
    """与 ASR 相同的百炼临时 OSS 上传；返回 oss:// URL（服务端负责解析）。"""
    response = requests.get(
        UPLOAD_POLICY_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        params={"action": "getPolicy", "model": model},
        timeout=30,
    )
    response.raise_for_status()
    policy = response.json().get("data") or {}
    upload_dir = str(policy.get("upload_dir") or "").strip("/")
    if not upload_dir:
        raise RuntimeError(f"dashscope_upload_policy_invalid: {response.text[:500]}")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", path.name) or "videoretalk_input.mp4"
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
            "file": (safe_name, file, _mime(path)),
        }
        upload = requests.post(str(policy.get("upload_host") or ""), files=files, timeout=600)
    upload.raise_for_status()
    return f"oss://{key}"


def _mime(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
    }.get(suffix, "application/octet-stream")
