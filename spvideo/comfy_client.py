from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse, urlunparse

import requests


COMFY_URL = (
    os.environ.get("SCAIL2_COMFY_URL")
    or os.environ.get("COMFY_URL")
    or "https://8188-cpod-1sqfx2anig0i.pod.compshare.cn"
)

_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_ATTEMPTS = 4


def _safe_slug(value: str, max_length: int = 48) -> str:
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", str(value or "")).strip("_")
    return (slug or "asset")[:max_length]


def _content_addressed_name(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    suffix = path.suffix.lower() or ".bin"
    return f"{_safe_slug(path.stem)}_{digest.hexdigest()[:12]}{suffix}"


class ComfyClient:
    """Small generic ComfyUI API client used by Mode2 utilities."""

    def __init__(self, comfy_url: str = COMFY_URL):
        self.comfy = comfy_url.rstrip("/")
        self._session: requests.Session | None = None

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def upload_file(self, local_path: str | Path, subfolder: str = "") -> str:
        path = Path(local_path)
        if not path.is_file():
            raise FileNotFoundError(f"upload file does not exist: {path}")
        filename = _content_addressed_name(path)
        last_error: Exception | None = None
        for attempt in range(1, _ATTEMPTS + 1):
            try:
                with path.open("rb") as handle:
                    response = self.session.post(
                        f"{self.comfy}/upload/image",
                        files={"image": (filename, handle)},
                        data={"subfolder": subfolder, "overwrite": "true"},
                        timeout=120,
                    )
                if response.status_code in _RETRYABLE_STATUS_CODES:
                    last_error = requests.HTTPError(
                        f"ComfyUI upload returned HTTP {response.status_code}: {response.text[:400]}",
                        response=response,
                    )
                else:
                    response.raise_for_status()
                    payload = response.json()
                    uploaded = payload.get("name") if isinstance(payload, dict) else None
                    if uploaded:
                        return str(uploaded)
                    last_error = RuntimeError(f"ComfyUI upload response missing name: {response.text[:400]}")
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError, ValueError) as exc:
                last_error = exc
            if attempt < _ATTEMPTS:
                time.sleep(float(2 ** (attempt - 1)))
        raise RuntimeError(f"ComfyUI upload failed after {_ATTEMPTS} attempts: {filename}; {last_error}") from last_error

    def submit_workflow(self, workflow: dict[str, Any], *, client_id: str | None = None) -> tuple[str, str]:
        client_id = client_id or str(uuid.uuid4())
        response = self.session.post(
            f"{self.comfy}/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=60,
        )
        if not response.ok:
            raise RuntimeError(f"ComfyUI /prompt HTTP {response.status_code}: {response.text[:4000]}")
        payload = response.json()
        prompt_id = str(payload.get("prompt_id") or "")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI /prompt response missing prompt_id: {payload}")
        return prompt_id, client_id

    def run_workflow(
        self,
        workflow: dict[str, Any],
        *,
        log: Callable[[str], None] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        logger = log or (lambda _message: None)
        prompt_id, client_id = self.submit_workflow(workflow)
        history = self.wait_for_completion(prompt_id, client_id, workflow, logger)
        self.raise_for_history_error(prompt_id, history)
        return prompt_id, history

    def wait_for_completion(
        self,
        prompt_id: str,
        client_id: str,
        workflow: dict[str, Any],
        log: Callable[[str], None],
    ) -> dict[str, Any]:
        log("等待 ComfyUI 推理完成...")
        ws_url = self._websocket_url(client_id)
        try:
            import websocket

            ws = websocket.create_connection(ws_url, timeout=5)
        except Exception as exc:  # noqa: BLE001
            log(f"ComfyUI 进度通道不可用，改用轮询: {exc}")
            return self._poll_history_until_done(prompt_id, log)

        last_node: str | None = None
        last_percent = -1
        last_history_poll = 0.0
        try:
            while True:
                now = time.time()
                try:
                    message = ws.recv()
                except websocket.WebSocketTimeoutException:
                    message = None
                except Exception as exc:  # noqa: BLE001
                    log(f"ComfyUI 进度通道中断，改用轮询: {exc}")
                    return self._poll_history_until_done(prompt_id, log)

                if message and not isinstance(message, (bytes, bytearray, memoryview)):
                    try:
                        event = json.loads(message)
                    except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
                        event = {}
                    if not isinstance(event, dict):
                        event = {}
                    kind = event.get("type")
                    data = event.get("data") or {}
                    event_prompt_id = data.get("prompt_id")
                    if event_prompt_id in (None, prompt_id):
                        if kind == "executing":
                            node_id = data.get("node")
                            if node_id is None:
                                history = self.fetch_history(prompt_id)
                                if prompt_id in history:
                                    log("ComfyUI 推理完成")
                                    return history
                            elif str(node_id) != last_node:
                                last_node = str(node_id)
                                last_percent = -1
                                log(f"ComfyUI 节点: {self._node_label(workflow, last_node)}")
                        elif kind == "progress":
                            value = int(data.get("value") or 0)
                            maximum = max(1, int(data.get("max") or 1))
                            percent = max(0, min(100, int(round(value * 100 / maximum))))
                            if percent >= last_percent + 10 or percent in (0, 100):
                                last_percent = percent
                                label = self._node_label(workflow, last_node or "") if last_node else "当前节点"
                                log(f"ComfyUI 进度: {percent}% ({label} {value}/{maximum})")

                if now - last_history_poll >= 10:
                    last_history_poll = now
                    history = self.fetch_history(prompt_id)
                    if prompt_id in history:
                        log("ComfyUI 推理完成")
                        return history
        finally:
            try:
                ws.close()
            except Exception:
                pass

    def _poll_history_until_done(self, prompt_id: str, log: Callable[[str], None]) -> dict[str, Any]:
        while True:
            time.sleep(10)
            history = self.fetch_history(prompt_id)
            if prompt_id in history:
                log("ComfyUI 推理完成")
                return history

    def fetch_history(self, prompt_id: str) -> dict[str, Any]:
        try:
            response = self.session.get(f"{self.comfy}/history/{prompt_id}", timeout=60)
        except (requests.Timeout, requests.ConnectionError):
            return {}
        if response.status_code in _RETRYABLE_STATUS_CODES:
            return {}
        response.raise_for_status()
        return response.json()

    def download_output_asset(self, asset: dict[str, Any], target: Path) -> str:
        target.parent.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for attempt in range(1, _ATTEMPTS + 1):
            try:
                response = self.session.get(
                    f"{self.comfy}/view",
                    params={
                        "filename": asset["filename"],
                        "subfolder": asset.get("subfolder", ""),
                        "type": asset.get("type", "output"),
                    },
                    timeout=120,
                )
                if response.ok:
                    target.write_bytes(response.content)
                    return response.url
                last_error = requests.HTTPError(
                    f"ComfyUI /view HTTP {response.status_code}: {response.text[:1000]}",
                    response=response,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
            if attempt < _ATTEMPTS:
                time.sleep(float(2 ** (attempt - 1)))
        raise RuntimeError(f"ComfyUI output download failed: {asset.get('filename')}; {last_error}") from last_error

    @staticmethod
    def output_assets(node_output: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not isinstance(node_output, dict):
            return []
        assets: list[dict[str, Any]] = []
        for key in ("images", "videos", "gifs"):
            value = node_output.get(key)
            if isinstance(value, list):
                assets.extend(item for item in value if isinstance(item, dict))
        return assets

    @classmethod
    def first_output_asset(cls, node_output: dict[str, Any] | None) -> dict[str, Any] | None:
        assets = cls.output_assets(node_output)
        return assets[0] if assets else None

    @staticmethod
    def raise_for_history_error(prompt_id: str, history: dict[str, Any]) -> None:
        item = history.get(prompt_id) if isinstance(history, dict) else None
        if not isinstance(item, dict):
            raise RuntimeError("ComfyUI history item is empty")
        status = item.get("status") if isinstance(item.get("status"), dict) else {}
        status_str = str(status.get("status_str") or "").strip().lower()
        completed = status.get("completed")
        messages = status.get("messages")
        if status_str == "error" or completed is False:
            raise RuntimeError("ComfyUI workflow failed: " + ComfyClient.history_debug_summary(item))
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, list) and message and message[0] == "execution_error":
                    raise RuntimeError("ComfyUI workflow failed: " + ComfyClient.history_debug_summary(item))

    @staticmethod
    def history_debug_summary(history_item: dict[str, Any] | None) -> str:
        if not isinstance(history_item, dict):
            return "history item is empty"
        outputs = history_item.get("outputs") if isinstance(history_item.get("outputs"), dict) else {}
        output_keys = ", ".join(map(str, outputs.keys())) or "none"
        status = history_item.get("status") if isinstance(history_item.get("status"), dict) else {}
        parts = [f"output nodes={output_keys}"]
        status_str = str(status.get("status_str") or "").strip()
        completed = status.get("completed")
        if status_str or completed is not None:
            parts.append(f"status={status_str or 'unknown'} completed={completed}")
        messages = status.get("messages")
        if isinstance(messages, list) and messages:
            compact: list[str] = []
            for item in messages[-3:]:
                try:
                    compact.append(json.dumps(item, ensure_ascii=False)[:500])
                except TypeError:
                    compact.append(str(item)[:500])
            parts.append("messages=" + " | ".join(compact))
        return "; ".join(parts)

    def _websocket_url(self, client_id: str) -> str:
        parsed = urlparse(self.comfy)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return urlunparse((scheme, parsed.netloc, "/ws", "", f"clientId={client_id}", ""))

    @staticmethod
    def _node_label(workflow: dict[str, Any], node_id: str) -> str:
        node = workflow.get(str(node_id)) or {}
        return f"[{node_id}] {node.get('class_type') or 'unknown'}"
