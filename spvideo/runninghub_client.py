"""RunningHub OpenAPI client for Bernini workflow tests."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

import requests


RUNNINGHUB_BASE_URL = "https://www.runninghub.cn"
DEFAULT_BERNINI_WORKFLOW_ID = "2062005701358215170"
DEFAULT_BERNINI_WORKFLOW_PATH = (
    r"C:\Users\Administrator\Downloads"
    r"\提示词自动反推！！Wan2.2 Bernini Plus版节点 视频编辑工作流v2版_api.json"
)


class RunningHubClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = RUNNINGHUB_BASE_URL,
        workflow_path: str | None = None,
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("RUNNINGHUB_API_KEY")
            or os.environ.get("RH_API_KEY")
            or ""
        ).strip()
        self.base_url = base_url.rstrip("/")
        self.workflow_path = Path(
            workflow_path
            or os.environ.get("RUNNINGHUB_BERNINI_WORKFLOW")
            or DEFAULT_BERNINI_WORKFLOW_PATH
        )
        self.workflow_id = (
            os.environ.get("RUNNINGHUB_BERNINI_WORKFLOW_ID")
            or DEFAULT_BERNINI_WORKFLOW_ID
        ).strip()
        self.session = requests.Session()

    def transfer_bernini(
        self,
        video_path: str,
        ref_images: list[str],
        *,
        role_names: list[str] | None = None,
        positive_prompt: str = "",
        width: int = 512,
        height: int = 896,
        video_window: dict | None = None,
        normalize_size: bool = True,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> dict:
        def log(message: str) -> None:
            if on_progress:
                on_progress(message)

        if not self.api_key:
            raise RuntimeError("RUNNINGHUB_API_KEY is not configured")
        if not ref_images:
            raise ValueError("RunningHub Bernini needs at least one reference image")
        if len(ref_images) > 10:
            raise ValueError("RunningHub Bernini supports at most 10 reference images")
        if not self.workflow_path.is_file():
            raise FileNotFoundError(f"RunningHub workflow not found: {self.workflow_path}")

        from .ffmpeg_tools import probe_video

        meta = probe_video(video_path)
        window = {
            "force_rate": 16,
            "frame_load_cap": int(os.environ.get("RUNNINGHUB_BERNINI_MAX_FRAMES") or 49),
            "skip_first_frames": 0,
            "select_every_nth": 1,
            **(video_window or {}),
        }
        window["frame_load_cap"] = max(17, min(int(window["frame_load_cap"]), 49))
        if normalize_size:
            width, height = self._normalized_size(meta.width, meta.height, width, height)

        output_dir = Path(video_path).parent.parent.parent / "04_AI输出成片"
        output_dir.mkdir(parents=True, exist_ok=True)
        log(
            f"RunningHub Bernini: {meta.width}x{meta.height} {meta.fps:.2f}fps "
            f"{meta.duration:.2f}s; window={window['frame_load_cap']} frames; output={width}x{height}"
        )

        workflow = self._load_workflow()
        self._ensure_source_video_workflow(workflow)

        log("RunningHub upload source video...")
        video_name = self.upload_file(video_path)
        ref_names: list[str] = []
        role_names = [str(value or "").strip() for value in (role_names or [])]
        for index, image_path in enumerate(ref_images, 1):
            label = role_names[index - 1] if index - 1 < len(role_names) and role_names[index - 1] else f"role{index}"
            log(f"RunningHub upload {label} reference {index}/{len(ref_images)}...")
            ref_names.append(self.upload_file(image_path))

        self._patch_bernini_workflow(
            workflow,
            video_name=video_name,
            ref_names=ref_names,
            positive_prompt=positive_prompt,
            width=width,
            height=height,
            window=window,
        )
        workflow_debug = output_dir / (
            f"runninghub_bernini_workflow_{Path(video_path).stem}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        workflow_debug.write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"RunningHub workflow saved: {workflow_debug}")

        task_id = self.create_task(workflow)
        log(f"RunningHub task id: {task_id}")
        task_debug = output_dir / (
            f"runninghub_bernini_task_{Path(video_path).stem}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        task_debug.write_text(
            json.dumps({"taskId": task_id, "workflow_path": str(workflow_debug)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        outputs = self.wait_for_outputs(task_id, log)
        output_item = self._first_output_file(outputs)
        file_url = str(output_item.get("fileUrl") or output_item.get("url") or "").strip()
        if not file_url:
            raise RuntimeError(f"RunningHub task finished but no downloadable fileUrl was found: {outputs}")

        out_path = output_dir / f"runninghub_bernini_{Path(video_path).stem}_{task_id}.mp4"
        log(f"RunningHub download result: {file_url}")
        self.download(file_url, out_path)
        return {
            "output_path": str(out_path),
            "output_url": file_url,
            "prompt_id": task_id,
            "workflow_path": str(workflow_debug),
            "workflow_mode": "runninghub_bernini",
            "role_names": role_names,
            "ref_images": ref_images,
            "video_meta": meta.to_dict(),
            "video_window": window,
            "output_size": [width, height],
        }

    def upload_file(self, local_path: str) -> str:
        path = Path(local_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as file_handle:
            response = self.session.post(
                f"{self.base_url}/openapi/v2/media/upload/binary",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": (path.name, file_handle)},
                timeout=180,
            )
        payload = self._json_response(response)
        data = payload.get("data") if isinstance(payload, dict) else None
        file_name = str((data or {}).get("fileName") or "").strip()
        if not file_name:
            raise RuntimeError(f"RunningHub upload response missing fileName: {payload}")
        return file_name

    def create_task(self, workflow: dict) -> str:
        payload = {
            "apiKey": self.api_key,
            "workflowId": self.workflow_id,
            "workflow": json.dumps(workflow, ensure_ascii=False),
            "retainSeconds": 60,
        }
        instance_type = os.environ.get("RUNNINGHUB_INSTANCE_TYPE", "").strip()
        if instance_type:
            payload["instanceType"] = instance_type
        response = self.session.post(
            f"{self.base_url}/task/openapi/create",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        body = self._json_response(response)
        task_id = self._extract_task_id(body)
        if not task_id:
            raise RuntimeError(f"RunningHub create response missing task id: {body}")
        return task_id

    def wait_for_outputs(self, task_id: str, log: Callable[[str], None]) -> list[dict]:
        last_state = ""
        started = time.time()
        while True:
            outputs_payload = self.outputs(task_id)
            outputs = self._extract_outputs(outputs_payload)
            if outputs:
                return outputs

            status_payload = self.status(task_id)
            state = self._status_text(status_payload)
            if state and state != last_state:
                last_state = state
                log(f"RunningHub status: {state}")
            if self._is_failed_status(status_payload):
                failed_outputs = self.outputs(task_id)
                raise RuntimeError(
                    f"RunningHub task failed: status={status_payload}; outputs={failed_outputs}"
                )
            if time.time() - started > 3600:
                raise TimeoutError(f"RunningHub task timed out: {task_id}")
            time.sleep(10)

    def status(self, task_id: str) -> dict:
        response = self.session.post(
            f"{self.base_url}/task/openapi/status",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"apiKey": self.api_key, "taskId": task_id},
            timeout=30,
        )
        return self._json_response(response)

    def outputs(self, task_id: str) -> dict:
        response = self.session.post(
            f"{self.base_url}/task/openapi/outputs",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"apiKey": self.api_key, "taskId": task_id},
            timeout=30,
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"RunningHub HTTP {response.status_code}: {response.text[:1000]}") from exc
        if response.status_code >= 400:
            raise RuntimeError(f"RunningHub HTTP {response.status_code}: {payload}")
        return payload

    def download(self, url: str, target: Path) -> None:
        response = self.session.get(url, timeout=300)
        response.raise_for_status()
        target.write_bytes(response.content)

    def _load_workflow(self) -> dict:
        return json.loads(self.workflow_path.read_text(encoding="utf-8-sig"))

    @staticmethod
    def _ensure_source_video_workflow(workflow: dict) -> None:
        bernini = RunningHubClient._find_first_node(workflow, "WanAnimatePlus Bernini")
        if bernini:
            if not RunningHubClient._find_first_node(workflow, "VHS_LoadVideo"):
                raise RuntimeError("RunningHub WanAnimatePlus Bernini workflow has no VHS_LoadVideo node")
            return

        if "72" not in workflow or workflow["72"].get("class_type") != "BerniniStudio":
            raise RuntimeError("RunningHub Bernini workflow has no BerniniStudio or WanAnimatePlus Bernini node")
        if "21" not in workflow:
            workflow["21"] = {
                "inputs": {
                    "video": "VIDEO_PLACEHOLDER.mp4",
                    "force_rate": 16,
                    "force_size": "Disabled",
                    "custom_width": 0,
                    "custom_height": 0,
                    "frame_load_cap": 81,
                    "skip_first_frames": 0,
                    "select_every_nth": 1,
                    "format": "AnimateDiff",
                },
                "class_type": "VHS_LoadVideo",
                "_meta": {"title": "Load Video"},
            }
        if workflow["21"].get("class_type") != "VHS_LoadVideo":
            raise RuntimeError("RunningHub Bernini workflow has no VHS_LoadVideo source-video node")
        workflow["72"].setdefault("inputs", {})["source_video"] = ["21", 0]
        if "22" in workflow and workflow["22"].get("class_type") == "VHS_VideoCombine":
            workflow["22"].setdefault("inputs", {})["audio"] = ["21", 2]

    @staticmethod
    def _patch_bernini_workflow(
        workflow: dict,
        *,
        video_name: str,
        ref_names: list[str],
        positive_prompt: str,
        width: int,
        height: int,
        window: dict,
    ) -> None:
        RunningHubClient._ensure_source_video_workflow(workflow)
        if RunningHubClient._find_first_node(workflow, "WanAnimatePlus Bernini"):
            RunningHubClient._patch_wananimate_bernini_workflow(
                workflow,
                video_name=video_name,
                ref_names=ref_names,
                positive_prompt=positive_prompt,
                width=width,
                height=height,
                window=window,
            )
            return

        workflow["21"]["inputs"]["video"] = video_name
        workflow["21"]["inputs"]["force_rate"] = int(window["force_rate"])
        workflow["21"]["inputs"]["frame_load_cap"] = int(window["frame_load_cap"])
        workflow["21"]["inputs"]["skip_first_frames"] = int(window["skip_first_frames"])
        workflow["21"]["inputs"]["select_every_nth"] = int(window["select_every_nth"])
        workflow["24"]["inputs"]["image"] = ref_names[0]
        if len(ref_names) > 1:
            workflow["26"]["inputs"]["image"] = ref_names[1]
        else:
            workflow["26"]["inputs"]["image"] = ref_names[0]
        workflow["29"]["inputs"]["seed"] = uuid.uuid4().int % 2_147_483_647
        workflow["31"]["inputs"]["value"] = max(1, min(3, (int(window["frame_load_cap"]) - 1) // 16))
        workflow["34"]["inputs"]["value"] = int(width)
        workflow["35"]["inputs"]["value"] = int(height)
        workflow["22"]["inputs"]["frame_rate"] = int(window["force_rate"])
        workflow["22"]["inputs"]["filename_prefix"] = f"Bernini_{time.strftime('%Y%m%d_%H%M%S')}"
        if positive_prompt:
            workflow["72"]["inputs"]["prompt"] = positive_prompt
        workflow["72"]["inputs"]["task_type"] = "rv2v"

    @staticmethod
    def _patch_wananimate_bernini_workflow(
        workflow: dict,
        *,
        video_name: str,
        ref_names: list[str],
        positive_prompt: str,
        width: int,
        height: int,
        window: dict,
    ) -> None:
        video = RunningHubClient._find_first_node(workflow, "VHS_LoadVideo")
        if not video:
            raise RuntimeError("RunningHub WanAnimatePlus Bernini workflow has no VHS_LoadVideo node")
        video[1]["inputs"]["video"] = video_name
        video[1]["inputs"]["force_rate"] = int(window["force_rate"])
        video[1]["inputs"]["skip_first_frames"] = int(window["skip_first_frames"])
        video[1]["inputs"]["select_every_nth"] = int(window["select_every_nth"])

        if "123" in workflow:
            workflow["123"]["inputs"]["value"] = int(width)
        if "124" in workflow:
            workflow["124"]["inputs"]["value"] = int(height)
        if "125" in workflow:
            workflow["125"]["inputs"]["value"] = int(window["frame_load_cap"])
        if "136" in workflow and workflow["136"].get("class_type") == "VHS_VideoCombine":
            workflow["136"]["inputs"]["frame_rate"] = int(window["force_rate"])
            workflow["136"]["inputs"]["filename_prefix"] = f"RunningHub_Bernini_{time.strftime('%Y%m%d_%H%M%S')}"

        bernini_id, bernini = RunningHubClient._find_first_node(workflow, "WanAnimatePlus Bernini") or ("", {})
        bernini_inputs = bernini.setdefault("inputs", {})
        bernini_inputs["task_type"] = "rv2v"
        for key in list(bernini_inputs):
            if key.startswith("reference_image_"):
                bernini_inputs.pop(key)
        for index, name in enumerate(ref_names[:10], 1):
            node_id = "319" if index == 1 else f"rh_ref_load_{index}"
            workflow[node_id] = {
                "inputs": {"image": name},
                "class_type": "LoadImage",
                "_meta": {"title": f"RH Reference {index}"},
            }
            bernini_inputs[f"reference_image_{index}"] = [node_id, 0]

        if positive_prompt:
            if "343" in workflow and workflow["343"].get("class_type") == "BerniniPromptEnhancer":
                workflow["343"]["inputs"]["prompt"] = positive_prompt
            elif "261" in workflow:
                workflow["261"]["inputs"]["positive_prompt"] = positive_prompt

    @staticmethod
    def _find_first_node(workflow: dict, class_type: str) -> tuple[str, dict] | None:
        for node_id, node in workflow.items():
            if isinstance(node, dict) and node.get("class_type") == class_type:
                return str(node_id), node
        return None

    @staticmethod
    def _json_response(response: requests.Response) -> dict:
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"RunningHub HTTP {response.status_code}: {response.text[:1000]}") from exc
        if response.status_code >= 400:
            raise RuntimeError(f"RunningHub HTTP {response.status_code}: {payload}")
        code = payload.get("code") if isinstance(payload, dict) else None
        if code not in (None, 0, "0"):
            raise RuntimeError(f"RunningHub API error: {payload}")
        return payload

    @staticmethod
    def _extract_task_id(payload: dict) -> str:
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, str) and data.strip():
            return data.strip()
        if isinstance(data, dict):
            for key in ("taskId", "task_id", "id"):
                value = str(data.get(key) or "").strip()
                if value:
                    return value
        if isinstance(data, str):
            return data.strip()
        return str(payload.get("taskId") or payload.get("task_id") or "").strip()

    @staticmethod
    def _extract_outputs(payload: dict) -> list[dict]:
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            for key in ("outputs", "files", "result"):
                value = data.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            if data.get("fileUrl") or data.get("url"):
                return [data]
        return []

    @staticmethod
    def _first_output_file(outputs: list[dict]) -> dict:
        videos = [
            item for item in outputs
            if str(item.get("fileType") or item.get("type") or "").lower() in {"mp4", "video", "mov"}
            or str(item.get("fileUrl") or item.get("url") or "").lower().split("?")[0].endswith((".mp4", ".mov"))
        ]
        return (videos or outputs)[0] if outputs else {}

    @staticmethod
    def _status_text(payload: dict) -> str:
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, dict):
            for key in ("status", "taskStatus", "state", "statusText"):
                value = str(data.get(key) or "").strip()
                if value:
                    return value
        return str(payload.get("msg") or payload.get("message") or "").strip()

    @staticmethod
    def _is_failed_status(payload: dict) -> bool:
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, str):
            return data.strip().upper() in {
                "FAILED",
                "FAIL",
                "ERROR",
                "EXCEPTION",
                "CANCELED",
                "CANCELLED",
            }
        if isinstance(data, dict):
            for key in ("status", "taskStatus", "state", "statusText"):
                value = str(data.get(key) or "").strip().upper()
                if value in {"FAILED", "FAIL", "ERROR", "EXCEPTION", "CANCELED", "CANCELLED"}:
                    return True
            failed_reason = data.get("failedReason") or data.get("error") or data.get("exception")
            return bool(failed_reason)
        return False

    @staticmethod
    def _normalized_size(src_width: int, src_height: int, width: int, height: int) -> tuple[int, int]:
        if src_width <= 0 or src_height <= 0:
            return width, height
        if src_height >= src_width:
            return 448, 768
        return 768, 448
