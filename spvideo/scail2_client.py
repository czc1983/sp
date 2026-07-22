"""
SCAIL-2 多人转绘客户端 — 直接调 ComfyUI 8188 API，不走中间层。
"""
import copy
import hashlib
import json
import math
import re
import time
import uuid
import requests
import os
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from typing import Any, Optional, Callable


_RETRYABLE_UPLOAD_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_UPLOAD_ATTEMPTS = 4
_UPLOAD_ERROR_BODY_LIMIT = 400
_SIMPLE_VIDEO_BASE_PROMPT = (
    "a cinematic live-action scene, preserve the original action, timing, "
    "camera movement, composition and background"
)
SAM3_PERSON_TEXT_PROMPT = "a single human person, full body"
SAM3_VIDEO_DETECTION_THRESHOLD = 0.58
SAM3_REFERENCE_DETECTION_THRESHOLD = 0.58
SAM3_VIDEO_DETECT_INTERVAL = 2
SAM3_REFERENCE_DETECT_INTERVAL = 1
_INTERIOR_REPAIR_PROMPT = (
    "live-action dramatic scene, preserve the source action, timing, camera, "
    "background, body proportions and physical contact; replace only the tracked "
    "person with the reference person, including the exact face, hair, apparent "
    "age, body shape, and clothing shown in the reference image"
)
_ADVANCED_SCAIL2_NODES = {
    "CLIPVisionEncode",
    "ImageBatchMulti",
    "SamplerCustom",
    "SCAIL2ColoredMask",
    "VAEDecode",
    "WanSCAILToVideo",
}
_WANANIMATE_SCAIL2_NODES = {
    "CLIPVisionLoader",
    "EmptyImage",
    "ImageBatchMultiV2",
    "ImageCompositeMasked",
    "SAM3_TrackToMask",
    "SCAIL2ColoredMaskV2",
    "WanAnimatePlus BlockSwap",
    "WanAnimatePlus ClipVisionEncode V2",
    "WanAnimatePlus Decode",
    "WanAnimatePlus LoraSelectMulti",
    "WanAnimatePlus ModelLoader",
    "WanAnimatePlus SCAIL_2 Embeds",
    "WanAnimatePlus SamplerFromSettings",
    "WanAnimatePlus SamplerSettings",
    "WanAnimatePlus SetBlockSwap",
    "WanAnimatePlus SetLoRAs",
    "WanAnimatePlus VAELoader",
    "WanVideoTextEncodeCached",
}
_WANANIMATE_MASK_ONLY_NODES = {
    "CheckpointLoaderSimple",
    "CLIPTextEncode",
    "ImageBatchMultiV2",
    "ImageFromBatch",
    "ImageResizeKJv2",
    "LoadImage",
    "MaskBatchMulti",
    "PrimitiveBoundingBox",
    "PrimitiveString",
    "SAM3_Detect",
    "SAM3_VideoTrack",
    "SaveImage",
    "SCAIL2ColoredMaskV2",
    "SCAIL2FitVideo",
    "VHS_LoadVideo",
    "VHS_VideoCombine",
}
_VDA_WHITE_MASK_NODES = {
    "LoadVideoDepthAnythingModel",
    "VideoDepthAnythingProcess",
    "VideoDepthAnythingOutput",
    "VHS_LoadVideo",
    "VHS_VideoCombine",
}
_DWPOSE_VIDEO_NODES = {
    "VHS_LoadVideo",
    "WanVideoUniAnimateDWPoseDetector",
    "VHS_VideoCombine",
}
_BERNINI_TEST_NODES = {
    "LoadImage",
    "SCAIL2FitVideo",
    "VHS_LoadVideo",
    "VHS_VideoCombine",
    "WanAnimatePlus Bernini",
    "WanAnimatePlus BlockSwap",
    "WanAnimatePlus Decode",
    "WanAnimatePlus ModelLoader",
    "WanAnimatePlus SamplerFromSettings",
    "WanAnimatePlus SamplerSettings",
    "WanAnimatePlus SetBlockSwap",
    "WanAnimatePlus VAELoader",
    "WanVideoTextEncodeCached",
}


def _summarize_upload_response(response) -> str:
    content_type = str(response.headers.get("content-type") or "").strip()
    try:
        body = str(response.text or "").strip().replace("\r", " ").replace("\n", " ")
    except Exception:  # noqa: BLE001
        body = ""
    if len(body) > _UPLOAD_ERROR_BODY_LIMIT:
        body = body[:_UPLOAD_ERROR_BODY_LIMIT] + "..."
    parts = [f"HTTP {response.status_code}"]
    if content_type:
        parts.append(f"content-type={content_type}")
    if body:
        parts.append(f"body={body}")
    return "; ".join(parts)


def _summarize_upload_error(error: Exception) -> str:
    message = str(error).strip()
    summary = f"{type(error).__name__}: {message}" if message else type(error).__name__
    response = getattr(error, "response", None)
    if response is not None:
        response_summary = _summarize_upload_response(response)
        if response_summary:
            summary = f"{summary} ({response_summary})"
    return summary


# ComfyUI 云端地址
COMFY_URL = (
    os.environ.get("SCAIL2_COMFY_URL")
    or os.environ.get("COMFY_URL")
    or "https://8188-cpod-1sqfx2anig0i.pod.compshare.cn"
)
INPUT_DIR = "/root/ComfyUI/input"  # 服务端 input 目录


class Scail2Client:
    """SCAIL-2 远程 API 客户端（直连 ComfyUI 8188）"""

    def __init__(self, comfy_url: str = COMFY_URL):
        self.comfy = comfy_url.rstrip("/")
        self._session = None

    @property
    def session(self):
        if self._session is None:
            self._session = requests.Session()
        return self._session

    # ------------------------------------------------------------------
    # 文件上传
    # ------------------------------------------------------------------

    def upload_file(self, local_path: str, subfolder: str = "") -> str:
        """上传文件到 ComfyUI 服务端，返回文件名。

        The Comfy pod is behind a gateway which can intermittently return 502
        while accepting consecutive multipart uploads. Retry only transient gateway
        and transport failures; malformed requests and other client errors fail fast.
        """
        path = Path(local_path)
        if not path.is_file():
            raise FileNotFoundError(f"上传文件不存在: {path}")
        fname = _content_addressed_name(path)

        last_error: Exception | None = None
        for attempt in range(1, _UPLOAD_ATTEMPTS + 1):
            try:
                # Reopen for every attempt: requests consumes the previous stream.
                with path.open("rb") as file_handle:
                    response = self.session.post(
                        f"{self.comfy}/upload/image",
                        files={"image": (fname, file_handle)},
                        data={"subfolder": subfolder, "overwrite": "true"},
                        timeout=120,
                    )
                if response.status_code in _RETRYABLE_UPLOAD_STATUS_CODES:
                    last_error = requests.HTTPError(
                        f"ComfyUI 上传返回 {_summarize_upload_response(response)}: {fname}",
                        response=response,
                    )
                else:
                    try:
                        response.raise_for_status()
                    except requests.HTTPError as error:
                        raise RuntimeError(
                            f"ComfyUI 上传 {fname} 失败：{_summarize_upload_response(response)}"
                        ) from error
                    try:
                        payload = response.json()
                    except ValueError as error:
                        last_error = RuntimeError(
                            f"ComfyUI 上传响应不是 JSON：{_summarize_upload_response(response)}"
                        )
                    else:
                        uploaded_name = payload.get("name") if isinstance(payload, dict) else None
                        if uploaded_name:
                            return str(uploaded_name)
                        last_error = RuntimeError(
                            f"ComfyUI 上传响应缺少 name：{_summarize_upload_response(response)}"
                        )
            except (requests.Timeout, requests.ConnectionError) as error:
                last_error = error

            if attempt < _UPLOAD_ATTEMPTS:
                time.sleep(float(2 ** (attempt - 1)))

        raise RuntimeError(
            f"ComfyUI 上传 {fname} 连续 {_UPLOAD_ATTEMPTS} 次失败；"
            f"最后一次错误：{_summarize_upload_error(last_error) if last_error else 'unknown'}；"
            "请确认云端 Pod 与反向代理稳定后重试。"
        ) from last_error

    # ------------------------------------------------------------------
    # 核心：提交工作流 + 轮询 + 下载
    # ------------------------------------------------------------------

    def transfer(
        self,
        video_path: str,
        ref_images: list[str],
        extra_ref_images: list[str] | None = None,
        subject_extra_ref_images: list[list[str]] | None = None,
        role_names: list[str] | None = None,
        sam_text: str = SAM3_PERSON_TEXT_PROMPT,
        positive_prompt: str = "a person talking",
        width: int = 512,
        height: int = 896,
        video_window: dict | None = None,
        sampler_preset: str = "balanced",
        normalize_size: bool = True,
        save_debug_masks: bool = False,
        source_identity_points: list[list[float] | None] | None = None,
        source_identity_shapes: list[dict | None] | None = None,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """
        提交 SCAIL-2 多人换人任务。

        Args:
            video_path: 原视频本地路径
            ref_images: 目标主体主图，顺序对应驱动视频首帧从左到右的人物
            extra_ref_images: 可选全局补充参考图，只提供场景细节
            subject_extra_ref_images: 每个目标主体自己的补充图，接到 subject_N_image_1...6
            sam_text: SAM3 跟踪文本
            positive_prompt: 生成正向提示词
            width, height: 输出分辨率

        Returns:
            {"output_path": str, "output_url": str}
        """

        def log(msg):
            if on_progress:
                on_progress(msg)

        if not ref_images:
            raise ValueError("SCAIL-2 至少需要一张目标人物参考图")
        if len(ref_images) > 6:
            raise ValueError("SCAIL-2 Reference Pack 最多支持 6 个目标人物")
        role_names = [str(value or "").strip() for value in (role_names or [])]
        if role_names and len(role_names) != len(ref_images):
            raise ValueError("SCAIL-2 role_names must match ref_images")
        extra_ref_images = list(extra_ref_images or [])
        if len(extra_ref_images) > 5:
            raise ValueError("SCAIL-2 Reference Pack 最多支持 5 张补充参考图")
        if subject_extra_ref_images is None:
            subject_extra_ref_images = [[] for _ in ref_images]
        subject_extra_ref_images = [
            [str(item or "").strip() for item in images if str(item or "").strip()]
            for images in subject_extra_ref_images
        ]
        if len(subject_extra_ref_images) != len(ref_images):
            raise ValueError("SCAIL-2 subject_extra_ref_images must match ref_images")
        for images in subject_extra_ref_images:
            if len(images) > 6:
                raise ValueError("SCAIL-2 每个主体最多支持 6 张补充参考图")

        from .comfy_inventory import check_scail2_server

        log("检查 ComfyUI 节点和模型...")
        requirements = check_scail2_server(self.comfy, session=self.session)
        if requirements.get("source"):
            log(f"SCAIL-2 资源检查: {requirements.get('source')}")
        if requirements.get("warning"):
            log(f"SCAIL-2 资源检查提醒: {requirements.get('warning')}")
        if requirements["missing_nodes"] or requirements["missing_models"]:
            details = requirements["missing_nodes"] + requirements["missing_models"]
            raise RuntimeError("ComfyUI 缺少 SCAIL2 必需资源: " + ", ".join(details))
        if normalize_size and "ImageResizeKJv2" in requirements["missing_optional_nodes"]:
            normalize_size = False
            log("ImageResizeKJv2 未安装，已降级为不做参考图尺寸归一")

        from .ffmpeg_tools import probe_video

        meta = probe_video(video_path)
        video_window = self._resolve_video_window(meta, video_window)
        if normalize_size:
            width, height = self._normalized_size(meta.width, meta.height, width, height)
        output_dir = Path(video_path).parent.parent.parent / "04_AI输出成片"
        output_dir.mkdir(parents=True, exist_ok=True)
        log(
            f"视频参数: {meta.width}x{meta.height} {meta.fps:.2f}fps {meta.duration:.2f}s; "
            f"窗口={video_window['frame_load_cap']}帧; 输出={width}x{height}; 采样={sampler_preset}"
        )
        if role_names:
            log("SCAIL-2 角色顺序: " + " / ".join(role_names))

        seed_items = self._coerce_source_identity_seed_items(
            source_identity_points,
            source_identity_shapes,
            len(ref_images),
        )
        seed_points = self._coerce_source_identity_points(
            source_identity_points,
            len(ref_images),
        )
        has_shape_seed = bool(seed_items and any(item.get("shape") for item in seed_items))

        # The explicit WanAnimate path rebuilds bodies from pose silhouettes and
        # distorts age/body shape. Keep SimpleVideo as a body-preserving multi-role
        # base, then repair only interior roles that overlap two other people.
        use_track_order_preflight = len(ref_images) > 1 and not seed_items
        use_interior_role_repair = len(ref_images) > 2
        subject_appearance_hints = [
            self._reference_clothing_hint(image_path) for image_path in ref_images
        ]
        reference_collage_path = None
        if len(ref_images) > 1:
            log("SCAIL-2 多人链路: SimpleVideo 原轮廓替换")
            if use_interior_role_repair:
                repaired_names = role_names[1:-1] if role_names else []
                repaired_label = " / ".join(repaired_names) or f"{len(ref_images) - 2} 个中间人物"
                log(f"SCAIL-2 一对一补绘中间人物: {repaired_label}")

        # 1. 上传视频和参考图
        log("上传视频...")
        vid_name = self.upload_file(video_path)

        subject_names = []
        for i, img in enumerate(ref_images):
            log(f"上传目标人物主图 {i+1}/{len(ref_images)}...")
            uploaded = self.upload_file(img)
            log(f"目标人物主图 {i+1} 远程文件: {uploaded}")
            subject_names.append(uploaded)
        extra_names = []
        for i, img in enumerate(extra_ref_images):
            log(f"上传补充参考图 {i+1}/{len(extra_ref_images)}...")
            uploaded = self.upload_file(img)
            log(f"补充参考图 {i+1} 远程文件: {uploaded}")
            extra_names.append(uploaded)
        subject_extra_names: list[list[str]] = []
        for subject_index, images in enumerate(subject_extra_ref_images):
            uploaded: list[str] = []
            label = (
                role_names[subject_index]
                if subject_index < len(role_names) and role_names[subject_index]
                else f"主体{subject_index + 1}"
            )
            for image_index, img in enumerate(images):
                log(
                    f"上传 {label} 的 SCAIL-2 补充图 "
                    f"{image_index + 1}/{len(images)}..."
                )
                uploaded_name = self.upload_file(img)
                log(f"{label} 补充图 {image_index + 1} 远程文件: {uploaded_name}")
                uploaded.append(uploaded_name)
            subject_extra_names.append(uploaded)
        reference_collage_name = ""
        if reference_collage_path:
            log("上传 SCAIL-2 参考拼图...")
            reference_collage_name = self.upload_file(str(reference_collage_path))
            log(f"SCAIL-2 参考拼图远程文件: {reference_collage_name}")

        driving_object_indices = list(range(len(subject_names)))
        if use_track_order_preflight:
            log("SCAIL-2 预检测 SAM 人物轨迹与画面位置的对应关系...")
            driving_object_indices = self._resolve_driving_track_order(
                vid_name=vid_name,
                subject_count=len(subject_names),
                sam_text=sam_text,
                width=width,
                height=height,
                video_window=video_window,
                output_dir=output_dir,
                log=log,
            )
        elif seed_items:
            log("SAM3 首帧种子: 正式换人已启用人工身份点/手绘区域")

        # 2. 加载模板
        template_path = Path(__file__).parent / "scail2_template.json"
        if not template_path.exists():
            wf = self._build_template()
        else:
            with open(template_path) as f:
                wf = json.load(f)

        # 3. 替换动态参数
        wf = self._patch_workflow(
            wf,
            vid_name,
            subject_names,
            extra_names,
            sam_text,
            positive_prompt or _SIMPLE_VIDEO_BASE_PROMPT,
            width,
            height,
            video_window=video_window,
            sampler_preset=sampler_preset,
            normalize_reference=normalize_size,
            subject_extra_ref_names=subject_extra_names,
            reference_collage_name=reference_collage_name,
            driving_object_indices=driving_object_indices,
            subject_appearance_hints=subject_appearance_hints,
        )
        if seed_items:
            if has_shape_seed:
                self._add_sam3_mixed_initial_mask_seed(
                    wf,
                    seed_items,
                    width=width,
                    height=height,
                    output_dir=output_dir,
                )
            elif seed_points:
                self._add_sam3_initial_mask_seed(wf, seed_points, width=width, height=height)
        if not save_debug_masks:
            wf = self._without_debug_mask_outputs(wf)
        workflow_path = self._save_workflow_debug(
            wf,
            output_dir=output_dir,
            video_path=video_path,
            role_names=role_names,
        )
        log(f"工作流已保存: {workflow_path}")

        # 4. 提交任务
        client_id = uuid.uuid4().hex
        log("提交推理任务...")
        resp = self.session.post(
            f"{self.comfy}/prompt",
            json={"prompt": wf, "client_id": client_id},
            timeout=30,
        )
        if not resp.ok:
            raise RuntimeError(
                f"ComfyUI /prompt HTTP {resp.status_code}: {resp.text[:4000]}"
            )
        task_id = resp.json()["prompt_id"]
        log(f"任务ID: {task_id}")

        # 5. 等待完成。优先用 ComfyUI WebSocket 取节点/采样进度，失败则轮询 history。
        history = self._wait_for_completion(task_id, client_id, wf, log)

        # 6. 下载结果
        outputs = history[task_id]["outputs"]

        # 主成片固定来自节点 43；增加调试输出后不能再取 history 中的第一个媒体文件。
        video_info = self._first_video_asset(outputs.get("43") or outputs.get(43))
        if not video_info:
            for node_id, node_out in outputs.items():
                if str(node_id).startswith("mask_"):
                    continue
                video_info = self._first_video_asset(node_out)
                if video_info:
                    break

        if not video_info:
            raise RuntimeError("未找到正式输出视频")

        out_path = output_dir / f"scail2_{Path(video_path).stem}_{task_id[:8]}.mp4"
        log(f"下载结果: {video_info['filename']}")
        url = self._download_output_asset(video_info, out_path)

        mask_output_paths: dict[str, list[str]] = {}
        debug_outputs = {
            "pose": "mask_pose_video",
            "reference": "mask_reference_save",
        }
        if save_debug_masks:
            pose_preview_node = wf.get("mask_pose_video")
            if isinstance(pose_preview_node, dict):
                pose_preview_node["class_type"] = "SaveImage"
                pose_preview_node["inputs"] = {
                    "filename_prefix": "SCAIL2/debug_pose_mask",
                    "images": ["34", 0],
                }
            for label, node_id in debug_outputs.items():
                assets = self._output_assets(outputs.get(node_id) or {})
                local_paths: list[str] = []
                if label == "pose" and assets and all(self._is_image_asset(asset) for asset in assets):
                    frame_dir = output_dir / f".scail2_mask_{Path(video_path).stem}_{task_id[:8]}_frames"
                    frame_dir.mkdir(parents=True, exist_ok=True)
                    frame_paths: list[Path] = []
                    for index, asset in enumerate(assets, 1):
                        suffix = Path(str(asset.get("filename") or "")).suffix or ".png"
                        target = frame_dir / f"frame_{index:05d}{suffix}"
                        self._download_output_asset(asset, target)
                        frame_paths.append(target)
                    preview_path = output_dir / (
                        f"scail2_mask_{label}_{Path(video_path).stem}_{task_id[:8]}.mp4"
                    )
                    self._encode_frame_sequence_to_video(
                        frame_paths,
                        preview_path,
                        float(video_window["force_rate"]),
                    )
                    local_paths.append(str(preview_path))
                    log(f"SCAIL-2 {label} ???????: {preview_path}")
                else:
                    for index, asset in enumerate(assets, 1):
                        suffix = Path(str(asset.get("filename") or "")).suffix or (
                            ".mp4" if label == "pose" else ".png"
                        )
                        index_part = "" if len(assets) == 1 else f"_{index:02d}"
                        target = output_dir / (
                            f"scail2_mask_{label}_{Path(video_path).stem}_{task_id[:8]}"
                            f"{index_part}{suffix}"
                        )
                        self._download_output_asset(asset, target)
                        local_paths.append(str(target))
                    if local_paths:
                        log(f"SCAIL-2 {label} ??: {'; '.join(local_paths)}")
                if local_paths:
                    mask_output_paths[label] = local_paths
            if not mask_output_paths and len(ref_images) > 1:

                log("SCAIL-2 当前生成路线没有直接输出彩色蒙版，开始追加彩色蒙版检查任务...")
                try:
                    mask_result = self.inspect_masks(
                        video_path=video_path,
                        ref_images=ref_images,
                        role_names=role_names,
                        sam_text=sam_text,
                        width=width,
                        height=height,
                        video_window=video_window,
                        sampler_preset=sampler_preset,
                        normalize_size=False,
                        on_progress=on_progress,
                    )
                    mask_output_paths.update(mask_result.get("mask_output_paths") or {})
                except Exception as exc:  # noqa: BLE001
                    log(f"SCAIL-2 彩色蒙版检查失败: {exc}")

        return {
            "output_path": str(out_path),
            "output_url": url,
            "prompt_id": task_id,
            "workflow_path": str(workflow_path),
            "role_names": role_names,
            "ref_images": ref_images,
            "subject_extra_ref_images": subject_extra_ref_images,
            "reference_collage_path": str(reference_collage_path) if reference_collage_path else "",
            "workflow_mode": (
                "simple_video_interior_repair"
                if use_interior_role_repair
                else "simple_video"
            ),
            "video_meta": meta.to_dict(),
            "video_window": video_window,
            "output_size": [width, height],
            "sampler_preset": sampler_preset,
            "mask_output_paths": mask_output_paths,
        }

    def inspect_masks(
        self,
        video_path: str,
        ref_images: list[str],
        *,
        role_names: list[str] | None = None,
        sam_text: str = SAM3_PERSON_TEXT_PROMPT,
        width: int = 512,
        height: int = 896,
        video_window: dict | None = None,
        sampler_preset: str = "balanced",
        normalize_size: bool = True,
        strict_track_preflight: bool = False,
        source_identity_points: list[list[float] | None] | None = None,
        source_identity_shapes: list[dict | None] | None = None,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Run only remote SAM3 + SCAIL2ColoredMaskV2 and download debug masks."""

        def log(msg):
            if on_progress:
                on_progress(msg)

        if not ref_images:
            raise ValueError("SCAIL-2 mask check needs at least one reference image")
        if len(ref_images) > 6:
            raise ValueError("SCAIL-2 mask check supports at most 6 people")
        role_names = [str(value or "").strip() for value in (role_names or [])]
        if role_names and len(role_names) != len(ref_images):
            raise ValueError("SCAIL-2 role_names must match ref_images")

        missing = self._missing_wananimate_mask_only_nodes()
        if missing:
            raise RuntimeError("ComfyUI 缺少远程 SAM3 蒙版检查节点: " + ", ".join(missing))

        from .ffmpeg_tools import probe_video

        meta = probe_video(video_path)
        video_window = self._resolve_video_window(meta, video_window)
        if normalize_size:
            width, height = self._normalized_size(meta.width, meta.height, width, height)
        output_dir = Path(video_path).parent.parent.parent / "04_AI输出成片"
        output_dir.mkdir(parents=True, exist_ok=True)
        log(
            f"远程 SAM3 蒙版检查: {meta.width}x{meta.height} {meta.fps:.2f}fps "
            f"{meta.duration:.2f}s; 窗口={video_window['frame_load_cap']}帧; 输出={width}x{height}"
        )
        if role_names:
            log("角色顺序: " + " / ".join(role_names))
        warnings: list[str] = []

        reference_collage_path = self._create_reference_collage(
            ref_images,
            output_dir=output_dir,
            width=width,
            height=height,
            role_names=role_names,
        )
        log(f"参考拼图: {reference_collage_path}")

        log("上传视频...")
        vid_name = self.upload_file(video_path)

        subject_names = []
        for i, img in enumerate(ref_images):
            label = role_names[i] if i < len(role_names) and role_names[i] else f"角色{i + 1}"
            log(f"上传 {label} 参考图 {i + 1}/{len(ref_images)}...")
            uploaded = self.upload_file(img)
            log(f"{label} 远程参考图: {uploaded}")
            subject_names.append(uploaded)

        log("上传参考拼图...")
        reference_collage_name = self.upload_file(str(reference_collage_path))
        log(f"远程参考拼图: {reference_collage_name}")

        seed_items = self._coerce_source_identity_seed_items(
            source_identity_points,
            source_identity_shapes,
            len(subject_names),
        )
        seed_points = self._coerce_source_identity_points(
            source_identity_points,
            len(subject_names),
        )
        has_shape_seed = bool(seed_items and any(item.get("shape") for item in seed_items))
        driving_object_indices = list(range(len(subject_names)))
        if seed_items:
            if has_shape_seed:
                log("SAM3 首帧种子: 使用手绘范围/身份点顺序作为角色 track 顺序，跳过文本预检")
            else:
                log("SAM3 首帧种子: 使用身份点顺序作为角色 track 顺序，跳过文本预检")
        elif len(subject_names) > 1:
            log("SAM3 ?????????...")
            driving_object_indices = self._resolve_driving_track_order(
                vid_name=vid_name,
                subject_count=len(subject_names),
                sam_text=sam_text,
                width=width,
                height=height,
                video_window=video_window,
                output_dir=output_dir,
                log=log,
                strict=strict_track_preflight,
            )
        wf = self._build_wananimate_scail2_template()
        wf = self._patch_workflow(
            wf,
            vid_name,
            subject_names,
            [],
            sam_text,
            "mask inspection only",
            width,
            height,
            video_window=video_window,
            sampler_preset=sampler_preset,
            normalize_reference=normalize_size,
            subject_extra_ref_names=[[] for _ in subject_names],
            reference_collage_name=reference_collage_name,
            driving_object_indices=driving_object_indices,
            limit_frame_cap_by_sampler=False,
        )
        independent_preview_saves: list[str] = []
        if seed_items and has_shape_seed:
            independent_preview_saves = self._add_sam3_independent_identity_tracks(
                wf,
                seed_items,
                width=width,
                height=height,
                output_dir=output_dir,
            )
            log("SAM3 首帧种子: 已启用手绘范围独立 track initial_mask")
        elif seed_points:
            self._add_sam3_initial_mask_seed(wf, seed_points, width=width, height=height)
            log("SAM3 首帧种子: 已启用 SAM3_Detect + 身份点 initial_mask")
        elif source_identity_points or source_identity_shapes:
            warnings.append("source_identity_seed_invalid_or_incomplete; initial_mask_seed_disabled")
            log("SAM3 首帧种子: 人工提示不完整，已降级为文本检测跟踪")
        if not independent_preview_saves:
            for raw_index in range(len(subject_names)):
                prefix = f"preview_track_{raw_index}"
                wf[f"{prefix}_mask"] = {
                    "class_type": "SAM3_TrackToMask",
                    "inputs": {
                        "track_data": ["32", 0],
                        "object_indices": str(driving_object_indices[raw_index]),
                    },
                }
                wf[f"{prefix}_image"] = {
                    "class_type": "MaskToImage",
                    "inputs": {"mask": [f"{prefix}_mask", 0]},
                }
                wf[f"{prefix}_save"] = {
                    "class_type": "SaveImage",
                    "inputs": {
                        "filename_prefix": f"SCAIL2/debug_track_{raw_index}",
                        "images": [f"{prefix}_image", 0],
                    },
                }
        for node in wf.values():
            if node.get("class_type") == "SCAIL2ColoredMaskV2":
                # Debug mask export should show distinct palette colors, not a single flat tint.
                node["inputs"]["prefix_mask_mode"] = "Multi Image Multi Color"
        pose_preview_node = wf.get("mask_pose_video")
        if isinstance(pose_preview_node, dict):
            pose_preview_node["class_type"] = "SaveImage"
            pose_preview_node["inputs"] = {
                "filename_prefix": "SCAIL2/debug_pose_mask",
                "images": ["34", 0],
            }
        if independent_preview_saves:
            wf = self._prune_workflow_for_outputs(wf, independent_preview_saves)
        else:
            wf = self._prune_workflow_for_outputs(
                wf,
                ["mask_pose_video", "mask_reference_save"],
            )
            for subject_index in range(len(subject_names)):
                raw_index = driving_object_indices[subject_index]
                prefix = f"preview_track_{subject_index}"
                wf[f"{prefix}_mask"] = {
                    "class_type": "SAM3_TrackToMask",
                    "inputs": {
                        "track_data": ["32", 0],
                        "object_indices": str(raw_index),
                    },
                }
                wf[f"{prefix}_image"] = {
                    "class_type": "MaskToImage",
                    "inputs": {"mask": [f"{prefix}_mask", 0]},
                }
                wf[f"{prefix}_save"] = {
                    "class_type": "SaveImage",
                    "inputs": {
                        "filename_prefix": f"SCAIL2/debug_track_{raw_index}",
                        "images": [f"{prefix}_image", 0],
                    },
                }
        workflow_path = self._save_workflow_debug(
            wf,
            output_dir=output_dir,
            video_path=video_path,
            role_names=role_names,
        )
        log(f"蒙版检查工作流已保存: {workflow_path}")

        client_id = uuid.uuid4().hex
        log("提交远程 SAM3 蒙版任务...")
        resp = self.session.post(
            f"{self.comfy}/prompt",
            json={"prompt": wf, "client_id": client_id},
            timeout=30,
        )
        if not resp.ok:
            raise RuntimeError(
                f"ComfyUI /prompt HTTP {resp.status_code}: {resp.text[:4000]}"
            )
        task_id = resp.json()["prompt_id"]
        log(f"任务ID: {task_id}")

        history = self._wait_for_completion(task_id, client_id, wf, log)
        outputs = history[task_id]["outputs"]

        mask_output_paths: dict[str, list[str]] = {}
        role_track_paths: list[list[Path]] = []
        for raw_index in range(len(subject_names)):
            node_id = f"preview_track_{raw_index}_save"
            assets = self._output_assets(outputs.get(node_id) or {})
            local_paths: list[str] = []
            for index, asset in enumerate(assets, 1):
                suffix = Path(str(asset.get("filename") or "")).suffix or ".png"
                target = output_dir / (
                    f"scail2_mask_only_track_{raw_index}_{Path(video_path).stem}_{task_id[:8]}"
                    f"_{index:05d}{suffix}"
                )
                self._download_output_asset(asset, target)
                local_paths.append(str(target))
            if local_paths:
                role_track_paths.append([Path(path) for path in local_paths])
        if role_track_paths:
            propagated_count = self._propagate_sparse_masks_from_video(role_track_paths, Path(video_path))
            if propagated_count:
                warnings.append(f"sparse_sam3_track_masks_propagated:{propagated_count}")
                log(f"SAM3 track 稀疏兜底: 已按源视频运动传播 {propagated_count} 帧")
            repaired_count = self._hold_last_nonempty_masks(role_track_paths)
            if repaired_count:
                warnings.append(f"sparse_sam3_track_masks_repaired_by_hold_last:{repaired_count}")
                log(f"SAM3 track 稀疏兜底: 已用上一帧非空 mask 补齐 {repaired_count} 帧")
            preview_path = output_dir / f"scail2_mask_only_colored_{Path(video_path).stem}_{task_id[:8]}.mp4"
            self._render_colored_mask_preview_video(
                role_track_paths,
                preview_path,
                float(video_window["force_rate"]),
            )
            mask_output_paths["pose"] = [str(preview_path)]
            log(f"彩色预览视频已生成: {preview_path}")

        debug_outputs = {
            "pose": "mask_pose_video",
            "reference": "mask_reference_save",
        }
        for label, node_id in debug_outputs.items():
            if label == "pose" and "pose" in mask_output_paths:
                continue
            assets = self._output_assets(outputs.get(node_id) or {})
            local_paths: list[str] = []
            if label == "pose" and assets and all(self._is_image_asset(asset) for asset in assets):
                frame_dir = output_dir / f".scail2_mask_only_{Path(video_path).stem}_{task_id[:8]}_frames"
                frame_dir.mkdir(parents=True, exist_ok=True)
                frame_paths: list[Path] = []
                for index, asset in enumerate(assets, 1):
                    suffix = Path(str(asset.get("filename") or "")).suffix or ".png"
                    target = frame_dir / f"frame_{index:05d}{suffix}"
                    self._download_output_asset(asset, target)
                    frame_paths.append(target)
                preview_path = output_dir / (
                    f"scail2_mask_only_{label}_{Path(video_path).stem}_{task_id[:8]}.mp4"
                )
                self._encode_frame_sequence_to_video(
                    frame_paths,
                    preview_path,
                    float(video_window["force_rate"]),
                )
                local_paths.append(str(preview_path))
                log(f"{label} ???????: {preview_path}")
            else:
                for index, asset in enumerate(assets, 1):
                    suffix = Path(str(asset.get("filename") or "")).suffix or (
                        ".mp4" if label == "pose" else ".png"
                    )
                    index_part = "" if len(assets) == 1 else f"_{index:02d}"
                    target = output_dir / (
                        f"scail2_mask_only_{label}_{Path(video_path).stem}_{task_id[:8]}"
                        f"{index_part}{suffix}"
                    )
                    self._download_output_asset(asset, target)
                    local_paths.append(str(target))
                if local_paths:
                    log(f"{label} ?????: {'; '.join(local_paths)}")
            if local_paths:
                mask_output_paths[label] = local_paths

        if not mask_output_paths:

            raise RuntimeError("远程 SAM3 蒙版任务完成，但没有找到输出文件")

        return {
            "prompt_id": task_id,
            "workflow_path": str(workflow_path),
            "role_names": role_names,
            "ref_images": ref_images,
            "reference_collage_path": str(reference_collage_path),
            "workflow_mode": "remote_sam3_mask_only",
            "video_meta": meta.to_dict(),
            "video_window": video_window,
            "output_size": [width, height],
            "sampler_preset": sampler_preset,
            "mask_output_paths": mask_output_paths,
            "output_path": (mask_output_paths.get("pose") or [""])[0],
            "sam3_initial_mask_seed": bool(seed_items or seed_points),
            "sam3_initial_mask_seed_mode": "shape_independent" if independent_preview_saves else ("shape" if has_shape_seed else ("point" if seed_points else "")),
            "source_identity_points": seed_points,
            "source_identity_shapes": source_identity_shapes or [],
            "warnings": warnings,
        }

    def inspect_white_mask(
        self,
        video_path: str,
        *,
        model: str = "video_depth_anything_vitb.pth",
        width: int = 512,
        height: int = 896,
        video_window: dict | None = None,
        normalize_size: bool = True,
        input_size: int = 518,
        max_res: int = 1280,
        precision: str = "fp16",
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Run Video Depth Anything and download a grey depth + whitened mask video."""

        def log(msg):
            if on_progress:
                on_progress(msg)

        missing = self._missing_vda_white_mask_nodes()
        if missing:
            raise RuntimeError("ComfyUI missing VDA white-mask nodes: " + ", ".join(missing))

        model = str(model or "").strip() or "video_depth_anything_vitb.pth"
        if model not in {
            "video_depth_anything_vits.pth",
            "video_depth_anything_vitb.pth",
            "video_depth_anything_vitl.pth",
            "metric_video_depth_anything_vits.pth",
            "metric_video_depth_anything_vitb.pth",
            "metric_video_depth_anything_vitl.pth",
        }:
            model = "video_depth_anything_vitb.pth"
        precision = "fp32" if str(precision).lower() == "fp32" else "fp16"
        input_size = max(128, min(int(input_size or 518), 2048))
        max_res = max(256, min(int(max_res or 1280), 4096))

        from .ffmpeg_tools import probe_video

        meta = probe_video(video_path)
        video_window = self._resolve_video_window(meta, video_window)
        if normalize_size:
            width, height = self._normalized_size(meta.width, meta.height, width, height)
        else:
            width, height = 0, 0
        output_dir = Path(video_path).parent.parent.parent / "04_AI输出成片"
        output_dir.mkdir(parents=True, exist_ok=True)
        log(
            f"VDA 白膜: {meta.width}x{meta.height} {meta.fps:.2f}fps "
            f"{meta.duration:.2f}s; 窗口={video_window['frame_load_cap']}帧; "
            f"输出={'原尺寸' if not normalize_size else f'{width}x{height}'}; model={model}"
        )

        log("上传视频...")
        vid_name = self.upload_file(video_path)
        wf = {
            "1": {
                "class_type": "VHS_LoadVideo",
                "inputs": {
                    "video": vid_name,
                    "force_rate": int(video_window["force_rate"]),
                    "custom_width": int(width),
                    "custom_height": int(height),
                    "frame_load_cap": int(video_window["frame_load_cap"]),
                    "skip_first_frames": int(video_window["skip_first_frames"]),
                    "select_every_nth": int(video_window["select_every_nth"]),
                    "format": "AnimateDiff",
                },
            },
            "2": {
                "class_type": "LoadVideoDepthAnythingModel",
                "inputs": {"model": model},
            },
            "3": {
                "class_type": "VideoDepthAnythingProcess",
                "inputs": {
                    "vda_model": ["2", 0],
                    "images": ["1", 0],
                    "input_size": int(input_size),
                    "max_res": int(max_res),
                    "precision": precision,
                },
            },
            "4": {
                "class_type": "VideoDepthAnythingOutput",
                "inputs": {
                    "depths": ["3", 0],
                    "colormap": "gray",
                },
            },
            "5": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["4", 0],
                    "frame_rate": float(video_window["force_rate"]),
                    "loop_count": 0,
                    "filename_prefix": "SCAIL2/vda_depth",
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 18,
                    "save_metadata": True,
                    "trim_to_audio": False,
                    "pingpong": False,
                    "save_output": True,
                },
            },
        }
        workflow_path = self._save_workflow_debug(
            wf,
            output_dir=output_dir,
            video_path=video_path,
            role_names=["vda_white_mask"],
        )
        log(f"VDA 工作流已保存: {workflow_path}")

        client_id = uuid.uuid4().hex
        log("提交远程 VDA 白膜任务...")
        resp = self.session.post(
            f"{self.comfy}/prompt",
            json={"prompt": wf, "client_id": client_id},
            timeout=180,
        )
        if not resp.ok:
            raise RuntimeError(f"ComfyUI /prompt HTTP {resp.status_code}: {resp.text[:4000]}")
        task_id = resp.json()["prompt_id"]
        log(f"任务ID: {task_id}")

        history = self._wait_for_completion(task_id, client_id, wf, log)
        outputs = history[task_id]["outputs"]
        asset = self._first_video_asset(outputs.get("5") or {}) or self._first_output_asset(outputs.get("5") or {})
        if not asset:
            raise RuntimeError("VDA 白膜任务完成，但没有找到输出视频")

        depth_path = output_dir / f"scail2_vda_depth_{Path(video_path).stem}_{task_id[:8]}.mp4"
        self._download_output_asset(asset, depth_path)
        log(f"VDA 灰度深度视频: {depth_path}")

        white_path = output_dir / f"scail2_vda_white_{Path(video_path).stem}_{task_id[:8]}.mp4"
        warnings: list[str] = []
        try:
            self._make_white_depth_video(depth_path, white_path)
            log(f"VDA 漂白白膜视频: {white_path}")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"white_postprocess_failed: {exc}")
            white_path = depth_path
            log(f"VDA 白膜后处理失败，改用灰度深度视频: {exc}")

        mask_output_paths = {
            "depth": [str(depth_path)],
            "white": [str(white_path)],
        }
        return {
            "prompt_id": task_id,
            "workflow_path": str(workflow_path),
            "workflow_mode": "remote_vda_white_mask",
            "video_meta": meta.to_dict(),
            "video_window": video_window,
            "output_size": [width or meta.width, height or meta.height],
            "model": model,
            "input_size": input_size,
            "max_res": max_res,
            "precision": precision,
            "mask_output_paths": mask_output_paths,
            "depth_path": str(depth_path),
            "white_mask_path": str(white_path),
            "output_path": str(white_path),
            "warnings": warnings,
        }

    def inspect_identity_gray_relief(
        self,
        video_path: str,
        *,
        width: int = 512,
        height: int = 896,
        video_window: dict | None = None,
        normalize_size: bool = True,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Convert source video to a low-contamination grayscale relief control video."""

        def log(msg):
            if on_progress:
                on_progress(msg)

        from .ffmpeg_tools import probe_video

        meta = probe_video(video_path)
        video_window = self._resolve_video_window(meta, video_window)
        if normalize_size:
            width, height = self._normalized_size(meta.width, meta.height, width, height)
        else:
            width, height = meta.width, meta.height
        output_dir = Path(video_path).parent.parent.parent / "04_AI输出成片"
        output_dir.mkdir(parents=True, exist_ok=True)
        task_id = uuid.uuid4().hex[:8]
        out_path = output_dir / f"identity_gray_relief_{Path(video_path).stem}_{task_id}.mp4"
        log(
            f"光影灰白控制: {meta.width}x{meta.height} {meta.fps:.2f}fps "
            f"{meta.duration:.2f}s; 窗口={video_window['frame_load_cap']}帧; 输出={width}x{height}"
        )
        self._make_identity_gray_relief_video(
            Path(video_path),
            out_path,
            width=int(width),
            height=int(height),
            video_window=video_window,
        )
        log(f"光影灰白控制视频: {out_path}")
        return {
            "prompt_id": task_id,
            "workflow_mode": "local_identity_gray_relief",
            "video_meta": meta.to_dict(),
            "video_window": video_window,
            "output_size": [width, height],
            "mask_output_paths": {"identity_gray_relief": [str(out_path)]},
            "identity_gray_relief_path": str(out_path),
            "output_path": str(out_path),
            "warnings": [],
        }

    def inspect_pose_video(
        self,
        video_path: str,
        *,
        width: int = 720,
        height: int = 1280,
        video_window: dict | None = None,
        normalize_size: bool = True,
        score_threshold: float = 0.18,
        stick_width: int = 7,
        body_keypoint_size: int = 6,
        hand_keypoint_size: int = 5,
        draw_feet: bool = True,
        draw_hands: bool = True,
        draw_head: bool = True,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Run remote DWPose/OpenPose-style detection and download a pose video."""

        def log(msg):
            if on_progress:
                on_progress(msg)

        missing = self._missing_dwpose_video_nodes()
        if missing:
            raise RuntimeError("ComfyUI missing DWPose video nodes: " + ", ".join(missing))

        from .ffmpeg_tools import probe_video

        meta = probe_video(video_path)
        video_window = self._resolve_video_window(meta, video_window)
        if normalize_size:
            width, height = self._normalized_size(meta.width, meta.height, width, height)
        else:
            width, height = 0, 0
        output_dir = Path(video_path).parent.parent.parent / "04_AI输出成片"
        output_dir.mkdir(parents=True, exist_ok=True)
        log(
            f"DWPose 骨架: {meta.width}x{meta.height} {meta.fps:.2f}fps "
            f"窗口={video_window['frame_load_cap']}帧; 输出={'原尺寸' if not normalize_size else f'{width}x{height}'}; "
            f"threshold={score_threshold:.2f}, stick={stick_width}"
        )

        log("上传视频用于 DWPose...")
        vid_name = self.upload_file(video_path)
        wf = {
            "1": {
                "class_type": "VHS_LoadVideo",
                "inputs": {
                    "video": vid_name,
                    "force_rate": int(video_window["force_rate"]),
                    "custom_width": int(width),
                    "custom_height": int(height),
                    "frame_load_cap": int(video_window["frame_load_cap"]),
                    "skip_first_frames": int(video_window["skip_first_frames"]),
                    "select_every_nth": int(video_window["select_every_nth"]),
                    "format": "AnimateDiff",
                },
            },
            "2": {
                "class_type": "WanVideoUniAnimateDWPoseDetector",
                "inputs": {
                    "pose_images": ["1", 0],
                    "score_threshold": max(0.0, min(float(score_threshold), 1.0)),
                    "stick_width": max(1, min(int(stick_width or 4), 24)),
                    "draw_body": True,
                    "body_keypoint_size": max(0, min(int(body_keypoint_size or 4), 24)),
                    "draw_feet": bool(draw_feet),
                    "draw_hands": bool(draw_hands),
                    "hand_keypoint_size": max(0, min(int(hand_keypoint_size or 4), 24)),
                    "colorspace": "RGB",
                    "handle_not_detected": "repeat",
                    "draw_head": bool(draw_head),
                },
            },
            "3": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["2", 0],
                    "frame_rate": float(video_window["force_rate"]),
                    "loop_count": 0,
                    "filename_prefix": "SCAIL2/dwpose",
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 18,
                    "save_metadata": True,
                    "trim_to_audio": False,
                    "pingpong": False,
                    "save_output": True,
                },
            },
        }
        workflow_path = self._save_workflow_debug(
            wf,
            output_dir=output_dir,
            video_path=video_path,
            role_names=["dwpose_body_pose"],
        )
        log(f"DWPose 工作流已保存: {workflow_path}")

        client_id = uuid.uuid4().hex
        log("提交远程 DWPose 骨架任务...")
        resp = self.session.post(
            f"{self.comfy}/prompt",
            json={"prompt": wf, "client_id": client_id},
            timeout=180,
        )
        if not resp.ok:
            raise RuntimeError(f"ComfyUI /prompt HTTP {resp.status_code}: {resp.text[:4000]}")
        task_id = resp.json()["prompt_id"]
        log(f"DWPose 任务ID: {task_id}")

        history = self._wait_for_completion(task_id, client_id, wf, log)
        outputs = history[task_id]["outputs"]
        asset = self._first_video_asset(outputs.get("3") or {}) or self._first_output_asset(outputs.get("3") or {})
        if not asset:
            raise RuntimeError(
                "DWPose 任务完成，但没有找到输出视频；"
                + self._history_debug_summary(history.get(task_id) or {})
            )

        pose_path = output_dir / f"scail2_dwpose_{Path(video_path).stem}_{task_id[:8]}.mp4"
        self._download_output_asset(asset, pose_path)
        log(f"DWPose 骨架视频: {pose_path}")

        return {
            "prompt_id": task_id,
            "workflow_path": str(workflow_path),
            "workflow_mode": "remote_dwpose_pose_video",
            "video_meta": meta.to_dict(),
            "video_window": video_window,
            "output_size": [width or meta.width, height or meta.height],
            "pose_path": str(pose_path),
            "output_path": str(pose_path),
            "mask_output_paths": {"body_pose": [str(pose_path)]},
            "warnings": [],
        }

    def inspect_capsule_control(
        self,
        video_path: str,
        *,
        video_window: dict | None = None,
        normalize_size: bool = True,
        pose_backend: str = "server_dwpose",
        pose_capsule_strength: str = "strong",
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Build a pure control video from pose only, without reusing source image texture."""

        def log(msg):
            if on_progress:
                on_progress(msg)

        from .ffmpeg_tools import probe_video

        meta = probe_video(video_path)
        video_window = self._resolve_video_window(meta, video_window)
        output_dir = Path(video_path).parent.parent.parent / "04_AI输出成片"
        output_dir.mkdir(parents=True, exist_ok=True)

        log("生成低模胶囊人控制：先跑 DWPose，再重绘抽象人形，不使用原视频纹理...")
        pose_result = self.inspect_pose_video(
            video_path=video_path,
            width=720,
            height=1280,
            video_window=video_window,
            normalize_size=normalize_size,
            score_threshold=0.16,
            stick_width=6,
            body_keypoint_size=5,
            hand_keypoint_size=0,
            draw_feet=False,
            draw_hands=False,
            draw_head=True,
            on_progress=on_progress,
        )
        pose_path = Path(str(pose_result.get("pose_path") or pose_result.get("output_path") or ""))
        if not pose_path.exists():
            raise RuntimeError("capsule_control_pose_missing")

        task_id = str(pose_result.get("prompt_id") or uuid.uuid4().hex)
        capsule_path = output_dir / f"scail2_capsule_control_{Path(video_path).stem}_{task_id[:8]}.mp4"
        self._make_capsule_control_video(
            pose_path=pose_path,
            output_path=capsule_path,
            on_progress=on_progress,
        )
        log(f"低模胶囊人控制视频: {capsule_path}")
        result = dict(pose_result)
        result.update({
            "prompt_id": task_id,
            "workflow_mode": "remote_dwpose_capsule_control",
            "capsule_control_path": str(capsule_path),
            "output_path": str(capsule_path),
            "mask_output_paths": {
                "body_pose": [str(pose_path)],
                "capsule_control": [str(capsule_path)],
            },
            "warnings": list(result.get("warnings") or []),
        })
        return result

    def inspect_expression_mask(
        self,
        video_path: str,
        *,
        model: str = "video_depth_anything_vitb.pth",
        width: int = 512,
        height: int = 896,
        video_window: dict | None = None,
        normalize_size: bool = True,
        input_size: int = 518,
        max_res: int = 1280,
        precision: str = "fp16",
        max_faces: int = 6,
        include_mouth: bool = True,
        color_faces: bool = True,
        include_eyes: bool = True,
        include_brows: bool = True,
        include_head_pose: bool = True,
        include_face_outline: bool = True,
        include_soft_face_relief: bool = False,
        face_relief: bool | None = None,
        strong_depth_relief: bool = False,
        safe_mode: bool = True,
        body_color_mode: str = "none",
        include_body_pose: bool = False,
        pose_backend: str = "server_dwpose",
        pose_render_style: str = "capsule",
        pose_capsule_strength: str = "strong",
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Run VDA white mask, then overlay face-expression guidance."""

        def log(msg):
            if on_progress:
                on_progress(msg)

        if face_relief is not None:
            include_soft_face_relief = bool(include_soft_face_relief or face_relief)

        result = self.inspect_white_mask(
            video_path=video_path,
            model=model,
            width=width,
            height=height,
            video_window=video_window,
            normalize_size=normalize_size,
            input_size=input_size,
            max_res=max_res,
            precision=precision,
            on_progress=on_progress,
        )
        white_path = Path(str(result.get("white_mask_path") or result.get("output_path") or ""))
        if not white_path.exists():
            raise RuntimeError("expression_mask_base_white_missing")

        task_id = str(result.get("prompt_id") or uuid.uuid4().hex)
        output_dir = white_path.parent
        base_path = white_path
        strong_depth_relief_enabled = False
        if strong_depth_relief:
            depth_path = Path(str(result.get("depth_path") or ((result.get("mask_output_paths") or {}).get("depth") or [""])[0] or ""))
            if not depth_path.exists():
                warnings = list(result.get("warnings") or [])
                warnings.append("strong_depth_relief_missing_depth_path")
                result["warnings"] = warnings
                log("强2.5D白模缺少 depth_path，改用普通白膜底图")
            else:
                strong_path = output_dir / f"scail2_vda_strong_clay_{Path(video_path).stem}_{task_id[:8]}.mp4"
                try:
                    self._make_strong_depth_relief_video(depth_path, strong_path)
                    base_path = strong_path
                    strong_depth_relief_enabled = True
                    mask_output_paths = dict(result.get("mask_output_paths") or {})
                    mask_output_paths["strong_clay"] = [str(strong_path)]
                    mask_output_paths["normal_lit_clay"] = [str(strong_path)]
                    result["mask_output_paths"] = mask_output_paths
                    result["normal_lit_clay_path"] = str(strong_path)
                    result["strong_clay_path"] = str(strong_path)
                    log(f"强2.5D白模视频: {strong_path}")
                except Exception as exc:  # noqa: BLE001
                    warnings = list(result.get("warnings") or [])
                    warnings.append(f"strong_depth_relief_failed: {exc}")
                    result["warnings"] = warnings
                    log(f"强2.5D白模生成失败，改用普通白膜底图: {exc}")
        body_color_mode = str(body_color_mode or "none").strip().lower() or "none"
        color_body_enabled = body_color_mode not in {"none", "off", "false", "0"}
        warnings = list(result.get("warnings") or [])
        body_pose_path: Path | None = None
        pose_render_style = str(pose_render_style or "capsule").strip().lower() or "capsule"
        pose_render_style = {
            "clean_white": "clean",
            "white": "clean",
            "lowpoly": "capsule",
            "low_poly": "capsule",
            "low_model": "capsule",
            "mannequin": "capsule",
            "capsule_mannequin": "capsule",
            "debug_skeleton": "debug_lines",
            "skeleton": "debug_lines",
            "lines": "debug_lines",
            "debug": "debug_lines",
            "debug_skeleton": "debug_lines",
            "none": "clean",
            "off": "clean",
            "false": "clean",
            "0": "clean",
        }.get(pose_render_style, pose_render_style)
        if pose_render_style not in {"capsule", "debug_lines", "clean"}:
            warnings.append(f"unknown_pose_render_style: {pose_render_style}")
            pose_render_style = "capsule"
        pose_capsule_strength = str(pose_capsule_strength or "strong").strip().lower() or "strong"
        pose_capsule_strength = {
            "weak": "light",
            "soft": "light",
            "low": "light",
            "medium": "normal",
            "standard": "normal",
            "default": "normal",
            "high": "strong",
            "heavy": "strong",
            "max": "strong",
        }.get(pose_capsule_strength, pose_capsule_strength)
        if pose_capsule_strength not in {"light", "normal", "strong"}:
            warnings.append(f"unknown_pose_capsule_strength: {pose_capsule_strength}")
            pose_capsule_strength = "strong"
        pose_backend = str(pose_backend or "server_dwpose").strip().lower()
        pose_backend = {
            "remote_dwpose": "server_dwpose",
            "dwpose": "server_dwpose",
            "server": "server_dwpose",
            "mediapipe": "local_mediapipe",
            "local": "local_mediapipe",
        }.get(pose_backend, pose_backend)
        include_body_pose_effective = bool(include_body_pose and pose_render_style != "clean")
        if include_body_pose_effective:
            if pose_backend in {"auto", "server_dwpose"}:
                try:
                    detail_pose = pose_render_style == "debug_lines"
                    log(
                        "生成人体姿态: 服务器 DWPose "
                        + ("调试彩色骨架..." if detail_pose else "低模胶囊底图...")
                    )
                    pose_width = max(720, int(width or 0))
                    pose_height = max(1280, int(height or 0))
                    pose_result = self.inspect_pose_video(
                        video_path=video_path,
                        width=pose_width,
                        height=pose_height,
                        video_window=video_window,
                        normalize_size=normalize_size,
                        score_threshold=0.18 if detail_pose else 0.16,
                        stick_width=7 if detail_pose else 6,
                        body_keypoint_size=6 if detail_pose else 5,
                        hand_keypoint_size=5 if detail_pose else 0,
                        draw_feet=detail_pose,
                        draw_hands=detail_pose,
                        draw_head=detail_pose,
                        on_progress=on_progress,
                    )
                    pose_path = Path(str(pose_result.get("pose_path") or pose_result.get("output_path") or ""))
                    if pose_path.exists():
                        body_pose_path = pose_path
                        mask_output_paths = dict(result.get("mask_output_paths") or {})
                        mask_output_paths["body_pose"] = [str(body_pose_path)]
                        result["mask_output_paths"] = mask_output_paths
                        result["body_pose_path"] = str(body_pose_path)
                        result["pose_render_style"] = pose_render_style
                        log(f"人体骨架视频: {body_pose_path}")
                    else:
                        warnings.append("body_pose_missing_after_dwpose")
                        log("人体骨架未找到输出，继续生成普通表情白膜")
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"body_pose_remote_dwpose_failed: {exc}")
                    log(f"人体骨架生成失败，继续生成普通表情白膜: {exc}")
            elif pose_backend == "local_mediapipe":
                warnings.append("local_mediapipe_not_implemented_or_not_installed")
                log("本机 MediaPipe 骨架暂未启用/未安装，继续生成普通表情白膜")
            else:
                warnings.append(f"unknown_pose_backend: {pose_backend}")
                log(f"未知骨架来源 {pose_backend}，继续生成普通表情白膜")

        if strong_depth_relief_enabled:
            output_prefix = "scail2_vda_strong_clay_pose_expression" if body_pose_path else "scail2_vda_strong_clay_expression"
        elif body_pose_path:
            pose_prefix = "capsule" if pose_render_style == "capsule" else "pose"
            output_prefix = f"scail2_vda_{pose_prefix}_color_body" if color_body_enabled else f"scail2_vda_{pose_prefix}_expression"
        else:
            output_prefix = "scail2_vda_color_body" if color_body_enabled else "scail2_vda_expression"
        expression_path = output_dir / f"{output_prefix}_{Path(video_path).stem}_{task_id[:8]}.mp4"
        try:
            if color_body_enabled:
                log("生成红绿白模: 叠加低污染表情线和红绿主体色...")
            else:
                guidance_parts = []
                if include_face_outline:
                    guidance_parts.append("脸部轮廓")
                if include_eyes or include_brows:
                    guidance_parts.append("眉眼")
                if include_mouth:
                    guidance_parts.append("口型")
                if include_head_pose:
                    guidance_parts.append("头向")
                if include_soft_face_relief:
                    guidance_parts.append("五官凹凸阴影")
                if body_pose_path and pose_render_style == "capsule":
                    guidance_parts.append("低模胶囊人形")
                elif body_pose_path and pose_render_style == "debug_lines":
                    guidance_parts.append("调试彩色骨架")
                guidance_text = "、".join(guidance_parts) if guidance_parts else "无额外线稿"
                log(f"生成表情白膜: 叠加低污染{guidance_text}线索...")
            stats = self._make_face_expression_guidance_video(
                video_path=Path(video_path),
                base_path=base_path,
                output_path=expression_path,
                body_pose_path=body_pose_path,
                max_faces=max_faces,
                include_mouth=include_mouth,
                color_faces=color_faces,
                include_eyes=include_eyes,
                include_brows=include_brows,
                include_head_pose=include_head_pose,
                include_face_outline=include_face_outline,
                include_soft_face_relief=include_soft_face_relief,
                safe_mode=safe_mode,
                body_color_mode=body_color_mode,
                include_body_pose=bool(body_pose_path),
                pose_render_style=pose_render_style,
                pose_capsule_strength=pose_capsule_strength,
                on_progress=log,
            )
            log(
                f"表情白膜视频: {expression_path} "
                f"(frames={stats.get('frames', 0)}, face_hits={stats.get('face_hits', 0)})"
            )
            mask_output_paths = dict(result.get("mask_output_paths") or {})
            mask_output_paths["expression"] = [str(expression_path)]
            result["mask_output_paths"] = mask_output_paths
            result["expression_mask_path"] = str(expression_path)
            result["output_path"] = str(expression_path)
            result["workflow_mode"] = "remote_vda_expression_mask"
            result["expression_stats"] = stats
            if isinstance(stats, dict):
                stats["strong_depth_relief"] = "enabled" if strong_depth_relief_enabled else "disabled"
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"expression_postprocess_failed: {exc}")
            log(f"表情白膜后处理失败，改用普通白膜视频: {exc}")
        result["warnings"] = warnings
        return result

    def transfer_wananimate_masked_test(
        self,
        video_path: str,
        ref_images: list[str],
        *,
        role_names: list[str] | None = None,
        sam_text: str = SAM3_PERSON_TEXT_PROMPT,
        positive_prompt: str = "a person talking",
        width: int = 512,
        height: int = 896,
        video_window: dict | None = None,
        sampler_preset: str = "balanced",
        normalize_size: bool = True,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Experimental SCAIL-2 path using WanAnimatePlus and explicit colored masks."""

        def log(msg):
            if on_progress:
                on_progress(msg)

        if not ref_images:
            raise ValueError("SCAIL-2 masked test needs at least one reference image")
        if len(ref_images) > 6:
            raise ValueError("SCAIL-2 masked test supports at most 6 people")
        role_names = [str(value or "").strip() for value in (role_names or [])]
        if role_names and len(role_names) != len(ref_images):
            raise ValueError("SCAIL-2 role_names must match ref_images")

        missing = self._missing_wananimate_nodes()
        if missing:
            raise RuntimeError("ComfyUI 缺少 SCAIL2 masked test 节点: " + ", ".join(missing))

        from .ffmpeg_tools import probe_video

        meta = probe_video(video_path)
        video_window = self._resolve_video_window(meta, video_window)
        if normalize_size:
            width, height = self._normalized_size(meta.width, meta.height, width, height)
        output_dir = Path(video_path).parent.parent.parent / "04_AI输出成片"
        output_dir.mkdir(parents=True, exist_ok=True)
        log(
            f"SCAIL-2 masked test: {meta.width}x{meta.height} {meta.fps:.2f}fps "
            f"{meta.duration:.2f}s; window={video_window['frame_load_cap']} frames; output={width}x{height}"
        )
        if role_names:
            log("SCAIL-2 masked role order: " + " / ".join(role_names))

        reference_collage_path = self._create_reference_collage(
            ref_images,
            output_dir=output_dir,
            width=width,
            height=height,
            role_names=role_names,
        )
        log(f"SCAIL-2 masked reference collage: {reference_collage_path}")

        log("上传视频...")
        vid_name = self.upload_file(video_path)

        subject_names: list[str] = []
        for index, image_path in enumerate(ref_images, 1):
            label = role_names[index - 1] if index - 1 < len(role_names) and role_names[index - 1] else f"role{index}"
            log(f"上传 {label} masked test 主图 {index}/{len(ref_images)}...")
            uploaded = self.upload_file(image_path)
            log(f"{label} masked test 远程文件: {uploaded}")
            subject_names.append(uploaded)

        log("上传 masked test 参考拼图...")
        reference_collage_name = self.upload_file(str(reference_collage_path))
        log(f"masked test 参考拼图远程文件: {reference_collage_name}")

        wf = self._build_wananimate_scail2_template()
        wf = self._patch_workflow(
            wf,
            vid_name,
            subject_names,
            [],
            sam_text,
            positive_prompt,
            width,
            height,
            video_window=video_window,
            sampler_preset=sampler_preset,
            normalize_reference=normalize_size,
            subject_extra_ref_names=[[] for _ in subject_names],
            reference_collage_name=reference_collage_name,
        )
        workflow_path = self._save_workflow_debug(
            wf,
            output_dir=output_dir,
            video_path=video_path,
            role_names=role_names,
        )
        log(f"SCAIL-2 masked test 工作流已保存: {workflow_path}")

        client_id = uuid.uuid4().hex
        log("提交 SCAIL-2 masked test 推理任务...")
        resp = self.session.post(
            f"{self.comfy}/prompt",
            json={"prompt": wf, "client_id": client_id},
            timeout=30,
        )
        if not resp.ok:
            raise RuntimeError(f"ComfyUI /prompt HTTP {resp.status_code}: {resp.text[:4000]}")
        task_id = resp.json()["prompt_id"]
        log(f"任务ID: {task_id}")

        history = self._wait_for_completion(task_id, client_id, wf, log)
        outputs = history[task_id]["outputs"]
        video_info = self._first_video_asset(outputs.get("43") or outputs.get(43))
        if not video_info:
            raise RuntimeError("SCAIL-2 masked test 未找到正式输出视频")

        out_path = output_dir / f"scail2_masked_{Path(video_path).stem}_{task_id[:8]}.mp4"
        log(f"下载 masked test 结果: {video_info['filename']}")
        url = self._download_output_asset(video_info, out_path)

        mask_output_paths: dict[str, list[str]] = {}
        for label, node_id in {"pose": "mask_pose_video", "reference": "mask_reference_save"}.items():
            assets = self._output_assets(outputs.get(node_id) or {})
            local_paths: list[str] = []
            for index, asset in enumerate(assets, 1):
                suffix = Path(str(asset.get("filename") or "")).suffix or (".mp4" if label == "pose" else ".png")
                index_part = "" if len(assets) == 1 else f"_{index:02d}"
                target = output_dir / (
                    f"scail2_masked_{label}_{Path(video_path).stem}_{task_id[:8]}{index_part}{suffix}"
                )
                self._download_output_asset(asset, target)
                local_paths.append(str(target))
            if local_paths:
                mask_output_paths[label] = local_paths
                log(f"SCAIL-2 masked {label} mask: {'; '.join(local_paths)}")

        return {
            "output_path": str(out_path),
            "output_url": url,
            "prompt_id": task_id,
            "workflow_path": str(workflow_path),
            "role_names": role_names,
            "ref_images": ref_images,
            "reference_collage_path": str(reference_collage_path),
            "workflow_mode": "wananimate_masked_test",
            "video_meta": meta.to_dict(),
            "video_window": video_window,
            "output_size": [width, height],
            "sampler_preset": sampler_preset,
            "mask_output_paths": mask_output_paths,
        }

    def transfer_colored_mask_test(
        self,
        video_path: str,
        ref_images: list[str],
        *,
        role_names: list[str] | None = None,
        source_positions: list[float | None] | None = None,
        sam_text: str = SAM3_PERSON_TEXT_PROMPT,
        positive_prompt: str = "a person talking",
        width: int = 512,
        height: int = 896,
        video_window: dict | None = None,
        sampler_preset: str = "balanced",
        normalize_size: bool = True,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Experimental SCAIL-2 path that feeds explicit colored masks into generation."""

        def log(msg):
            if on_progress:
                on_progress(msg)

        if not ref_images:
            raise ValueError("SCAIL-2 colored mask test needs at least one reference image")
        if len(ref_images) > 6:
            raise ValueError("SCAIL-2 colored mask test supports at most 6 people")
        role_names = [str(value or "").strip() for value in (role_names or [])]
        if role_names and len(role_names) != len(ref_images):
            raise ValueError("SCAIL-2 role_names must match ref_images")

        missing = self._missing_advanced_nodes()
        if missing:
            raise RuntimeError("ComfyUI 缺少 SCAIL2 colored mask test 节点: " + ", ".join(missing))

        from .ffmpeg_tools import probe_video

        meta = probe_video(video_path)
        video_window = self._resolve_video_window(meta, video_window)
        if normalize_size:
            width, height = self._normalized_size(meta.width, meta.height, width, height)
        output_dir = Path(video_path).parent.parent.parent / "04_AI输出成片"
        output_dir.mkdir(parents=True, exist_ok=True)
        log(
            f"SCAIL-2 colored mask test: {meta.width}x{meta.height} {meta.fps:.2f}fps "
            f"{meta.duration:.2f}s; window={video_window['frame_load_cap']} frames; output={width}x{height}"
        )
        if role_names:
            log("SCAIL-2 colored role order: " + " / ".join(role_names))

        log("SCAIL-2 colored preflight: 先生成源视频蓝红位置蒙版，用它重建同位置参考图...")
        preflight_mask_result = self.inspect_masks(
            video_path=video_path,
            ref_images=ref_images,
            role_names=role_names,
            sam_text=sam_text,
            width=width,
            height=height,
            video_window=video_window,
            sampler_preset=sampler_preset,
            normalize_size=False,
            on_progress=on_progress,
        )
        preflight_pose_mask = (
            (preflight_mask_result.get("mask_output_paths") or {}).get("pose") or [""]
        )[0]
        sampler_settings = self._sampler_settings(sampler_preset)
        pose_segments = self._analyze_pose_mask_segments(
            preflight_pose_mask,
            output_dir=output_dir,
            expected_subjects=len(ref_images),
            frame_rate=video_window["force_rate"],
            max_segment_frames=int(sampler_settings["chunk_frames"]),
            log=log,
        ) if preflight_pose_mask and Path(preflight_pose_mask).exists() else []
        if not pose_segments:
            fallback_reference = self._create_source_position_reference_collage(
                ref_images,
                output_dir=output_dir,
                width=width,
                height=height,
                role_names=role_names,
                source_positions=source_positions,
            )
            pose_segments = [{
                "start_frame": 0,
                "frame_count": int(video_window["frame_load_cap"]),
                "reference_frame": str(fallback_reference),
            }]

        log("上传视频...")
        vid_name = self.upload_file(video_path)

        subject_names: list[str] = []
        for index, image_path in enumerate(ref_images, 1):
            label = role_names[index - 1] if index - 1 < len(role_names) and role_names[index - 1] else f"role{index}"
            log(f"上传 {label} colored test 主图 {index}/{len(ref_images)}...")
            uploaded = self.upload_file(image_path)
            log(f"{label} colored test 远程文件: {uploaded}")
            subject_names.append(uploaded)

        def run_colored_segment(segment: dict, segment_index: int) -> dict:
            segment_start = int(segment.get("start_frame") or 0)
            segment_count = max(1, int(segment.get("frame_count") or video_window["frame_load_cap"]))
            source_reference = str(segment.get("reference_frame") or "")
            if source_reference and Path(source_reference).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                background_frame_path = self._extract_source_reference_frame(
                    video_path,
                    output_dir=output_dir,
                    frame_index=segment_start,
                    video_window=video_window,
                    role_names=role_names,
                    segment_index=segment_index,
                )
                reference_collage_path = self._create_reference_collage_from_pose_mask(
                    ref_images,
                    pose_mask_path=source_reference,
                    background_image_path=str(background_frame_path) if background_frame_path else "",
                    output_dir=output_dir,
                    width=width,
                    height=height,
                    role_names=role_names,
                )
            elif source_reference and Path(source_reference).exists():
                reference_collage_path = Path(source_reference)
            else:
                reference_collage_path = self._create_source_position_reference_collage(
                    ref_images,
                    output_dir=output_dir,
                    width=width,
                    height=height,
                    role_names=role_names,
                    source_positions=source_positions,
                )
            log(f"SCAIL-2 colored segment {segment_index}: reference={reference_collage_path}")
            reference_collage_name = self.upload_file(str(reference_collage_path))
            segment_window = dict(video_window)
            segment_window["skip_first_frames"] = (
                int(video_window.get("skip_first_frames") or 0)
                + segment_start * int(video_window.get("select_every_nth") or 1)
            )
            segment_window["frame_load_cap"] = segment_count

            wf = self._build_colored_mask_template()
            wf = self._patch_workflow(
                wf,
                vid_name,
                subject_names,
                [],
                sam_text,
                positive_prompt,
                width,
                height,
                video_window=segment_window,
                sampler_preset=sampler_preset,
                normalize_reference=normalize_size,
                subject_extra_ref_names=[[] for _ in subject_names],
                reference_collage_name=reference_collage_name,
            )
            if "ref_resize" in wf:
                wf["ref_resize"].setdefault("inputs", {})["width"] = width
                wf["ref_resize"].setdefault("inputs", {})["height"] = height
            if "43" in wf:
                wf["43"].setdefault("inputs", {})["filename_prefix"] = (
                    f"SCAIL2/colored_mask_seg{segment_index:02d}"
                )
            workflow_path = self._save_workflow_debug(
                wf,
                output_dir=output_dir,
                video_path=video_path,
                role_names=role_names,
            )
            log(
                f"SCAIL-2 colored segment {segment_index}/{len(pose_segments)}: "
                f"start={segment_start} frames={segment_count}; workflow={workflow_path}"
            )

            client_id = uuid.uuid4().hex
            resp = self.session.post(
                f"{self.comfy}/prompt",
                json={"prompt": wf, "client_id": client_id},
                timeout=30,
            )
            if not resp.ok:
                raise RuntimeError(f"ComfyUI /prompt HTTP {resp.status_code}: {resp.text[:4000]}")
            task_id = resp.json()["prompt_id"]
            log(f"任务ID: {task_id}")

            history = self._wait_for_completion(task_id, client_id, wf, log)
            outputs = history[task_id]["outputs"]
            video_info = self._first_video_asset(outputs.get("43") or outputs.get(43))
            if not video_info:
                raise RuntimeError("SCAIL-2 colored test 未找到正式输出视频")

            out_path = output_dir / (
                f"scail2_colored_seg{segment_index:02d}_{Path(video_path).stem}_{task_id[:8]}.mp4"
            )
            log(f"下载 colored segment {segment_index}: {video_info['filename']}")
            url = self._download_output_asset(video_info, out_path)

            mask_output_paths: dict[str, list[str]] = {}
            for label, node_id in {"pose": "mask_pose_video", "reference": "mask_reference_save"}.items():
                assets = self._output_assets(outputs.get(node_id) or {})
                local_paths: list[str] = []
                for index, asset in enumerate(assets, 1):
                    suffix = Path(str(asset.get("filename") or "")).suffix or (".mp4" if label == "pose" else ".png")
                    index_part = "" if len(assets) == 1 else f"_{index:02d}"
                    target = output_dir / (
                        f"scail2_colored_seg{segment_index:02d}_{label}_{Path(video_path).stem}_{task_id[:8]}"
                        f"{index_part}{suffix}"
                    )
                    self._download_output_asset(asset, target)
                    local_paths.append(str(target))
                if local_paths:
                    mask_output_paths[label] = local_paths
                    log(f"SCAIL-2 colored segment {segment_index} {label} mask: {'; '.join(local_paths)}")
            return {
                "output_path": str(out_path),
                "output_url": url,
                "prompt_id": task_id,
                "workflow_path": str(workflow_path),
                "reference_collage_path": str(reference_collage_path),
                "mask_output_paths": mask_output_paths,
                "video_window": segment_window,
            }

        segment_results: list[dict] = []
        combined_masks: dict[str, list[str]] = {}
        for index, segment in enumerate(pose_segments, 1):
            result = run_colored_segment(segment, index)
            segment_results.append(result)
            for label, paths in (result.get("mask_output_paths") or {}).items():
                combined_masks.setdefault(label, []).extend(paths)

        if len(segment_results) > 1:
            out_path = output_dir / f"scail2_colored_auto_{Path(video_path).stem}_{uuid.uuid4().hex[:8]}.mp4"
            self._concat_video_segments(
                [Path(item["output_path"]) for item in segment_results],
                out_path,
                log=log,
            )
            url = str(out_path)
            workflow_path = segment_results[-1].get("workflow_path", "")
            reference_collage_path = segment_results[0].get("reference_collage_path", "")
        else:
            out_path = Path(segment_results[0]["output_path"])
            url = str(segment_results[0].get("output_url") or "")
            workflow_path = segment_results[0].get("workflow_path", "")
            reference_collage_path = segment_results[0].get("reference_collage_path", "")

        return {
            "output_path": str(out_path),
            "output_url": url,
            "prompt_id": segment_results[-1].get("prompt_id", ""),
            "workflow_path": str(workflow_path),
            "role_names": role_names,
            "ref_images": ref_images,
            "reference_collage_path": str(reference_collage_path),
            "workflow_mode": "colored_mask_auto_segments" if len(segment_results) > 1 else "colored_mask_test",
            "video_meta": meta.to_dict(),
            "video_window": video_window,
            "output_size": [width, height],
            "sampler_preset": sampler_preset,
            "mask_output_paths": combined_masks,
            "segment_results": segment_results,
            "auto_segments": pose_segments,
        }

    def transfer_bernini_test(
        self,
        video_path: str,
        ref_images: list[str],
        *,
        role_names: list[str] | None = None,
        positive_prompt: str = "a person talking",
        width: int = 512,
        height: int = 896,
        video_window: dict | None = None,
        sampler_preset: str = "balanced",
        normalize_size: bool = True,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Experimental WanAnimatePlus Bernini rv2v path, isolated from SCAIL-2 masks."""

        def log(msg):
            if on_progress:
                on_progress(msg)

        if len(ref_images) > 10:
            raise ValueError("Bernini test supports at most 10 reference images")
        role_names = [str(value or "").strip() for value in (role_names or [])]
        if role_names and len(role_names) != len(ref_images):
            raise ValueError("Bernini role_names must match ref_images")

        missing = self._missing_bernini_nodes()
        if missing:
            raise RuntimeError("ComfyUI missing Bernini test nodes: " + ", ".join(missing))

        from .ffmpeg_tools import probe_video

        meta = probe_video(video_path)
        video_window = self._resolve_video_window(meta, video_window)
        if normalize_size:
            width, height = self._normalized_size(meta.width, meta.height, width, height)
        output_dir = Path(video_path).parent.parent.parent / "04_AI输出成片"
        output_dir.mkdir(parents=True, exist_ok=True)
        log(
            f"Bernini test: {meta.width}x{meta.height} {meta.fps:.2f}fps "
            f"{meta.duration:.2f}s; window={video_window['frame_load_cap']} frames; output={width}x{height}"
        )
        if role_names:
            log("Bernini reference order: " + " / ".join(role_names))

        log("upload Bernini source video...")
        vid_name = self.upload_file(video_path)

        subject_names: list[str] = []
        for index, image_path in enumerate(ref_images, 1):
            label = role_names[index - 1] if index - 1 < len(role_names) and role_names[index - 1] else f"role{index}"
            log(f"upload {label} Bernini reference {index}/{len(ref_images)}...")
            uploaded = self.upload_file(image_path)
            log(f"{label} Bernini remote file: {uploaded}")
            subject_names.append(uploaded)

        wf = self._build_bernini_test_template(len(subject_names))
        wf = self._patch_workflow(
            wf,
            vid_name,
            subject_names,
            [],
            "person",
            positive_prompt,
            width,
            height,
            video_window=video_window,
            sampler_preset=sampler_preset,
            normalize_reference=False,
            subject_extra_ref_names=[[] for _ in subject_names],
        )
        workflow_path = self._save_workflow_debug(
            wf,
            output_dir=output_dir,
            video_path=video_path,
            role_names=role_names,
        )
        log(f"Bernini test workflow saved: {workflow_path}")

        client_id = uuid.uuid4().hex
        log("submit Bernini test inference...")
        resp = self.session.post(
            f"{self.comfy}/prompt",
            json={"prompt": wf, "client_id": client_id},
            timeout=30,
        )
        if not resp.ok:
            raise RuntimeError(f"ComfyUI /prompt HTTP {resp.status_code}: {resp.text[:4000]}")
        task_id = resp.json()["prompt_id"]
        log(f"task id: {task_id}")

        history = self._wait_for_completion(task_id, client_id, wf, log)
        outputs = history[task_id]["outputs"]
        video_info = self._first_video_asset(outputs.get("43") or outputs.get(43))
        if not video_info:
            for node_output in outputs.values():
                video_info = self._first_video_asset(node_output)
                if video_info:
                    break
        if not video_info:
            raise RuntimeError(
                "Bernini test finished but no output video was found; "
                + self._history_debug_summary(history.get(task_id) or {})
            )

        out_path = output_dir / f"bernini_{Path(video_path).stem}_{task_id[:8]}.mp4"
        log(f"download Bernini result: {video_info['filename']}")
        url = self._download_output_asset(video_info, out_path)

        return {
            "output_path": str(out_path),
            "output_url": url,
            "prompt_id": task_id,
            "workflow_path": str(workflow_path),
            "role_names": role_names,
            "ref_images": ref_images,
            "workflow_mode": "bernini_rv2v_test",
            "video_meta": meta.to_dict(),
            "video_window": video_window,
            "output_size": [width, height],
            "sampler_preset": sampler_preset,
        }

    # ------------------------------------------------------------------
    # 模板构建
    # ------------------------------------------------------------------

    def _wait_for_completion(
        self,
        task_id: str,
        client_id: str,
        wf: dict,
        log: Callable[[str], None],
    ) -> dict:
        log("等待推理完成...")
        ws_url = self._websocket_url(client_id)
        try:
            import websocket

            ws = websocket.create_connection(ws_url, timeout=5)
        except Exception as exc:  # noqa: BLE001
            log(f"ComfyUI 进度通道不可用，改用轮询: {exc}")
            return self._poll_history_until_done(task_id, log)

        last_node = None
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
                    return self._poll_history_until_done(task_id, log)

                if message:
                    if isinstance(message, (bytes, bytearray, memoryview)):
                        continue
                    try:
                        event = json.loads(message)
                    except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
                        event = {}
                    if not isinstance(event, dict):
                        event = {}
                    kind = event.get("type")
                    data = event.get("data") or {}
                    prompt_id = data.get("prompt_id")
                    if prompt_id in (None, task_id):
                        if kind == "executing":
                            node_id = data.get("node")
                            if node_id is None:
                                history = self._fetch_history(task_id)
                                if task_id in history:
                                    log("推理完成")
                                    return history
                            elif node_id != last_node:
                                last_node = node_id
                                last_percent = -1
                                log(f"ComfyUI 节点: {self._node_label(wf, str(node_id))}")
                        elif kind == "progress":
                            value = int(data.get("value") or 0)
                            maximum = max(1, int(data.get("max") or 1))
                            percent = max(0, min(100, int(round(value * 100 / maximum))))
                            if percent >= last_percent + 5 or percent in (0, 100):
                                last_percent = percent
                                label = self._node_label(wf, str(last_node)) if last_node else "当前节点"
                                log(f"ComfyUI 进度: {percent}% ({label} {value}/{maximum})")

                if now - last_history_poll >= 10:
                    last_history_poll = now
                    history = self._fetch_history(task_id)
                    if task_id in history:
                        log("推理完成")
                        return history
        finally:
            try:
                ws.close()
            except Exception:
                pass

    def _poll_history_until_done(self, task_id: str, log: Callable[[str], None]) -> dict:
        while True:
            time.sleep(10)
            history = self._fetch_history(task_id)
            if task_id in history:
                log("推理完成")
                return history

    def _fetch_history(self, task_id: str) -> dict:
        try:
            hist_resp = self.session.get(f"{self.comfy}/history/{task_id}", timeout=120)
        except (requests.Timeout, requests.ConnectionError):
            return {}
        if hist_resp.status_code in _RETRYABLE_UPLOAD_STATUS_CODES:
            return {}
        hist_resp.raise_for_status()
        return hist_resp.json()

    def _resolve_driving_track_order(
        self,
        *,
        vid_name: str,
        subject_count: int,
        sam_text: str,
        width: int,
        height: int,
        video_window: dict,
        output_dir: Path,
        log: Callable[[str], None],
        strict: bool = False,
    ) -> list[int]:
        """Map SAM's opaque object indices to the source frame's left-to-right order."""
        fallback = list(range(subject_count))
        if subject_count < 2:
            return fallback

        source_frame_cap = max(1, int((video_window or {}).get("frame_load_cap") or 1))
        sample_count = min(source_frame_cap, max(1, min(18, max(6, subject_count * 5))))
        sample_every = max(1, math.ceil(source_frame_cap / sample_count))
        window = {
            "force_rate": 24,
            "skip_first_frames": 0,
            **(video_window or {}),
            "select_every_nth": sample_every,
            "frame_load_cap": sample_count,
        }
        log(
            "SCAIL-2 SAM track preflight samples "
            f"{window['frame_load_cap']} frames every {window['select_every_nth']} source frames"
        )
        workflow = {
            "source_video": {
                "class_type": "VHS_LoadVideo",
                "inputs": {
                    "video": vid_name,
                    "force_rate": int(window["force_rate"]),
                    "custom_width": 0,
                    "custom_height": 0,
                    "frame_load_cap": int(window["frame_load_cap"]),
                    "skip_first_frames": int(window["skip_first_frames"]),
                    "select_every_nth": int(window["select_every_nth"]),
                    "format": "AnimateDiff",
                },
            },
            "fit_source": {
                "class_type": "SCAIL2FitVideo",
                "inputs": {
                    "resolution": "custom",
                    "custom_width": width,
                    "custom_height": height,
                    "video": ["source_video", 0],
                },
            },
            "sam_model": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "sam3.1_multiplex_fp16.safetensors"},
            },
            "sam_text": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": sam_text, "clip": ["sam_model", 1]},
            },
            "source_tracks": {
                "class_type": "SAM3_VideoTrack",
                "inputs": {
                    "detection_threshold": SAM3_VIDEO_DETECTION_THRESHOLD,
                    "max_objects": subject_count,
                    "detect_interval": SAM3_VIDEO_DETECT_INTERVAL,
                    "images": ["fit_source", 0],
                    "model": ["sam_model", 0],
                    "conditioning": ["sam_text", 0],
                },
            },
        }
        for raw_index in range(subject_count):
            prefix = f"track_order_{raw_index}"
            workflow[f"{prefix}_mask"] = {
                "class_type": "SAM3_TrackToMask",
                "inputs": {
                    "track_data": ["source_tracks", 0],
                    "object_indices": str(raw_index),
                },
            }
            workflow[f"{prefix}_image"] = {
                "class_type": "MaskToImage",
                "inputs": {"mask": [f"{prefix}_mask", 0]},
            }
            workflow[f"{prefix}_save"] = {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": f"SCAIL2/track_order_{raw_index}",
                    "images": [f"{prefix}_image", 0],
                },
            }

        try:
            client_id = uuid.uuid4().hex
            response = self.session.post(
                f"{self.comfy}/prompt",
                json={"prompt": workflow, "client_id": client_id},
                timeout=30,
            )
            response.raise_for_status()
            task_id = response.json()["prompt_id"]
            history = self._wait_for_completion(task_id, client_id, workflow, log)
            outputs = (history.get(task_id) or {}).get("outputs") or {}

            centers: dict[int, float] = {}
            from PIL import Image

            debug_dir = output_dir / "_debug_scail2"
            debug_dir.mkdir(parents=True, exist_ok=True)
            for raw_index in range(subject_count):
                node_id = f"track_order_{raw_index}_save"
                assets = self._output_assets(outputs.get(node_id) or {})
                if not assets:
                    raise RuntimeError(f"no output for SAM track {raw_index}")
                sampled_centers: list[tuple[int, float]] = []
                for sample_index, asset in enumerate(assets, 1):
                    suffix = Path(str(asset.get("filename") or "")).suffix or ".png"
                    target = debug_dir / (
                        f"scail2_track_order_{Path(vid_name).stem}_{task_id[:8]}_"
                        f"{raw_index}_{sample_index:03d}{suffix}"
                    )
                    self._download_output_asset(asset, target)
                    with Image.open(target) as image:
                        binary = image.convert("L").point(
                            lambda value: 255 if value > 16 else 0
                        )
                        bbox = binary.getbbox()
                    if bbox is None:
                        continue
                    area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                    sampled_centers.append((area, (bbox[0] + bbox[2]) / 2.0))
                if not sampled_centers:
                    raise RuntimeError(f"empty SAM track {raw_index}")
                centers[raw_index] = max(sampled_centers, key=lambda item: item[0])[1]

            if len(centers) != subject_count:
                raise RuntimeError("incomplete SAM track mapping")
            ordered = [raw_index for raw_index, _ in sorted(centers.items(), key=lambda item: item[1])]
            if len(set(ordered)) != subject_count:
                raise RuntimeError("duplicate SAM track mapping")
            log("SCAIL-2 SAM raw-to-position mapping: " + " -> ".join(map(str, ordered)))
            return ordered
        except Exception as exc:  # noqa: BLE001
            if strict:
                raise RuntimeError(f"SCAIL-2 SAM track preflight failed: {exc}") from exc
            log(f"SCAIL-2 SAM track preflight failed; using raw order: {exc}")
            return fallback

    @staticmethod
    def _coerce_source_identity_points(
        source_identity_points: list[list[float] | None] | None,
        subject_count: int,
    ) -> list[tuple[float, float]] | None:
        if not source_identity_points or len(source_identity_points) < subject_count:
            return None
        points: list[tuple[float, float]] = []
        for index in range(subject_count):
            point = source_identity_points[index]
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                return None
            try:
                px = max(0.0, min(1.0, float(point[0])))
                py = max(0.0, min(1.0, float(point[1])))
            except (TypeError, ValueError):
                return None
            points.append((px, py))
        return points

    @staticmethod
    def _coerce_source_identity_seed_items(
        source_identity_points: list[list[float] | None] | None,
        source_identity_shapes: list[dict | None] | None,
        subject_count: int,
    ) -> list[dict] | None:
        if subject_count <= 0:
            return None
        items: list[dict] = []
        for index in range(subject_count):
            shape = None
            if source_identity_shapes and index < len(source_identity_shapes):
                shape = Scail2Client._coerce_source_identity_shape(source_identity_shapes[index])
            if shape:
                items.append({"shape": shape})
                continue
            point = None
            if source_identity_points and index < len(source_identity_points):
                point = source_identity_points[index]
            if isinstance(point, (list, tuple)) and len(point) == 2:
                try:
                    px = max(0.0, min(1.0, float(point[0])))
                    py = max(0.0, min(1.0, float(point[1])))
                except (TypeError, ValueError):
                    return None
                items.append({"point": (px, py)})
                continue
            return None
        return items

    @staticmethod
    def _coerce_source_identity_shape(shape: dict | None) -> dict | None:
        if not isinstance(shape, dict):
            return None
        raw_points = shape.get("points")
        if not isinstance(raw_points, list):
            return None
        points: list[tuple[float, float]] = []
        for point in raw_points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            try:
                x = max(0.0, min(1.0, float(point[0])))
                y = max(0.0, min(1.0, float(point[1])))
            except (TypeError, ValueError):
                continue
            if points and abs(points[-1][0] - x) < 0.002 and abs(points[-1][1] - y) < 0.002:
                continue
            points.append((x, y))
        if len(points) < 3:
            return None
        return {"type": "freehand", "points": points}

    def _create_source_identity_shape_mask(
        self,
        shape: dict,
        *,
        width: int,
        height: int,
        output_dir: Path,
        index: int,
    ) -> Path:
        try:
            from PIL import Image, ImageDraw
        except ImportError as exc:  # pragma: no cover - Pillow is a project dependency
            raise RuntimeError("Pillow is required to build hand-drawn SAM3 seed masks") from exc

        points = [
            (int(round(float(x) * width)), int(round(float(y) * height)))
            for x, y in shape.get("points", [])
        ]
        if len(points) < 3:
            raise ValueError("source_identity_shape_needs_at_least_3_points")
        seed_dir = output_dir / ".sam3_identity_seeds"
        seed_dir.mkdir(parents=True, exist_ok=True)
        mask_path = seed_dir / f"source_identity_shape_{index}_{uuid.uuid4().hex[:8]}.png"
        image = Image.new("RGBA", (width, height), (0, 0, 0, 255))
        draw = ImageDraw.Draw(image)
        brush = max(6, int(round(min(width, height) * 0.025)))
        selected = (255, 255, 255, 0)
        draw.polygon(points, fill=selected)
        draw.line(points + [points[0]], fill=selected, width=brush, joint="curve")
        image.save(mask_path)
        preview_path = mask_path.with_name(mask_path.stem + "_preview.png")
        preview = Image.new("RGB", (width, height), (0, 0, 0))
        preview_draw = ImageDraw.Draw(preview)
        preview_draw.polygon(points, fill=(255, 255, 255))
        preview_draw.line(points + [points[0]], fill=(255, 255, 255), width=brush, joint="curve")
        preview.save(preview_path)
        return mask_path

    @staticmethod
    def _sam3_seed_geometry(
        points: list[tuple[float, float]],
        index: int,
        *,
        width: int,
        height: int,
    ) -> tuple[dict[str, int], list[dict[str, int]], list[dict[str, int]]]:
        x, y = points[index]
        left_candidates = [other_x for point_index, (other_x, _other_y) in enumerate(points) if point_index != index and other_x < x]
        right_candidates = [other_x for point_index, (other_x, _other_y) in enumerate(points) if point_index != index and other_x > x]
        left = max(left_candidates) if left_candidates else None
        right = min(right_candidates) if right_candidates else None
        x_min = 0.0 if left is None else (left + x) / 2.0
        x_max = 1.0 if right is None else (x + right) / 2.0
        if x_max - x_min < 0.18:
            x_min = max(0.0, x - 0.18)
            x_max = min(1.0, x + 0.18)
        x_min = max(0.0, min(1.0, x_min - 0.02))
        x_max = max(0.0, min(1.0, x_max + 0.02))
        y_min = 0.02
        y_max = 0.98
        bbox = {
            "x": int(round(x_min * width)),
            "y": int(round(y_min * height)),
            "width": max(1, int(round((x_max - x_min) * width))),
            "height": max(1, int(round((y_max - y_min) * height))),
        }
        positives = [
            {"x": int(round(x * width)), "y": int(round(y * height))},
            {"x": int(round(x * width)), "y": int(round(max(y_min, y - 0.18) * height))},
            {"x": int(round(x * width)), "y": int(round(min(y_max, y + 0.22) * height))},
        ]
        negatives = [
            {"x": int(round(max(0.0, x_min - 0.03) * width)), "y": int(round(y * height))},
            {"x": int(round(min(1.0, x_max + 0.03) * width)), "y": int(round(y * height))},
        ]
        for other_index, (ox, oy) in enumerate(points):
            if other_index == index:
                continue
            negatives.append({"x": int(round(ox * width)), "y": int(round(oy * height))})
        return bbox, positives, negatives

    def _add_sam3_initial_mask_seed(
        self,
        wf: dict,
        source_identity_points: list[tuple[float, float]],
        *,
        width: int,
        height: int,
    ) -> None:
        if not source_identity_points:
            return

        wf["sam_seed_frame"] = {
            "class_type": "ImageFromBatch",
            "inputs": {"image": ["3", 0], "batch_index": 0, "length": 1},
        }
        seed_mask_refs: list[list[str | int]] = []
        for index in range(len(source_identity_points)):
            bbox, positives, negatives = self._sam3_seed_geometry(
                source_identity_points,
                index,
                width=width,
                height=height,
            )
            bbox_id = f"sam_seed_bbox_{index}"
            pos_id = f"sam_seed_positive_{index}"
            neg_id = f"sam_seed_negative_{index}"
            detect_id = f"sam_seed_detect_{index}"
            wf[bbox_id] = {
                "class_type": "PrimitiveBoundingBox",
                "inputs": {
                    "x": bbox["x"],
                    "y": bbox["y"],
                    "width": bbox["width"],
                    "height": bbox["height"],
                },
            }
            wf[pos_id] = {
                "class_type": "PrimitiveString",
                "inputs": {"value": json.dumps(positives, ensure_ascii=False)},
            }
            wf[neg_id] = {
                "class_type": "PrimitiveString",
                "inputs": {"value": json.dumps(negatives, ensure_ascii=False)},
            }
            wf[detect_id] = {
                "class_type": "SAM3_Detect",
                "inputs": {
                    "model": ["30", 0],
                    "image": ["sam_seed_frame", 0],
                    "threshold": SAM3_VIDEO_DETECTION_THRESHOLD,
                    "refine_iterations": 2,
                    "individual_masks": True,
                    "conditioning": ["31", 0],
                    "bboxes": [bbox_id, 0],
                    "positive_coords": [pos_id, 0],
                    "negative_coords": [neg_id, 0],
                },
            }
            seed_mask_refs.append([detect_id, 0])

        if len(seed_mask_refs) == 1:
            wf["sam_seed_initial_mask"] = {
                "class_type": "MaskToImage",
                "inputs": {"mask": seed_mask_refs[0]},
            }
            wf["32"]["inputs"]["initial_mask"] = seed_mask_refs[0]
            return

        wf["sam_seed_initial_mask_batch"] = {
            "class_type": "MaskBatchMulti",
            "inputs": {
                "inputcount": len(seed_mask_refs),
                "mask_1": seed_mask_refs[0],
                "mask_2": seed_mask_refs[1],
            },
        }
        for index in range(2, len(seed_mask_refs)):
            wf["sam_seed_initial_mask_batch"]["inputs"][f"mask_{index + 1}"] = seed_mask_refs[index]
        wf["32"]["inputs"]["initial_mask"] = ["sam_seed_initial_mask_batch", 0]

    def _add_sam3_mixed_initial_mask_seed(
        self,
        wf: dict,
        seed_items: list[dict],
        *,
        width: int,
        height: int,
        output_dir: Path,
    ) -> None:
        if not seed_items:
            return

        seed_centers: list[tuple[float, float]] = []
        for item in seed_items:
            point = item.get("point")
            if isinstance(point, tuple) and len(point) == 2:
                seed_centers.append(point)
                continue
            shape = item.get("shape") or {}
            points = shape.get("points") or [(0.5, 0.5)]
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            seed_centers.append((sum(xs) / max(1, len(xs)), sum(ys) / max(1, len(ys))))

        if any(item.get("point") for item in seed_items):
            wf["sam_seed_frame"] = {
                "class_type": "ImageFromBatch",
                "inputs": {"image": ["3", 0], "batch_index": 0, "length": 1},
            }

        seed_mask_refs: list[list[str | int]] = []
        for index, item in enumerate(seed_items):
            shape = item.get("shape")
            if shape:
                mask_path = self._create_source_identity_shape_mask(
                    shape,
                    width=width,
                    height=height,
                    output_dir=output_dir,
                    index=index,
                )
                uploaded_mask = self.upload_file(str(mask_path))
                load_id = f"sam_seed_shape_load_{index}"
                wf[load_id] = {
                    "class_type": "LoadImage",
                    "inputs": {"image": uploaded_mask},
                }
                seed_mask_refs.append([load_id, 1])
                continue

            point = item.get("point")
            if not isinstance(point, tuple) or len(point) != 2:
                raise ValueError("source_identity_seed_item_missing_shape_or_point")
            bbox, positives, negatives = self._sam3_seed_geometry(
                seed_centers,
                index,
                width=width,
                height=height,
            )
            bbox_id = f"sam_seed_bbox_{index}"
            pos_id = f"sam_seed_positive_{index}"
            neg_id = f"sam_seed_negative_{index}"
            detect_id = f"sam_seed_detect_{index}"
            wf[bbox_id] = {
                "class_type": "PrimitiveBoundingBox",
                "inputs": {
                    "x": bbox["x"],
                    "y": bbox["y"],
                    "width": bbox["width"],
                    "height": bbox["height"],
                },
            }
            wf[pos_id] = {
                "class_type": "PrimitiveString",
                "inputs": {"value": json.dumps(positives, ensure_ascii=False)},
            }
            wf[neg_id] = {
                "class_type": "PrimitiveString",
                "inputs": {"value": json.dumps(negatives, ensure_ascii=False)},
            }
            wf[detect_id] = {
                "class_type": "SAM3_Detect",
                "inputs": {
                    "model": ["30", 0],
                    "image": ["sam_seed_frame", 0],
                    "threshold": SAM3_VIDEO_DETECTION_THRESHOLD,
                    "refine_iterations": 2,
                    "individual_masks": True,
                    "conditioning": ["31", 0],
                    "bboxes": [bbox_id, 0],
                    "positive_coords": [pos_id, 0],
                    "negative_coords": [neg_id, 0],
                },
            }
            seed_mask_refs.append([detect_id, 0])

        if len(seed_mask_refs) == 1:
            wf["32"]["inputs"]["initial_mask"] = seed_mask_refs[0]
            return

        wf["sam_seed_initial_mask_batch"] = {
            "class_type": "MaskBatchMulti",
            "inputs": {
                "inputcount": len(seed_mask_refs),
                "mask_1": seed_mask_refs[0],
                "mask_2": seed_mask_refs[1],
            },
        }
        for index in range(2, len(seed_mask_refs)):
            wf["sam_seed_initial_mask_batch"]["inputs"][f"mask_{index + 1}"] = seed_mask_refs[index]
        wf["32"]["inputs"]["initial_mask"] = ["sam_seed_initial_mask_batch", 0]

    def _add_sam3_independent_identity_tracks(
        self,
        wf: dict,
        seed_items: list[dict],
        *,
        width: int,
        height: int,
        output_dir: Path,
    ) -> list[str]:
        if not seed_items:
            return []

        seed_centers: list[tuple[float, float]] = []
        for item in seed_items:
            point = item.get("point")
            if isinstance(point, tuple) and len(point) == 2:
                seed_centers.append(point)
                continue
            shape = item.get("shape") or {}
            points = shape.get("points") or [(0.5, 0.5)]
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            seed_centers.append((sum(xs) / max(1, len(xs)), sum(ys) / max(1, len(ys))))

        if any(item.get("point") for item in seed_items):
            wf["sam_seed_frame"] = {
                "class_type": "ImageFromBatch",
                "inputs": {"image": ["3", 0], "batch_index": 0, "length": 1},
            }

        preview_save_ids: list[str] = []
        for index, item in enumerate(seed_items):
            shape = item.get("shape")
            if shape:
                mask_path = self._create_source_identity_shape_mask(
                    shape,
                    width=width,
                    height=height,
                    output_dir=output_dir,
                    index=index,
                )
                uploaded_mask = self.upload_file(str(mask_path))
                load_id = f"sam_seed_shape_load_{index}"
                seed_mask_ref = [load_id, 1]
                wf[load_id] = {
                    "class_type": "LoadImage",
                    "inputs": {"image": uploaded_mask},
                }
            else:
                point = item.get("point")
                if not isinstance(point, tuple) or len(point) != 2:
                    raise ValueError("source_identity_seed_item_missing_shape_or_point")
                bbox, positives, negatives = self._sam3_seed_geometry(
                    seed_centers,
                    index,
                    width=width,
                    height=height,
                )
                bbox_id = f"sam_seed_bbox_{index}"
                pos_id = f"sam_seed_positive_{index}"
                neg_id = f"sam_seed_negative_{index}"
                detect_id = f"sam_seed_detect_{index}"
                wf[bbox_id] = {
                    "class_type": "PrimitiveBoundingBox",
                    "inputs": {
                        "x": bbox["x"],
                        "y": bbox["y"],
                        "width": bbox["width"],
                        "height": bbox["height"],
                    },
                }
                wf[pos_id] = {
                    "class_type": "PrimitiveString",
                    "inputs": {"value": json.dumps(positives, ensure_ascii=False)},
                }
                wf[neg_id] = {
                    "class_type": "PrimitiveString",
                    "inputs": {"value": json.dumps(negatives, ensure_ascii=False)},
                }
                wf[detect_id] = {
                    "class_type": "SAM3_Detect",
                    "inputs": {
                        "model": ["30", 0],
                        "image": ["sam_seed_frame", 0],
                        "threshold": SAM3_VIDEO_DETECTION_THRESHOLD,
                        "refine_iterations": 2,
                        "individual_masks": True,
                        "conditioning": ["31", 0],
                        "bboxes": [bbox_id, 0],
                        "positive_coords": [pos_id, 0],
                        "negative_coords": [neg_id, 0],
                    },
                }
                seed_mask_ref = [detect_id, 0]

            track_id = f"sam_seed_track_{index}"
            track_inputs = copy.deepcopy(wf["32"]["inputs"])
            track_inputs["max_objects"] = 1
            track_inputs["initial_mask"] = seed_mask_ref
            wf[track_id] = {
                "class_type": "SAM3_VideoTrack",
                "inputs": track_inputs,
            }
            prefix = f"preview_track_{index}"
            wf[f"{prefix}_mask"] = {
                "class_type": "SAM3_TrackToMask",
                "inputs": {
                    "track_data": [track_id, 0],
                    "object_indices": "0",
                },
            }
            wf[f"{prefix}_image"] = {
                "class_type": "MaskToImage",
                "inputs": {"mask": [f"{prefix}_mask", 0]},
            }
            save_id = f"{prefix}_save"
            wf[save_id] = {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": f"SCAIL2/debug_track_{index}",
                    "images": [f"{prefix}_image", 0],
                },
            }
            preview_save_ids.append(save_id)

        return preview_save_ids

    @staticmethod
    def _mask_nonzero_ratio(path: Path, *, threshold: int = 8) -> float:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - Pillow is a project dependency
            raise RuntimeError("Pillow is required to inspect SAM3 mask frames") from exc
        with Image.open(path) as raw:
            image = raw.convert("L")
            pixels = image.getdata()
            total = max(1, image.width * image.height)
            return sum(1 for value in pixels if value > threshold) / total

    def _hold_last_nonempty_masks(
        self,
        role_track_paths: list[list[Path]],
        *,
        empty_ratio: float = 0.001,
        min_valid_ratio: float = 0.003,
    ) -> int:
        repaired = 0
        for paths in role_track_paths:
            last_good: Path | None = None
            for path in paths:
                ratio = self._mask_nonzero_ratio(path)
                if ratio >= min_valid_ratio:
                    last_good = path
                    continue
                if last_good is None or ratio > empty_ratio:
                    continue
                try:
                    from PIL import Image
                except ImportError as exc:  # pragma: no cover - Pillow is a project dependency
                    raise RuntimeError("Pillow is required to repair SAM3 mask frames") from exc
                with Image.open(last_good) as source:
                    source.save(path)
                repaired += 1
        return repaired

    def _propagate_sparse_masks_from_video(
        self,
        role_track_paths: list[list[Path]],
        video_path: Path,
        *,
        min_valid_ratio: float = 0.003,
    ) -> int:
        try:
            import cv2
            import numpy as np
            from PIL import Image
        except ImportError:
            return 0

        if not role_track_paths:
            return 0
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return 0
        try:
            frames: list[np.ndarray] = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if gray.shape[1] > 320:
                    scale = 320.0 / max(1, gray.shape[1])
                    gray = cv2.resize(gray, (max(1, int(round(gray.shape[1] * scale))), max(1, int(round(gray.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
                frames.append(gray.astype(np.float32))
        finally:
            capture.release()
        if len(frames) < 2:
            return 0

        shifts: list[tuple[float, float]] = [(0.0, 0.0)]
        prev = frames[0]
        window = cv2.createHanningWindow((prev.shape[1], prev.shape[0]), cv2.CV_32F)
        for frame in frames[1:]:
            cur = frame
            if cur.shape != prev.shape:
                cur = cv2.resize(cur, (prev.shape[1], prev.shape[0]), interpolation=cv2.INTER_AREA)
            try:
                dx, dy = cv2.phaseCorrelate(prev, cur, window)[0]
            except Exception:
                dx, dy = (0.0, 0.0)
            shifts.append((float(dx), float(dy)))
            prev = cur

        def load_mask(path: Path) -> np.ndarray | None:
            try:
                with Image.open(path) as image:
                    return np.array(image.convert("L"))
            except Exception:
                return None

        def save_mask(path: Path, mask: np.ndarray) -> None:
            from PIL import Image
            Image.fromarray(mask.astype(np.uint8)).save(path)

        def mask_ratio(mask: np.ndarray) -> float:
            total = max(1, mask.shape[0] * mask.shape[1])
            return float(np.count_nonzero(mask > 8)) / float(total)

        def shift_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
            matrix = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
            return cv2.warpAffine(mask, matrix, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        repaired = 0
        for paths in role_track_paths:
            if not paths:
                continue
            masks = [load_mask(path) for path in paths]
            if not masks or masks[0] is None:
                continue
            first_good_index = next((i for i, mask in enumerate(masks) if mask is not None and mask_ratio(mask) >= min_valid_ratio), None)
            if first_good_index is None:
                continue
            base_mask = masks[first_good_index]
            if base_mask is None:
                continue
            for frame_index, path in enumerate(paths):
                current = masks[frame_index]
                if current is not None and mask_ratio(current) >= min_valid_ratio:
                    base_mask = current
                    continue
                ref_index = min(frame_index, len(shifts) - 1)
                dx, dy = shifts[ref_index]
                source_index = max(0, first_good_index)
                accumulated_dx = sum(item[0] for item in shifts[min(source_index, ref_index):ref_index + 1]) if ref_index > source_index else 0.0
                accumulated_dy = sum(item[1] for item in shifts[min(source_index, ref_index):ref_index + 1]) if ref_index > source_index else 0.0
                propagated = shift_mask(base_mask, accumulated_dx, accumulated_dy)
                if mask_ratio(propagated) < min_valid_ratio:
                    propagated = base_mask
                save_mask(path, propagated)
                repaired += 1
        return repaired

    def _websocket_url(self, client_id: str) -> str:
        parsed = urlparse(self.comfy)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return urlunparse((scheme, parsed.netloc, "/ws", "", f"clientId={client_id}", ""))

    @staticmethod
    def _node_label(wf: dict, node_id: str) -> str:
        node = wf.get(str(node_id)) or {}
        class_type = str(node.get("class_type") or "unknown")
        return f"[{node_id}] {class_type}"

    @staticmethod
    def _prune_workflow_for_outputs(wf: dict, output_node_ids: list[str]) -> dict:
        required: set[str] = set()
        stack = [str(node_id) for node_id in output_node_ids]

        def push_refs(value) -> None:
            if isinstance(value, list):
                if len(value) == 2 and isinstance(value[0], str) and value[0] in wf:
                    stack.append(value[0])
                    return
                for item in value:
                    push_refs(item)
            elif isinstance(value, dict):
                for item in value.values():
                    push_refs(item)

        while stack:
            node_id = stack.pop()
            if node_id in required or node_id not in wf:
                continue
            required.add(node_id)
            push_refs((wf[node_id].get("inputs") or {}))

        return {node_id: copy.deepcopy(wf[node_id]) for node_id in wf if node_id in required}

    @staticmethod
    def _without_debug_mask_outputs(wf: dict) -> dict:
        wf = copy.deepcopy(wf)
        for node_id in ("mask_pose_video", "mask_reference_save", "mask_prefix_save"):
            wf.pop(node_id, None)
        return wf

    @staticmethod
    def _output_assets(node_output: dict | None) -> list[dict]:
        if not isinstance(node_output, dict):
            return []
        assets: list[dict] = []
        for key in ("videos", "gifs", "images"):
            value = node_output.get(key)
            if isinstance(value, list):
                assets.extend(item for item in value if isinstance(item, dict))
        return assets

    @classmethod
    def _first_output_asset(cls, node_output: dict | None) -> dict | None:
        assets = cls._output_assets(node_output)
        return assets[0] if assets else None

    @staticmethod
    def _video_assets(node_output: dict | None) -> list[dict]:
        if not isinstance(node_output, dict):
            return []
        assets: list[dict] = []
        for key in ("videos", "gifs"):
            value = node_output.get(key)
            if isinstance(value, list):
                assets.extend(item for item in value if isinstance(item, dict))
        return assets

    @classmethod
    def _first_video_asset(cls, node_output: dict | None) -> dict | None:
        assets = cls._video_assets(node_output)
        return assets[0] if assets else None

    @staticmethod
    def _is_image_asset(asset: dict) -> bool:
        suffix = Path(str(asset.get("filename") or "")).suffix.lower()
        return suffix in {".png", ".jpg", ".jpeg", ".webp"}

    @staticmethod
    def _encode_frame_sequence_to_video(frame_paths: list[Path], output_path: Path, fps: float) -> None:
        if not frame_paths:
            raise RuntimeError("No frames available for preview video encoding")
        import shutil
        import tempfile
        from PIL import Image

        def read_frame(path: Path) -> Image.Image | None:
            try:
                with Image.open(path) as image:
                    return image.convert("RGB")
            except Exception:  # noqa: BLE001
                return None

        first = read_frame(frame_paths[0])
        if first is None:
            raise RuntimeError(f"Failed to read preview frame: {frame_paths[0]}")
        width, height = first.size
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_path.stem}_frames_", dir=str(output_path.parent)))
        try:
            written = 0
            for index, frame_path in enumerate(frame_paths, 1):
                frame = read_frame(frame_path)
                if frame is None:
                    continue
                if frame.size != (width, height):
                    frame = frame.resize((width, height), Image.Resampling.NEAREST)
                frame.save(temp_dir / f"frame_{index:06d}.png")
                written += 1
        finally:
            if written <= 0:
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise RuntimeError("No readable preview frames were written")

        from .ffmpeg_tools import ffmpeg_path, run_command

        try:
            run_command([
                ffmpeg_path(),
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(max(1.0, float(fps) or 24.0)),
                "-i",
                str(temp_dir / "frame_%06d.png"),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def _make_white_depth_video(depth_path: Path, output_path: Path) -> None:
        import shutil
        import tempfile

        import cv2
        import numpy as np

        capture = cv2.VideoCapture(str(depth_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot_open_depth_video: {depth_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 24.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError("invalid_depth_video_size")

        frames: list[np.ndarray] = []
        samples: list[np.ndarray] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            samples.append(gray.reshape(-1)[:: max(1, gray.size // 2048)])
        capture.release()
        if not frames:
            raise RuntimeError("empty_depth_video")

        sample = np.concatenate(samples).astype(np.float32)
        lo = float(np.percentile(sample, 2.0))
        hi = float(np.percentile(sample, 98.0))
        if hi <= lo + 1.0:
            lo, hi = float(sample.min()), float(sample.max())
        if hi <= lo + 1.0:
            lo, hi = 0.0, 255.0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = Path(tempfile.mktemp(prefix=f".{output_path.stem}_", suffix=".mp4", dir=str(output_path.parent)))
        writer = cv2.VideoWriter(
            str(temp_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("cannot_open_white_mask_writer")
        try:
            for gray in frames:
                norm = (gray.astype(np.float32) - lo) / max(1.0, hi - lo)
                norm = np.clip(norm, 0.0, 1.0)
                lifted = 0.22 + 0.78 * np.power(norm, 0.72)
                white = np.clip(lifted * 255.0, 0, 255).astype(np.uint8)
                writer.write(cv2.cvtColor(white, cv2.COLOR_GRAY2BGR))
        finally:
            writer.release()

        if not temp_path.exists() or temp_path.stat().st_size <= 0:
            raise RuntimeError("white_mask_temp_output_missing")

        try:
            from .ffmpeg_tools import ffmpeg_path, run_command

            run_command([
                ffmpeg_path(),
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(temp_path),
                "-map",
                "0:v:0",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ])
            temp_path.unlink(missing_ok=True)
        except Exception:
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            shutil.move(str(temp_path), str(output_path))

        if not output_path.exists() or output_path.stat().st_size <= 0:
            raise RuntimeError("white_mask_output_missing")

    @staticmethod
    def _make_identity_gray_relief_video(
        video_path: Path,
        output_path: Path,
        *,
        width: int,
        height: int,
        video_window: dict,
    ) -> None:
        import shutil
        import tempfile

        import cv2
        import numpy as np

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot_open_video: {video_path}")
        fps = max(1.0, float((video_window or {}).get("force_rate") or capture.get(cv2.CAP_PROP_FPS) or 24.0))
        width = max(1, int(width or capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0))
        height = max(1, int(height or capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
        skip_first = max(0, int((video_window or {}).get("skip_first_frames") or 0))
        select_every = max(1, int((video_window or {}).get("select_every_nth") or 1))
        frame_cap = max(1, int((video_window or {}).get("frame_load_cap") or 1_000_000))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = Path(tempfile.mktemp(prefix=f".{output_path.stem}_", suffix=".mp4", dir=str(output_path.parent)))
        writer = cv2.VideoWriter(
            str(temp_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("cannot_open_gray_relief_writer")

        clahe = cv2.createCLAHE(clipLimit=1.25, tileGridSize=(8, 8))
        face_xml = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        profile_xml = Path(cv2.data.haarcascades) / "haarcascade_profileface.xml"
        face_cascade = cv2.CascadeClassifier(str(face_xml))
        profile_cascade = cv2.CascadeClassifier(str(profile_xml))

        def detect_identity_boxes(gray_frame):
            rects: list[tuple[int, int, int, int]] = []
            if not face_cascade.empty():
                rects.extend(
                    [tuple(map(int, rect)) for rect in face_cascade.detectMultiScale(
                        gray_frame,
                        scaleFactor=1.08,
                        minNeighbors=4,
                        minSize=(30, 30),
                    )]
                )
            if not profile_cascade.empty():
                rects.extend(
                    [tuple(map(int, rect)) for rect in profile_cascade.detectMultiScale(
                        gray_frame,
                        scaleFactor=1.08,
                        minNeighbors=4,
                        minSize=(30, 30),
                    )]
                )
                flipped = cv2.flip(gray_frame, 1)
                width_px = gray_frame.shape[1]
                rects.extend(
                    [
                        (int(width_px - x - w), int(y), int(w), int(h))
                        for x, y, w, h in profile_cascade.detectMultiScale(
                            flipped,
                            scaleFactor=1.08,
                            minNeighbors=4,
                            minSize=(30, 30),
                        )
                    ]
                )
            unique: list[tuple[int, int, int, int]] = []
            for rect in sorted(rects, key=lambda item: item[2] * item[3], reverse=True):
                x, y, w, h = rect
                if any(abs(x - ux) < 10 and abs(y - uy) < 10 and abs(w - uw) < 12 and abs(h - uh) < 12 for ux, uy, uw, uh in unique):
                    continue
                unique.append(rect)
            return unique[:4]

        def soften_identity_regions(gray_frame, rects):
            if not rects:
                return gray_frame
            out = gray_frame.copy()
            height_px, width_px = out.shape[:2]
            for x, y, w, h in rects:
                expand_left = int(round(w * 0.55))
                expand_right = int(round(w * 0.55))
                expand_top = int(round(h * 0.75))
                expand_bottom = int(round(h * 0.45))
                x1 = max(0, x - expand_left)
                y1 = max(0, y - expand_top)
                x2 = min(width_px, x + w + expand_right)
                y2 = min(height_px, y + h + expand_bottom)
                if x2 <= x1 + 4 or y2 <= y1 + 4:
                    continue
                roi = out[y1:y2, x1:x2]
                if roi.size <= 0:
                    continue
                blur = cv2.GaussianBlur(roi, (0, 0), 11.0)
                fill = np.full_like(blur, int(np.clip(float(np.mean(blur)), 92.0, 208.0)))
                flattened = cv2.addWeighted(blur, 0.58, fill, 0.42, 0)
                mask = np.zeros_like(roi, dtype=np.uint8)
                center = (roi.shape[1] // 2, roi.shape[0] // 2)
                axes = (max(8, int(round(roi.shape[1] * 0.46))), max(10, int(round(roi.shape[0] * 0.56))))
                cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1, cv2.LINE_AA)
                mask = cv2.GaussianBlur(mask, (0, 0), 9.0).astype(np.float32) / 255.0
                softened = roi.astype(np.float32) * (1.0 - mask) + flattened.astype(np.float32) * mask
                out[y1:y2, x1:x2] = np.clip(softened, 0, 255).astype(np.uint8)
            return out

        def suppress_subtitle_regions(gray_frame):
            height_px, width_px = gray_frame.shape[:2]
            if height_px <= 0 or width_px <= 0:
                return gray_frame
            search_y = int(round(height_px * 0.48))
            roi = gray_frame[search_y:, :]
            if roi.size <= 0:
                return gray_frame
            bright = (roi > 205).astype(np.uint8) * 255
            if np.count_nonzero(bright) < max(20, int(width_px * height_px * 0.00025)):
                return gray_frame
            bright = cv2.morphologyEx(
                bright,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3)),
                iterations=1,
            )
            bright = cv2.dilate(
                bright,
                cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)),
                iterations=2,
            )
            count, labels, stats, _centroids = cv2.connectedComponentsWithStats(bright, 8)
            mask = np.zeros_like(gray_frame, dtype=np.uint8)
            min_area = max(24, int(width_px * height_px * 0.00004))
            max_area = int(width_px * height_px * 0.045)
            for label in range(1, count):
                x, y, w, h, area = [int(value) for value in stats[label]]
                abs_y = search_y + y
                if area < min_area or area > max_area:
                    continue
                if h > height_px * 0.16:
                    continue
                if abs_y < height_px * 0.50:
                    continue
                if w < max(6, width_px * 0.015):
                    continue
                y1 = max(0, abs_y - 5)
                y2 = min(height_px, abs_y + h + 7)
                x1 = max(0, x - 5)
                x2 = min(width_px, x + w + 5)
                mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], 255)
            if np.count_nonzero(mask) <= 0:
                return gray_frame
            try:
                cleaned = cv2.inpaint(gray_frame, mask, 5, cv2.INPAINT_TELEA)
            except Exception:
                cleaned = cv2.GaussianBlur(gray_frame, (0, 0), 9.0)
            alpha = cv2.GaussianBlur(mask, (0, 0), 3.5).astype(np.float32) / 255.0
            return np.clip(
                gray_frame.astype(np.float32) * (1.0 - alpha) + cleaned.astype(np.float32) * alpha,
                0,
                255,
            ).astype(np.uint8)

        written = 0
        source_index = -1
        try:
            while written < frame_cap:
                ok, frame = capture.read()
                if not ok:
                    break
                source_index += 1
                if source_index < skip_first:
                    continue
                if (source_index - skip_first) % select_every != 0:
                    continue
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.bilateralFilter(gray, 11, 28, 9)
                gray = cv2.GaussianBlur(gray, (0, 0), 0.65)
                local = clahe.apply(gray)
                mix = cv2.addWeighted(gray, 0.62, local, 0.38, 0)
                norm = mix.astype(np.float32) / 255.0

                lifted = 0.52 + 0.44 * np.power(norm, 0.82)
                blur_big = cv2.GaussianBlur(norm, (0, 0), 5.0)
                blur_med = cv2.GaussianBlur(norm, (0, 0), 1.4)
                relief = np.clip((blur_med - blur_big) * 1.6, -0.12, 0.12)
                dx = cv2.Sobel(blur_med, cv2.CV_32F, 1, 0, ksize=3)
                dy = cv2.Sobel(blur_med, cv2.CV_32F, 0, 1, ksize=3)
                grad = np.clip(np.sqrt(dx * dx + dy * dy) * 1.25, 0.0, 1.0)
                grad = cv2.GaussianBlur(grad, (0, 0), 0.9)

                clay = np.clip(lifted + relief - 0.105 * grad, 0.50, 0.985)
                out = np.clip(clay * 255.0, 0, 255).astype(np.uint8)
                out = cv2.GaussianBlur(out, (0, 0), 0.38)
                out = soften_identity_regions(out, detect_identity_boxes(gray))
                out = suppress_subtitle_regions(out)
                writer.write(cv2.cvtColor(out, cv2.COLOR_GRAY2BGR))
                written += 1
        finally:
            capture.release()
            writer.release()

        if written <= 0 or not temp_path.exists() or temp_path.stat().st_size <= 0:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError("gray_relief_output_empty")

        try:
            from .ffmpeg_tools import ffmpeg_path, run_command

            run_command([
                ffmpeg_path(),
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(temp_path),
                "-map",
                "0:v:0",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ])
            temp_path.unlink(missing_ok=True)
        except Exception:
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            shutil.move(str(temp_path), str(output_path))

        if not output_path.exists() or output_path.stat().st_size <= 0:
            raise RuntimeError("gray_relief_output_missing")

    @staticmethod
    def _make_strong_depth_relief_video(depth_path: Path, output_path: Path) -> None:
        import shutil
        import tempfile

        import cv2
        import numpy as np

        capture = cv2.VideoCapture(str(depth_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot_open_depth_video: {depth_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 24.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError("invalid_depth_video_size")

        frames: list[np.ndarray] = []
        samples: list[np.ndarray] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            samples.append(gray.reshape(-1)[:: max(1, gray.size // 4096)])
        capture.release()
        if not frames:
            raise RuntimeError("empty_depth_video")

        sample = np.concatenate(samples).astype(np.float32)
        lo = float(np.percentile(sample, 1.0))
        hi = float(np.percentile(sample, 99.0))
        if hi <= lo + 1.0:
            lo, hi = float(sample.min()), float(sample.max())
        if hi <= lo + 1.0:
            lo, hi = 0.0, 255.0

        clahe = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(8, 8))
        light_key = np.asarray([-0.36, -0.54, 0.76], dtype=np.float32)
        light_key = light_key / max(1e-6, float(np.linalg.norm(light_key)))
        light_fill = np.asarray([0.44, 0.20, 0.88], dtype=np.float32)
        light_fill = light_fill / max(1e-6, float(np.linalg.norm(light_fill)))
        light_rim = np.asarray([0.62, -0.20, 0.76], dtype=np.float32)
        light_rim = light_rim / max(1e-6, float(np.linalg.norm(light_rim)))
        normal_strength = max(5.0, min(10.0, 4600.0 / max(1.0, float(min(width, height)))))
        previous_relief = None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = Path(tempfile.mktemp(prefix=f".{output_path.stem}_", suffix=".mp4", dir=str(output_path.parent)))
        writer = cv2.VideoWriter(
            str(temp_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("cannot_open_strong_relief_writer")

        try:
            for gray in frames:
                norm = (gray.astype(np.float32) - lo) / max(1.0, hi - lo)
                norm = np.clip(norm, 0.0, 1.0)
                norm_u8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
                local = clahe.apply(norm_u8).astype(np.float32) / 255.0
                norm = np.clip(0.72 * norm + 0.28 * local, 0.0, 1.0)
                norm = cv2.GaussianBlur(norm.astype(np.float32), (0, 0), 0.45)
                norm = cv2.bilateralFilter(norm.astype(np.float32), 7, 0.045, 5.0)

                large = cv2.GaussianBlur(norm, (0, 0), 3.2)
                medium = cv2.GaussianBlur(norm, (0, 0), 1.15)
                detail = np.clip((medium - large) * 1.9, -0.14, 0.14)
                relief = np.clip(norm + detail, 0.0, 1.0)
                if previous_relief is not None:
                    relief = np.clip(0.84 * relief + 0.16 * previous_relief, 0.0, 1.0)
                previous_relief = relief.copy()

                relief_for_normal = cv2.GaussianBlur(relief, (0, 0), 0.70)
                dx = cv2.Scharr(relief_for_normal, cv2.CV_32F, 1, 0) / 32.0
                dy = cv2.Scharr(relief_for_normal, cv2.CV_32F, 0, 1) / 32.0
                nx = -dx * normal_strength
                ny = -dy * normal_strength
                nz = np.ones_like(relief_for_normal, dtype=np.float32)
                length = np.sqrt(nx * nx + ny * ny + nz * nz)
                nx /= np.maximum(length, 1e-6)
                ny /= np.maximum(length, 1e-6)
                nz /= np.maximum(length, 1e-6)

                key = np.clip(nx * light_key[0] + ny * light_key[1] + nz * light_key[2], 0.0, 1.0)
                fill = np.clip(nx * light_fill[0] + ny * light_fill[1] + nz * light_fill[2], 0.0, 1.0)
                rim = np.clip(nx * light_rim[0] + ny * light_rim[1] + nz * light_rim[2], 0.0, 1.0)
                lap = cv2.Laplacian(relief_for_normal, cv2.CV_32F, ksize=3)
                cavities = cv2.GaussianBlur(np.clip(-lap * 1.9, 0.0, 1.0), (0, 0), 1.35)
                ridges = cv2.GaussianBlur(np.clip(lap * 1.2, 0.0, 1.0), (0, 0), 0.85)
                slope = cv2.GaussianBlur(np.clip(np.sqrt(dx * dx + dy * dy) * 1.6, 0.0, 1.0), (0, 0), 0.9)

                albedo = 0.83 + 0.12 * np.power(relief, 0.76)
                lighting = 0.58 + 0.34 * key + 0.12 * fill + 0.10 * np.power(rim, 1.7)
                clay = albedo * lighting
                clay -= 0.09 * cavities
                clay -= 0.035 * slope
                clay += 0.025 * ridges
                clay = np.clip(clay, 0.57, 1.0)
                clay_u8 = np.clip(clay * 255.0, 0, 255).astype(np.uint8)
                smooth = cv2.GaussianBlur(clay_u8, (0, 0), 0.38)
                clay_u8 = cv2.addWeighted(clay_u8, 1.12, smooth, -0.12, 0)
                writer.write(cv2.cvtColor(clay_u8, cv2.COLOR_GRAY2BGR))
        finally:
            writer.release()

        if not temp_path.exists() or temp_path.stat().st_size <= 0:
            raise RuntimeError("strong_relief_temp_output_missing")

        try:
            from .ffmpeg_tools import ffmpeg_path, run_command

            run_command([
                ffmpeg_path(),
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(temp_path),
                "-map",
                "0:v:0",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ])
            temp_path.unlink(missing_ok=True)
        except Exception:
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            shutil.move(str(temp_path), str(output_path))

        if not output_path.exists() or output_path.stat().st_size <= 0:
            raise RuntimeError("strong_relief_output_missing")

    @staticmethod
    def _make_capsule_control_video(
        *,
        pose_path: Path,
        output_path: Path,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> None:
        import shutil
        import tempfile

        import cv2
        import numpy as np

        def log(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        capture = cv2.VideoCapture(str(pose_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot_open_pose_video: {pose_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 24.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError("invalid_pose_video_size")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = Path(tempfile.mktemp(prefix=f".{output_path.stem}_", suffix=".mp4", dir=str(output_path.parent)))
        writer = cv2.VideoWriter(
            str(temp_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("cannot_open_capsule_writer")

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        kernel_blur = (0, 0)
        last_mask: np.ndarray | None = None
        written = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mask = (gray > 12).astype(np.uint8) * 255
                if np.count_nonzero(mask) > 0:
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
                    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1)
                    last_mask = mask
                elif last_mask is not None:
                    mask = last_mask
                else:
                    mask = np.zeros((height, width), dtype=np.uint8)

                alpha = cv2.GaussianBlur(mask, (0, 0), 11.0).astype(np.float32) / 255.0
                alpha = np.clip(alpha * 0.92, 0.0, 0.92)
                y_grad = np.linspace(0.86, 1.05, height, dtype=np.float32).reshape(height, 1)
                x_grad = np.linspace(0.94, 1.04, width, dtype=np.float32).reshape(1, width)
                lighting = np.clip(y_grad * x_grad, 0.80, 1.08)
                body = np.dstack([
                    np.full((height, width), 218.0, dtype=np.float32),
                    np.full((height, width), 216.0, dtype=np.float32),
                    np.full((height, width), 214.0, dtype=np.float32),
                ]) * lighting[..., None]
                core = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
                core_alpha = cv2.GaussianBlur(core, (0, 0), 7.0).astype(np.float32) / 255.0
                core_alpha = np.clip(core_alpha * 0.55, 0.0, 0.55)
                bg = np.dstack([
                    np.full((height, width), 22.0, dtype=np.float32),
                    np.full((height, width), 22.0, dtype=np.float32),
                    np.full((height, width), 24.0, dtype=np.float32),
                ])
                out = bg * (1.0 - alpha[..., None]) + body * alpha[..., None]
                out = out * (1.0 - core_alpha[..., None]) + 245.0 * core_alpha[..., None]
                edge = cv2.Canny(mask, 40, 120)
                edge = cv2.GaussianBlur(edge, kernel_blur, 0.8).astype(np.float32) / 255.0
                out = np.clip(out - edge[..., None] * 18.0, 0, 255).astype(np.uint8)
                writer.write(out)
                written += 1
        finally:
            capture.release()
            writer.release()

        if written <= 0 or not temp_path.exists() or temp_path.stat().st_size <= 0:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError("capsule_control_output_empty")

        try:
            from .ffmpeg_tools import ffmpeg_path, run_command

            run_command([
                ffmpeg_path(),
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(temp_path),
                "-map",
                "0:v:0",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ])
            temp_path.unlink(missing_ok=True)
        except Exception:
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            shutil.move(str(temp_path), str(output_path))

        if not output_path.exists() or output_path.stat().st_size <= 0:
            raise RuntimeError("capsule_control_output_missing")

    @staticmethod
    def _make_face_expression_guidance_video(
        video_path: Path,
        base_path: Path,
        output_path: Path,
        *,
        body_pose_path: Path | None = None,
        max_faces: int = 6,
        include_mouth: bool = True,
        color_faces: bool = True,
        include_eyes: bool = True,
        include_brows: bool = True,
        include_head_pose: bool = True,
        include_face_outline: bool = True,
        include_soft_face_relief: bool = False,
        face_relief: bool | None = None,
        safe_mode: bool = True,
        body_color_mode: str = "none",
        include_body_pose: bool = False,
        pose_render_style: str = "capsule",
        pose_capsule_strength: str = "strong",
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> dict:
        import shutil
        import tempfile

        import cv2
        import numpy as np

        def log(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        if face_relief is not None:
            include_soft_face_relief = bool(include_soft_face_relief or face_relief)

        try:
            from .face_identity import _get_face_app
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"insightface_unavailable: {exc}") from exc

        source_capture = cv2.VideoCapture(str(video_path))
        if not source_capture.isOpened():
            raise RuntimeError(f"cannot_open_expression_source_video: {video_path}")
        base_capture = cv2.VideoCapture(str(base_path))
        if not base_capture.isOpened():
            source_capture.release()
            raise RuntimeError(f"cannot_open_expression_base_video: {base_path}")
        pose_capture = None
        if include_body_pose and body_pose_path and Path(body_pose_path).exists():
            pose_capture = cv2.VideoCapture(str(body_pose_path))
            if not pose_capture.isOpened():
                log(f"cannot_open_body_pose_video: {body_pose_path}")
                pose_capture = None

        fps = float(base_capture.get(cv2.CAP_PROP_FPS) or source_capture.get(cv2.CAP_PROP_FPS) or 0.0) or 24.0
        base_width = int(base_capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        base_height = int(base_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        source_width = int(source_capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        source_height = int(source_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if base_width <= 0 or base_height <= 0 or source_width <= 0 or source_height <= 0:
            source_capture.release()
            base_capture.release()
            if pose_capture is not None:
                pose_capture.release()
            raise RuntimeError("invalid_expression_video_size")

        scale_x = base_width / max(1, source_width)
        scale_y = base_height / max(1, source_height)
        max_faces = max(1, min(int(max_faces or 1), 6))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = Path(tempfile.mktemp(prefix=f".{output_path.stem}_", suffix=".mp4", dir=str(output_path.parent)))
        writer = cv2.VideoWriter(
            str(temp_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (base_width, base_height),
        )
        if not writer.isOpened():
            source_capture.release()
            base_capture.release()
            if pose_capture is not None:
                pose_capture.release()
            raise RuntimeError("cannot_open_expression_writer")

        app = _get_face_app()
        frame_count = 0
        face_hit_frames = 0
        face_overlays = 0
        silhouette_head_frames = 0
        face_palette = [
            (255, 120, 40),
            (40, 80, 255),
            (80, 220, 80),
            (40, 220, 240),
            (220, 80, 220),
            (60, 160, 255),
        ]
        face_tracks: list[dict[str, Any]] = []
        next_track_id = 0
        body_color_mode = str(body_color_mode or "none").strip().lower()
        body_colors_enabled = body_color_mode not in {"", "none", "off", "false", "0"}
        pose_render_style = str(pose_render_style or "capsule").strip().lower() or "capsule"
        pose_render_style = {
            "clean_white": "clean",
            "white": "clean",
            "lowpoly": "capsule",
            "low_poly": "capsule",
            "low_model": "capsule",
            "mannequin": "capsule",
            "capsule_mannequin": "capsule",
            "debug_skeleton": "debug_lines",
            "skeleton": "debug_lines",
            "lines": "debug_lines",
            "debug": "debug_lines",
            "debug_skeleton": "debug_lines",
            "none": "clean",
            "off": "clean",
            "false": "clean",
            "0": "clean",
        }.get(pose_render_style, pose_render_style)
        if pose_render_style not in {"capsule", "debug_lines", "clean"}:
            pose_render_style = "capsule"
        pose_capsule_strength = str(pose_capsule_strength or "strong").strip().lower() or "strong"
        pose_capsule_strength = {
            "weak": "light",
            "soft": "light",
            "low": "light",
            "medium": "normal",
            "standard": "normal",
            "default": "normal",
            "high": "strong",
            "heavy": "strong",
            "max": "strong",
        }.get(pose_capsule_strength, pose_capsule_strength)
        if pose_capsule_strength not in {"light", "normal", "strong"}:
            pose_capsule_strength = "strong"
        capsule_strength_settings = {
            "light": {
                "radius_scale": 0.016,
                "radius_min": 6,
                "radius_max": 22,
                "soft_alpha": 0.46,
                "core_alpha": 0.12,
                "shadow_alpha": 0.18,
                "fill_base": 144.0,
                "fill_luma": 78.0,
                "core_value": 238.0,
            },
            "normal": {
                "radius_scale": 0.018,
                "radius_min": 7,
                "radius_max": 28,
                "soft_alpha": 0.62,
                "core_alpha": 0.18,
                "shadow_alpha": 0.24,
                "fill_base": 152.0,
                "fill_luma": 84.0,
                "core_value": 245.0,
            },
            "strong": {
                "radius_scale": 0.026,
                "radius_min": 10,
                "radius_max": 42,
                "soft_alpha": 0.82,
                "core_alpha": 0.26,
                "shadow_alpha": 0.30,
                "fill_base": 166.0,
                "fill_luma": 88.0,
                "core_value": 252.0,
            },
        }
        body_color_frames = 0
        body_pose_frames = 0
        body_palette = [
            (52, 58, 220),   # red in BGR
            (72, 190, 70),   # green in BGR
            (225, 135, 52),  # blue/orange fallback colors after red/green
            (52, 196, 225),
            (190, 72, 190),
            (92, 160, 240),
        ]

        def prune_tracks(frame_index: int) -> None:
            face_tracks[:] = [
                track for track in face_tracks
                if frame_index - int(track.get("last_frame", frame_index)) <= 24
            ]

        def assign_track(face_bbox, frame_index: int) -> dict[str, Any]:
            nonlocal next_track_id
            x1, y1, x2, y2 = [float(value) for value in list(face_bbox)[:4]]
            center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
            face_w = max(1.0, x2 - x1)
            face_h = max(1.0, y2 - y1)
            match_radius = max(24.0, 0.42 * max(face_w, face_h))
            match_radius_sq = match_radius * match_radius
            best_index = None
            best_dist = None
            for index, track in enumerate(face_tracks):
                if frame_index - int(track.get("last_frame", frame_index)) > 24:
                    continue
                raw_center = track.get("center")
                track_center = np.asarray(raw_center if raw_center is not None else center, dtype=np.float32)
                dx = float(center[0] - track_center[0])
                dy = float(center[1] - track_center[1])
                dist_sq = dx * dx + dy * dy
                if dist_sq > match_radius_sq:
                    continue
                if best_dist is None or dist_sq < best_dist:
                    best_index = index
                    best_dist = dist_sq
            if best_index is None:
                track = {
                    "track_id": next_track_id,
                    "center": center,
                    "last_frame": frame_index,
                    "color_index": next_track_id % len(face_palette),
                }
                face_tracks.append(track)
                next_track_id += 1
                return track
            track = face_tracks[best_index]
            track["center"] = center
            track["last_frame"] = frame_index
            return track

        def draw_direction_badge(frame, text: str, anchor, color) -> None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.72
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
            x = int(round(anchor[0]))
            y = int(round(anchor[1]))
            x = max(6, min(base_width - text_w - 10, x))
            y = max(text_h + 8, min(base_height - baseline - 8, y))
            top_left = (x - 5, y - text_h - 5)
            bottom_right = (x + text_w + 5, y + baseline + 5)
            cv2.rectangle(frame, top_left, bottom_right, (18, 18, 18), -1)
            cv2.rectangle(frame, top_left, bottom_right, color, 1, cv2.LINE_AA)
            cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

        def apply_body_tint(frame, base_frame, mask, color) -> tuple[Any, bool]:
            if mask is None or not np.any(mask):
                return frame, False
            alpha = cv2.GaussianBlur(mask.astype(np.uint8), (17, 17), 0).astype(np.float32) / 255.0
            alpha = np.clip(alpha * 0.72, 0.0, 0.72)[..., None]
            luma = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            color_arr = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
            shaded = color_arr * (0.36 + 0.64 * luma[..., None])
            mixed = frame.astype(np.float32) * (1.0 - alpha) + shaded * alpha
            return np.clip(mixed, 0, 255).astype(np.uint8), True

        def body_mask_from_face(base_frame, bbox) -> Any:
            bbox = np.asarray(bbox, dtype=np.float32)[:4].copy()
            if bbox.shape[0] < 4:
                return None
            bbox[0] *= scale_x
            bbox[1] *= scale_y
            bbox[2] *= scale_x
            bbox[3] *= scale_y
            x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
            face_w = max(1.0, x2 - x1)
            face_h = max(1.0, y2 - y1)
            cx = (x1 + x2) / 2.0
            bx1 = max(0, int(round(cx - 2.2 * face_w)))
            bx2 = min(base_width, int(round(cx + 2.2 * face_w)))
            by1 = max(0, int(round(y1 - 0.35 * face_h)))
            by2 = min(base_height, int(round(y2 + max(3.4 * face_h, 0.22 * base_height))))
            if bx2 <= bx1 + 4 or by2 <= by1 + 4:
                return None

            gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
            roi = gray[by1:by2, bx1:bx2]
            if roi.size <= 0:
                return None
            blurred = cv2.GaussianBlur(roi, (5, 5), 0)
            threshold = max(24.0, float(np.percentile(roi, 46.0)))
            candidate = (blurred >= threshold).astype(np.uint8) * 255
            kernel = np.ones((5, 5), np.uint8)
            candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
            candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)

            seed = np.zeros_like(candidate)
            sx1 = max(0, int(round(x1 - 0.30 * face_w)) - bx1)
            sx2 = min(candidate.shape[1], int(round(x2 + 0.30 * face_w)) - bx1)
            sy1 = max(0, int(round(y1 - 0.12 * face_h)) - by1)
            sy2 = min(candidate.shape[0], int(round(y2 + 1.25 * face_h)) - by1)
            if sx2 > sx1 and sy2 > sy1:
                seed[sy1:sy2, sx1:sx2] = 255

            count, labels, stats, _centroids = cv2.connectedComponentsWithStats(candidate, 8)
            local_mask = np.zeros_like(candidate)
            min_area = max(90, int(base_width * base_height * 0.0012))
            for label in range(1, count):
                area = int(stats[label, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue
                component = labels == label
                if np.count_nonzero(seed[component]) <= 0:
                    continue
                local_mask[component] = 255

            if np.count_nonzero(local_mask) < min_area:
                center = (int(round(cx)) - bx1, int(round(y1 + 1.7 * face_h)) - by1)
                axes = (
                    max(8, int(round(1.18 * face_w))),
                    max(12, int(round(2.35 * face_h))),
                )
                cv2.ellipse(local_mask, center, axes, 0, 0, 360, 255, -1, cv2.LINE_AA)

            mask = np.zeros((base_height, base_width), dtype=np.uint8)
            mask[by1:by2, bx1:bx2] = local_mask
            return mask

        def largest_body_mask(base_frame) -> Any:
            gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            threshold = max(24.0, float(np.percentile(gray, 58.0)))
            candidate = (blurred >= threshold).astype(np.uint8) * 255
            kernel = np.ones((5, 5), np.uint8)
            candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
            candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)
            count, labels, stats, _centroids = cv2.connectedComponentsWithStats(candidate, 8)
            best_label = None
            best_area = 0
            min_area = max(180, int(base_width * base_height * 0.006))
            for label in range(1, count):
                x, y, w, h, area = [int(value) for value in stats[label]]
                if area < min_area or w < base_width * 0.05 or h < base_height * 0.08:
                    continue
                score = area * (1.15 if y < base_height * 0.78 else 0.75)
                if score > best_area:
                    best_label = label
                    best_area = score
            if best_label is None:
                return None
            return np.where(labels == best_label, 255, 0).astype(np.uint8)

        def overlay_body_pose(frame, pose_frame):
            if pose_frame is None:
                return frame, False
            if pose_frame.shape[1] != base_width or pose_frame.shape[0] != base_height:
                pose_frame = cv2.resize(pose_frame, (base_width, base_height), interpolation=cv2.INTER_LINEAR)
            # DWPose videos are black-background control maps. Keep only visible lines/keypoints.
            visible = np.max(pose_frame, axis=2) > 22
            if np.count_nonzero(visible) < max(20, int(base_width * base_height * 0.0002)):
                return frame, False
            mask = (visible.astype(np.uint8) * 255)
            if pose_render_style == "clean":
                return frame, False
            if pose_render_style != "debug_lines":
                strength = capsule_strength_settings.get(pose_capsule_strength, capsule_strength_settings["strong"])
                radius = max(
                    int(strength["radius_min"]),
                    min(
                        int(strength["radius_max"]),
                        int(round(min(base_width, base_height) * float(strength["radius_scale"]))),
                    ),
                )
                capsule_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (radius * 2 + 1, radius * 2 + 1),
                )
                close_radius = max(2, radius // 2)
                close_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (close_radius * 2 + 1, close_radius * 2 + 1),
                )
                capsule = cv2.dilate(mask, capsule_kernel, iterations=1)
                capsule = cv2.morphologyEx(capsule, cv2.MORPH_CLOSE, close_kernel, iterations=1)
                if np.count_nonzero(capsule) < max(30, int(base_width * base_height * 0.0003)):
                    return frame, False

                def odd_kernel(value: float, cap: int = 99) -> int:
                    size = int(round(max(3.0, value)))
                    if size % 2 == 0:
                        size += 1
                    return min(size, cap)

                soft_size = odd_kernel(radius * 1.8)
                soft = cv2.GaussianBlur(capsule, (soft_size, soft_size), 0).astype(np.float32) / 255.0
                core_radius = max(3, radius // 2)
                core_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (core_radius * 2 + 1, core_radius * 2 + 1),
                )
                core = cv2.dilate(mask, core_kernel, iterations=1)
                core_size = odd_kernel(max(3, core_radius * 1.2))
                core = cv2.GaussianBlur(core, (core_size, core_size), 0).astype(np.float32) / 255.0
                edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                edge = cv2.subtract(capsule, cv2.erode(capsule, edge_kernel, iterations=1))
                edge = cv2.GaussianBlur(edge, (7, 7), 0).astype(np.float32) / 255.0

                frame_f = frame.astype(np.float32)
                luma = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                fill_gray = np.clip(
                    float(strength["fill_base"]) + luma * float(strength["fill_luma"]),
                    0.0,
                    float(strength["core_value"]),
                )
                fill = np.dstack([fill_gray, fill_gray, fill_gray]).astype(np.float32)

                shadow_limit = float(strength["shadow_alpha"])
                shadow_alpha = np.clip(edge * shadow_limit, 0.0, shadow_limit)[..., None]
                shadowed = frame_f * (1.0 - shadow_alpha) + 28.0 * shadow_alpha
                soft_limit = float(strength["soft_alpha"])
                capsule_alpha = np.clip(soft * soft_limit, 0.0, soft_limit)[..., None]
                mixed = shadowed * (1.0 - capsule_alpha) + fill * capsule_alpha
                core_limit = float(strength["core_alpha"])
                core_alpha = np.clip(core * core_limit, 0.0, core_limit)[..., None]
                mixed = mixed * (1.0 - core_alpha) + float(strength["core_value"]) * core_alpha
                return np.clip(mixed, 0, 255).astype(np.uint8), True
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
            alpha = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0
            alpha = np.clip(alpha * 0.92, 0.0, 0.92)[..., None]
            mixed = frame.astype(np.float32) * (1.0 - alpha) + pose_frame.astype(np.float32) * alpha
            return np.clip(mixed, 0, 255).astype(np.uint8), True

        def draw_face_guidance(frame, landmarks, bbox, keypoints, mouth_y, face_color):
            points = np.asarray(landmarks, dtype=np.float32)
            if points.ndim != 2 or points.shape[0] < 6 or points.shape[1] < 2:
                return frame
            points = points[:, :2].copy()
            points[:, 0] *= scale_x
            points[:, 1] *= scale_y
            bbox = np.asarray(bbox, dtype=np.float32)[:4].copy()
            if bbox.shape[0] < 4:
                return frame
            bbox[0] *= scale_x
            bbox[1] *= scale_y
            bbox[2] *= scale_x
            bbox[3] *= scale_y
            x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
            face_w = max(1.0, x2 - x1)
            face_h = max(1.0, y2 - y1)
            mouth_y_scaled = float(mouth_y) * scale_y
            in_bbox = (
                (points[:, 0] >= x1 - 2.0)
                & (points[:, 0] <= x2 + 2.0)
                & (points[:, 1] >= y1 - 2.0)
                & (points[:, 1] <= y2 + 2.0)
            )
            points = points[in_bbox]
            points = points[
                (points[:, 0] >= 0)
                & (points[:, 0] < base_width)
                & (points[:, 1] >= 0)
                & (points[:, 1] < base_height)
            ]
            overlay = frame.copy()
            line_color = (18, 18, 18) if safe_mode else (face_color if color_faces else (18, 18, 18))
            point_color = (8, 8, 8) if safe_mode else (face_color if color_faces else (8, 8, 8))
            mouth_color = tuple(int(min(255, channel * 0.95 + 18)) for channel in line_color)
            fine_color = tuple(int(max(0, channel * 0.72)) for channel in line_color)
            drawn = False

            scaled_kps = None
            if keypoints is not None and len(keypoints) >= 5:
                scaled_kps = np.asarray(keypoints, dtype=np.float32)[:5, :2].copy()
                scaled_kps[:, 0] *= scale_x
                scaled_kps[:, 1] *= scale_y

            def apply_soft_face_relief(relief_frame):
                shadow = np.zeros((base_height, base_width), dtype=np.float32)
                highlight = np.zeros((base_height, base_width), dtype=np.float32)

                def odd_kernel(value: float) -> int:
                    size = int(round(max(7.0, value)))
                    if size % 2 == 0:
                        size += 1
                    return min(size, 99)

                def add_soft_blob(target, center, axes, angle=0.0, strength=0.12, blur_scale=0.34):
                    cx, cy = float(center[0]), float(center[1])
                    ax = max(2, int(round(float(axes[0]))))
                    ay = max(2, int(round(float(axes[1]))))
                    if cx < -ax or cx > base_width + ax or cy < -ay or cy > base_height + ay:
                        return False
                    mask = np.zeros((base_height, base_width), dtype=np.uint8)
                    cv2.ellipse(
                        mask,
                        (int(round(cx)), int(round(cy))),
                        (ax, ay),
                        float(angle),
                        0,
                        360,
                        255,
                        -1,
                        cv2.LINE_AA,
                    )
                    kernel = odd_kernel(max(ax, ay) * blur_scale)
                    blurred = cv2.GaussianBlur(mask, (kernel, kernel), 0).astype(np.float32) / 255.0
                    np.maximum(target, np.clip(blurred * float(strength), 0.0, 0.34), out=target)
                    return True

                def angle_between(a, b) -> float:
                    delta = np.asarray(b, dtype=np.float32) - np.asarray(a, dtype=np.float32)
                    return float(np.degrees(np.arctan2(float(delta[1]), float(delta[0]))))

                drawn_relief = False
                if scaled_kps is not None:
                    left_eye, right_eye, nose, left_mouth, right_mouth = scaled_kps
                    eye_angle = angle_between(left_eye, right_eye)
                    mouth_angle = angle_between(left_mouth, right_mouth)
                    mouth_center = (left_mouth + right_mouth) / 2.0
                    eye_gap = float(np.linalg.norm(right_eye - left_eye)) or max(1.0, face_w * 0.24)

                    if include_eyes:
                        for eye in (left_eye, right_eye):
                            drawn_relief = add_soft_blob(
                                shadow,
                                eye,
                                (max(3.0, 0.13 * face_w), max(2.0, 0.035 * face_h)),
                                eye_angle,
                                0.115,
                                0.46,
                            ) or drawn_relief
                            drawn_relief = add_soft_blob(
                                highlight,
                                (eye[0], eye[1] - 0.045 * face_h),
                                (max(3.0, 0.09 * face_w), max(2.0, 0.025 * face_h)),
                                eye_angle,
                                0.050,
                                0.62,
                            ) or drawn_relief

                    if include_brows:
                        for eye in (left_eye, right_eye):
                            drawn_relief = add_soft_blob(
                                shadow,
                                (eye[0], eye[1] - 0.105 * face_h),
                                (max(3.0, 0.12 * face_w), max(2.0, 0.025 * face_h)),
                                eye_angle - 4.0,
                                0.080,
                                0.58,
                            ) or drawn_relief

                    # Nose is a soft raised plane and under-nose shadow, not a direction marker.
                    drawn_relief = add_soft_blob(
                        highlight,
                        (nose[0], nose[1] - 0.020 * face_h),
                        (max(2.0, 0.040 * face_w), max(4.0, 0.110 * face_h)),
                        0.0,
                        0.045,
                        0.74,
                    ) or drawn_relief
                    drawn_relief = add_soft_blob(
                        shadow,
                        (nose[0], nose[1] + 0.105 * face_h),
                        (max(2.0, 0.055 * face_w), max(2.0, 0.030 * face_h)),
                        0.0,
                        0.070,
                        0.66,
                    ) or drawn_relief

                    if include_mouth:
                        mouth_width = max(eye_gap * 0.42, float(np.linalg.norm(right_mouth - left_mouth)) * 1.18)
                        drawn_relief = add_soft_blob(
                            shadow,
                            mouth_center,
                            (max(4.0, 0.5 * mouth_width), max(2.0, 0.028 * face_h)),
                            mouth_angle,
                            0.125,
                            0.58,
                        ) or drawn_relief
                        drawn_relief = add_soft_blob(
                            highlight,
                            (mouth_center[0], mouth_center[1] - 0.045 * face_h),
                            (max(3.0, 0.38 * mouth_width), max(2.0, 0.020 * face_h)),
                            mouth_angle,
                            0.038,
                            0.75,
                        ) or drawn_relief
                else:
                    center_x = (x1 + x2) / 2.0
                    if include_eyes:
                        for dx in (-0.18, 0.18):
                            drawn_relief = add_soft_blob(
                                shadow,
                                (center_x + dx * face_w, y1 + 0.38 * face_h),
                                (0.10 * face_w, 0.035 * face_h),
                                0.0,
                                0.09,
                                0.52,
                            ) or drawn_relief
                    if include_mouth:
                        drawn_relief = add_soft_blob(
                            shadow,
                            (center_x, mouth_y_scaled),
                            (0.18 * face_w, 0.032 * face_h),
                            0.0,
                            0.10,
                            0.60,
                        ) or drawn_relief

                if not drawn_relief:
                    return relief_frame
                relief = relief_frame.astype(np.float32)
                relief = relief * (1.0 - np.clip(shadow, 0.0, 0.28)[..., None])
                relief = relief + np.clip(highlight, 0.0, 0.18)[..., None] * 34.0
                relief = cv2.GaussianBlur(relief, (3, 3), 0)
                return np.clip(relief, 0, 255).astype(np.uint8)

            if include_soft_face_relief:
                return apply_soft_face_relief(frame)

            upper_points = np.empty((0, 2), dtype=np.float32)
            if include_face_outline:
                face_center = (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0)))
                face_axes = (
                    int(round(max(8.0, face_w * 0.47))),
                    int(round(max(10.0, face_h * 0.52))),
                )
                cv2.ellipse(overlay, face_center, face_axes, 0, 0, 360, line_color, 1, cv2.LINE_AA)
                drawn = True

                upper_limit = mouth_y_scaled - max(2.0, 0.025 * face_h)
                upper_points = points[points[:, 1] <= upper_limit]

            if include_face_outline and len(upper_points) >= 6:
                max_link_distance = (max(face_w, face_h) * 0.14) ** 2
                for index, point in enumerate(upper_points):
                    distances = np.sum((upper_points - point) ** 2, axis=1)
                    neighbors = np.argsort(distances)[1:4]
                    p1 = (int(round(point[0])), int(round(point[1])))
                    for neighbor in neighbors:
                        if distances[neighbor] >= max_link_distance:
                            continue
                        p2 = (int(round(upper_points[neighbor][0])), int(round(upper_points[neighbor][1])))
                        cv2.line(overlay, p1, p2, line_color, 1, cv2.LINE_AA)
                    if index % 2 == 0:
                        cv2.circle(overlay, p1, 2, point_color, -1, cv2.LINE_AA)
                hull = cv2.convexHull(upper_points.astype(np.int32))
                cv2.polylines(overlay, [hull], True, line_color, 2, cv2.LINE_AA)
                drawn = True

            if scaled_kps is not None and include_head_pose:
                left_eye, right_eye, nose, left_mouth, right_mouth = scaled_kps
                eye_center = (left_eye + right_eye) / 2.0
                mouth_center = (left_mouth + right_mouth) / 2.0
                eye_dist = float(np.linalg.norm(right_eye - left_eye)) or max(1.0, face_w * 0.22)
                yaw = float((nose[0] - eye_center[0]) / max(1.0, eye_dist))
                yaw = max(-0.65, min(0.65, yaw))
                cv2.line(
                    overlay,
                    (int(round(left_eye[0])), int(round(left_eye[1]))),
                    (int(round(right_eye[0])), int(round(right_eye[1]))),
                    fine_color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.line(
                    overlay,
                    (int(round(eye_center[0])), int(round(eye_center[1]))),
                    (int(round(nose[0])), int(round(nose[1]))),
                    fine_color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.line(
                    overlay,
                    (int(round(nose[0])), int(round(nose[1]))),
                    (int(round(mouth_center[0])), int(round(mouth_center[1]))),
                    fine_color,
                    1,
                    cv2.LINE_AA,
                )
                if not safe_mode:
                    arrow_end = (
                        int(round(nose[0] + yaw * 0.26 * face_w)),
                        int(round(nose[1] - 0.10 * face_h)),
                    )
                    cv2.arrowedLine(
                        overlay,
                        (int(round(nose[0])), int(round(nose[1]))),
                        arrow_end,
                        line_color,
                        2,
                        cv2.LINE_AA,
                        tipLength=0.35,
                    )
                    draw_direction_badge(
                        overlay,
                        "HEAD R" if yaw >= 0 else "HEAD L",
                        (
                            nose[0] + (0.16 if yaw >= 0 else -0.42) * face_w,
                            nose[1] - 0.24 * face_h,
                        ),
                        line_color,
                    )
                    for point in (left_eye, right_eye, nose):
                        cv2.circle(overlay, (int(round(point[0])), int(round(point[1]))), 3, line_color, -1, cv2.LINE_AA)
                drawn = True

            if scaled_kps is not None and (include_eyes or include_brows):
                left_eye, right_eye = scaled_kps[0], scaled_kps[1]
                eye_centers = (left_eye, right_eye)
                for eye in eye_centers:
                    eye_mask = (
                        (np.abs(points[:, 0] - eye[0]) <= 0.18 * face_w)
                        & (np.abs(points[:, 1] - eye[1]) <= 0.12 * face_h)
                    )
                    eye_points = points[eye_mask]
                    if include_eyes and len(eye_points) >= 4:
                        eye_hull = cv2.convexHull(eye_points.astype(np.int32))
                        cv2.polylines(overlay, [eye_hull], True, line_color, 2, cv2.LINE_AA)
                        cv2.circle(
                            overlay,
                            (int(round(eye[0])), int(round(eye[1]))),
                            max(2, int(round(face_w * 0.012))),
                            point_color,
                            -1,
                            cv2.LINE_AA,
                        )
                        drawn = True

                    brow_mask = (
                        (np.abs(points[:, 0] - eye[0]) <= 0.20 * face_w)
                        & (points[:, 1] >= eye[1] - 0.20 * face_h)
                        & (points[:, 1] <= eye[1] - 0.035 * face_h)
                    )
                    brow_points = points[brow_mask]
                    if include_brows and len(brow_points) >= 3:
                        brow_points = brow_points[np.argsort(brow_points[:, 0])]
                        step = max(1, len(brow_points) // 7)
                        brow_poly = brow_points[::step][:8].astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(overlay, [brow_poly], False, line_color, 2, cv2.LINE_AA)
                        drawn = True
                    elif include_brows:
                        y = int(round(eye[1] - 0.085 * face_h))
                        x_left = int(round(eye[0] - 0.12 * face_w))
                        x_right = int(round(eye[0] + 0.12 * face_w))
                        cv2.line(overlay, (x_left, y), (x_right, y - int(round(0.018 * face_h))), line_color, 2, cv2.LINE_AA)
                        drawn = True

            if include_mouth:
                mouth_center_x = float((x1 + x2) / 2.0)
                if scaled_kps is not None:
                    mouth_center_x = float((float(scaled_kps[3][0]) + float(scaled_kps[4][0])) / 2.0)
                mouth_mask = (
                    (points[:, 1] >= mouth_y_scaled - 0.05 * face_h)
                    & (points[:, 1] <= mouth_y_scaled + 0.22 * face_h)
                    & (points[:, 0] >= mouth_center_x - 0.22 * face_w)
                    & (points[:, 0] <= mouth_center_x + 0.22 * face_w)
                )
                mouth_points = points[mouth_mask]
                if len(mouth_points) >= 3:
                    center = np.array([mouth_center_x, mouth_y_scaled], dtype=np.float32)
                    scale = np.array([max(1.0, 0.22 * face_w), max(1.0, 0.18 * face_h)], dtype=np.float32)
                    distances = np.sum(((mouth_points - center) / scale) ** 2, axis=1)
                    mouth_points = mouth_points[np.argsort(distances)[: min(len(mouth_points), 24)]]
                    if len(mouth_points) >= 3:
                        mouth_hull = cv2.convexHull(mouth_points.astype(np.int32))
                        cv2.polylines(overlay, [mouth_hull], True, mouth_color, 2, cv2.LINE_AA)
                        if len(mouth_points) >= 5:
                            mouth_span_x = float(np.max(mouth_points[:, 0]) - np.min(mouth_points[:, 0]))
                            mouth_span_y = float(np.max(mouth_points[:, 1]) - np.min(mouth_points[:, 1]))
                            axes = (
                                int(round(max(8.0, mouth_span_x * 0.55))),
                                int(round(max(5.0, mouth_span_y * 0.75))),
                            )
                            cv2.ellipse(
                                overlay,
                                (int(round(mouth_center_x)), int(round(mouth_y_scaled))),
                                axes,
                                0,
                                0,
                                360,
                                mouth_color,
                                1,
                                cv2.LINE_AA,
                            )
                        for point in mouth_points[::2]:
                            p1 = (int(round(point[0])), int(round(point[1])))
                            cv2.circle(overlay, p1, 2, mouth_color, -1, cv2.LINE_AA)
                        drawn = True

            if not drawn:
                return frame
            return cv2.addWeighted(overlay, 0.87, frame, 0.13, 0.0)

        def draw_silhouette_head_direction(frame, base_frame):
            gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _threshold, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.count_nonzero(mask) < base_width * base_height * 0.04:
                cutoff = max(25, int(np.percentile(gray, 68)))
                mask = (gray >= cutoff).astype(np.uint8) * 255
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
            candidates = []
            min_area = max(180, int(base_width * base_height * 0.012))
            for label in range(1, count):
                x, y, w, h, area = [int(value) for value in stats[label]]
                if area < min_area or w < base_width * 0.08 or h < base_height * 0.12:
                    continue
                if y > base_height * 0.65:
                    continue
                candidates.append((area, x, y, w, h, label))
            if not candidates:
                return frame, False
            _area, x, y, w, h, label = max(candidates, key=lambda item: item[0])
            component = labels == label
            head_y1 = y
            head_y2 = min(base_height, y + max(int(h * 0.42), int(base_height * 0.18)))
            head = component[head_y1:head_y2, x:x + w]
            ys, xs = np.nonzero(head)
            if len(xs) < 80:
                return frame, False
            xs_full = xs.astype(np.float32) + x
            ys_full = ys.astype(np.float32) + head_y1
            head_cx = float(np.mean(xs_full))
            head_cy = float(np.mean(ys_full))
            head_w = float(np.max(xs_full) - np.min(xs_full) + 1.0)
            head_h = float(np.max(ys_full) - np.min(ys_full) + 1.0)
            if head_w < 20 or head_h < 25:
                return frame, False

            band_top = head_cy - 0.18 * head_h
            band_bottom = head_cy + 0.18 * head_h
            band = (ys_full >= band_top) & (ys_full <= band_bottom)
            band_xs = xs_full[band]
            if len(band_xs) >= 20:
                left_extent = head_cx - float(np.min(band_xs))
                right_extent = float(np.max(band_xs)) - head_cx
            else:
                left_extent = head_cx - float(np.min(xs_full))
                right_extent = float(np.max(xs_full)) - head_cx
            if max(left_extent, right_extent) <= 1:
                return frame, False
            side_score = (right_extent - left_extent) / max(left_extent, right_extent)
            direction = 1.0 if side_score >= 0 else -1.0
            confidence = abs(side_score)

            low_y1 = min(base_height - 1, y + int(h * 0.46))
            low_y2 = min(base_height, y + int(h * 0.72))
            low = component[low_y1:low_y2, x:x + w]
            low_ys, low_xs = np.nonzero(low)
            neck_cx = float(np.mean(low_xs) + x) if len(low_xs) >= 30 else head_cx
            tilt_dx = head_cx - neck_cx
            if confidence < 0.08:
                direction = 1.0 if tilt_dx >= 0 else -1.0

            overlay = frame.copy()
            color = (24, 24, 24) if safe_mode else (70, 220, 255)
            fine_color = (16, 16, 16) if safe_mode else (50, 150, 175)
            center = (int(round(head_cx)), int(round(head_cy)))
            axes = (
                int(round(max(8.0, head_w * 0.52))),
                int(round(max(10.0, head_h * 0.54))),
            )
            cv2.ellipse(overlay, center, axes, 0, 0, 360, color, 2, cv2.LINE_AA)
            cv2.line(
                overlay,
                (int(round(neck_cx)), int(round(y + h * 0.66))),
                center,
                fine_color,
                2,
                cv2.LINE_AA,
            )
            if safe_mode:
                nose_hint_end = (
                    int(round(head_cx + direction * max(8.0, 0.18 * head_w))),
                    int(round(head_cy - 0.02 * head_h)),
                )
                cv2.line(overlay, center, nose_hint_end, fine_color, 1, cv2.LINE_AA)
            else:
                arrow_start = (int(round(head_cx - direction * 0.08 * head_w)), int(round(head_cy)))
                arrow_end = (
                    int(round(head_cx + direction * max(18.0, 0.42 * head_w))),
                    int(round(head_cy - 0.04 * head_h)),
                )
                cv2.arrowedLine(overlay, arrow_start, arrow_end, color, 3, cv2.LINE_AA, tipLength=0.32)
                draw_direction_badge(
                    overlay,
                    "HEAD R" if direction >= 0 else "HEAD L",
                    (
                        head_cx + (0.12 if direction >= 0 else -0.48) * head_w,
                        head_cy - 0.28 * head_h,
                    ),
                    color,
                )
            ear_x = int(round(head_cx - direction * 0.32 * head_w))
            ear_y = int(round(head_cy + 0.05 * head_h))
            cv2.ellipse(
                overlay,
                (ear_x, ear_y),
                (max(4, int(round(head_w * 0.055))), max(6, int(round(head_h * 0.075)))),
                0,
                0,
                360,
                fine_color,
                1,
                cv2.LINE_AA,
            )
            return cv2.addWeighted(overlay, 0.82, frame, 0.18, 0.0), True

        try:
            while True:
                ok_base, base_frame = base_capture.read()
                ok_source, source_frame = source_capture.read()
                if not ok_base or not ok_source:
                    break

                output_frame = base_frame.copy()
                if pose_capture is not None:
                    ok_pose, pose_frame = pose_capture.read()
                    if ok_pose:
                        output_frame, posed = overlay_body_pose(output_frame, pose_frame)
                        if posed:
                            body_pose_frames += 1
                faces = app.get(source_frame)
                drew_face_guidance = False
                body_tint_this_frame = False
                if faces:
                    prune_tracks(frame_count)
                    faces = sorted(
                        faces,
                        key=lambda face: float(
                            max(0.0, face.bbox[2] - face.bbox[0])
                            * max(0.0, face.bbox[3] - face.bbox[1])
                        ),
                        reverse=True,
                    )
                    face_hit_frames += 1
                    for face in faces[:max_faces]:
                        landmarks = getattr(face, "landmark_2d_106", None)
                        if landmarks is None:
                            continue
                        keypoints = getattr(face, "kps", None)
                        if keypoints is not None and len(keypoints) >= 5:
                            mouth_y = float((keypoints[3][1] + keypoints[4][1]) / 2.0)
                        else:
                            mouth_y = float(face.bbox[1] + (face.bbox[3] - face.bbox[1]) * 0.62)
                        track = assign_track(face.bbox, frame_count)
                        if body_colors_enabled:
                            body_color = body_palette[int(track.get("color_index", 0)) % len(body_palette)]
                            body_mask = body_mask_from_face(base_frame, face.bbox)
                            output_frame, tinted = apply_body_tint(output_frame, base_frame, body_mask, body_color)
                            body_tint_this_frame = body_tint_this_frame or tinted
                        palette = face_palette[int(track.get("color_index", 0)) % len(face_palette)] if color_faces else (18, 18, 18)
                        before = output_frame
                        output_frame = draw_face_guidance(
                            output_frame,
                            landmarks,
                            face.bbox,
                            keypoints,
                            mouth_y,
                            palette,
                        )
                        if output_frame is not before:
                            drew_face_guidance = True
                        face_overlays += 1
                if body_colors_enabled and not body_tint_this_frame:
                    body_mask = largest_body_mask(base_frame)
                    output_frame, body_tint_this_frame = apply_body_tint(output_frame, base_frame, body_mask, body_palette[0])
                if body_tint_this_frame:
                    body_color_frames += 1
                if include_head_pose and not drew_face_guidance:
                    output_frame, drew_silhouette = draw_silhouette_head_direction(output_frame, base_frame)
                    if drew_silhouette:
                        silhouette_head_frames += 1

                writer.write(output_frame)
                frame_count += 1
                if frame_count % 60 == 0:
                    log(
                        f"表情白膜进度: {frame_count}帧，检测到脸 {face_hit_frames}帧，"
                        f"轮廓头向 {silhouette_head_frames}帧，人体骨架 {body_pose_frames}帧"
                    )
        finally:
            source_capture.release()
            base_capture.release()
            if pose_capture is not None:
                pose_capture.release()
            writer.release()

        if not temp_path.exists() or temp_path.stat().st_size <= 0:
            raise RuntimeError("expression_temp_output_missing")

        try:
            from .ffmpeg_tools import ffmpeg_path, run_command

            run_command([
                ffmpeg_path(),
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(temp_path),
                "-map",
                "0:v:0",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ])
            temp_path.unlink(missing_ok=True)
        except Exception:
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            shutil.move(str(temp_path), str(output_path))

        if not output_path.exists() or output_path.stat().st_size <= 0:
            raise RuntimeError("expression_output_missing")

        return {
            "frames": frame_count,
            "face_hit_frames": face_hit_frames,
            "face_hits": face_overlays,
            "mouth_control": "enabled" if include_mouth else "disabled",
            "safe_mode": bool(safe_mode),
            "colored_faces": bool(color_faces and not safe_mode),
            "face_outline": "enabled" if include_face_outline else "disabled",
            "soft_face_relief": "enabled" if include_soft_face_relief else "disabled",
            "body_color_mode": body_color_mode,
            "body_color_frames": body_color_frames,
            "body_pose_path": str(body_pose_path) if body_pose_path else "",
            "body_pose_render_style": pose_render_style,
            "body_pose_capsule_strength": pose_capsule_strength,
            "body_pose_frames": body_pose_frames,
            "silhouette_head_direction_frames": silhouette_head_frames,
            "guidance_layers": [
                "2.5d_depth_body",
                *(
                    []
                    if not body_pose_path
                    else ["dwpose_debug_skeleton" if pose_render_style == "debug_lines" else "dwpose_capsule_mannequin"]
                ),
                *([] if not body_colors_enabled else ["red_green_body_id"]),
                *([] if not include_face_outline else ["face_outline"]),
                *([] if not include_head_pose else ["head_pose", "silhouette_head_direction", "ear_side_hint", "nose_bridge"]),
                *([] if include_soft_face_relief else [
                    *([] if not include_eyes else ["eyes"]),
                    *([] if not include_brows else ["brows"]),
                    *([] if not include_mouth else ["mouth_lips"]),
                ]),
                *([] if not include_soft_face_relief else ["soft_face_relief"]),
                *([] if safe_mode or not color_faces else ["per_person_color"]),
            ],
            "face_track_count": next_track_id,
            "base_path": str(base_path),
        }

    @staticmethod
    def _render_colored_mask_preview_video(
        role_frame_paths: list[list[Path]],
        output_path: Path,
        fps: float,
        colors: list[tuple[int, int, int]] | None = None,
    ) -> None:
        if not role_frame_paths:
            raise RuntimeError("No role mask frames available for preview video")
        import shutil
        import tempfile
        import cv2
        import numpy as np
        from PIL import Image

        def read_mask(path: Path):
            try:
                with Image.open(path) as image:
                    return np.array(image.convert("L"))
            except Exception:  # noqa: BLE001
                return None

        palette = colors or [
            (255, 0, 0),
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
        ]
        usable_tracks = [paths for paths in role_frame_paths if paths]
        if not usable_tracks:
            raise RuntimeError("No usable role masks available for preview video")
        first = read_mask(usable_tracks[0][0])
        if first is None:
            raise RuntimeError(f"Failed to read preview frame: {usable_tracks[0][0]}")
        height, width = first.shape[:2]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_path.stem}_frames_", dir=str(output_path.parent)))
        nonempty_frames = 0
        try:
            frame_count = max(len(paths) for paths in usable_tracks)
            for frame_index in range(frame_count):
                canvas = np.zeros((height, width, 3), dtype=np.uint8)
                frame_has_content = False
                for role_index, paths in enumerate(role_frame_paths):
                    if frame_index >= len(paths):
                        continue
                    mask = read_mask(paths[frame_index])
                    if mask is None:
                        continue
                    if mask.shape[:2] != (height, width):
                        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    selected = mask > 16
                    if not np.any(selected):
                        continue
                    color = palette[role_index % len(palette)]
                    canvas[selected] = color
                    contours, _ = cv2.findContours(
                        (selected.astype(np.uint8) * 255),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    cv2.drawContours(canvas, contours, -1, (255, 255, 255), 2, cv2.LINE_AA)
                    frame_has_content = True
                if frame_has_content:
                    nonempty_frames += 1
                Image.fromarray(canvas[:, :, ::-1]).save(temp_dir / f"frame_{frame_index + 1:06d}.png")
        finally:
            if nonempty_frames <= 0:
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise RuntimeError("No non-empty preview masks were rendered")

        from .ffmpeg_tools import ffmpeg_path, run_command

        try:
            run_command([
                ffmpeg_path(),
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(max(1.0, float(fps) or 24.0)),
                "-i",
                str(temp_dir / "frame_%06d.png"),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def _history_debug_summary(history_item: dict | None) -> str:
        if not isinstance(history_item, dict):
            return "history item is empty"
        outputs = history_item.get("outputs") or {}
        output_keys = ", ".join(map(str, outputs.keys())) or "none"
        status = history_item.get("status") or {}
        messages = status.get("messages") if isinstance(status, dict) else None
        parts = [f"output nodes={output_keys}"]
        if isinstance(status, dict):
            status_str = str(status.get("status_str") or "").strip()
            completed = status.get("completed")
            if status_str or completed is not None:
                parts.append(f"status={status_str or 'unknown'} completed={completed}")
        if isinstance(messages, list) and messages:
            tail = messages[-3:]
            compact = []
            for item in tail:
                try:
                    compact.append(json.dumps(item, ensure_ascii=False)[:500])
                except TypeError:
                    compact.append(str(item)[:500])
            parts.append("messages=" + " | ".join(compact))
        return "; ".join(parts)

    def _download_output_asset(self, asset: dict, target: Path) -> str:
        last_error: Exception | None = None
        for attempt in range(1, _UPLOAD_ATTEMPTS + 1):
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
            except (requests.Timeout, requests.ConnectionError) as error:
                last_error = error
            if attempt < _UPLOAD_ATTEMPTS:
                time.sleep(float(2 ** (attempt - 1)))
        raise RuntimeError(
            f"ComfyUI 下载输出资产失败: {asset.get('filename')}；"
            f"最后一次错误：{last_error}"
        ) from last_error

    def _build_template(self) -> dict:
        """最小 SCAIL-2 模板（单参考图 + 单人追踪）"""
        return {
            "2": {
                "inputs": {
                    "video": "VIDEO_PLACEHOLDER", "force_rate": 24,
                    "custom_width": 0, "custom_height": 0,
                    "frame_load_cap": 120, "skip_first_frames": 0,
                    "select_every_nth": 1, "format": "AnimateDiff"
                },
                "class_type": "VHS_LoadVideo",
            },
            "3": {
                "inputs": {"resolution": "512p", "custom_width": 352, "custom_height": 640, "video": ["2", 0]},
                "class_type": "SCAIL2FitVideo",
            },
            "10": {
                "inputs": {"model_name": "wan2.1_14B_SCAIL_2_fp8_scaled.safetensors", "weight_dtype": "default", "compute_dtype": "default", "patch_cublaslinear": False, "sage_attention": "auto", "enable_fp16_accumulation": True},
                "class_type": "DiffusionModelLoaderKJ",
            },
            "11": {
                "inputs": {"chunks": 2, "dim_threshold": 4096, "model": ["10", 0]},
                "class_type": "WanChunkFeedForward",
            },
            "12": {
                "inputs": {"lora_name": "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors", "strength_model": 0.8, "model": ["65", 0]},
                "class_type": "LoraLoaderModelOnly",
            },
            "13": {
                "inputs": {"shift": 5, "model": ["12", 0]},
                "class_type": "ModelSamplingSD3",
            },
            "14": {
                "inputs": {"scheduler": "simple", "steps": 8, "denoise": 1, "model": ["12", 0]},
                "class_type": "BasicScheduler",
            },
            "15": {
                "inputs": {"sampler_name": "euler"},
                "class_type": "KSamplerSelect",
            },
            "16": {
                "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"},
                "class_type": "CLIPLoader",
            },
            "17": {
                "inputs": {"text": "POSITIVE_PLACEHOLDER", "clip": ["16", 0]},
                "class_type": "CLIPTextEncode",
            },
            "18": {
                "inputs": {"text": "bad video, blurry, low quality, distorted, artifacts", "clip": ["16", 0]},
                "class_type": "CLIPTextEncode",
            },
            "19": {
                "inputs": {"vae_name": "Wan2_1_VAE_bf16.safetensors"},
                "class_type": "VAELoader",
            },
            "20": {
                "inputs": {"clip_name": "clip_vision_vit_h.safetensors"},
                "class_type": "CLIPVisionLoader",
            },
            "30": {
                "inputs": {"ckpt_name": "sam3.1_multiplex_fp16.safetensors"},
                "class_type": "CheckpointLoaderSimple",
            },
            "31": {
                "inputs": {"text": "SAM_TEXT_PLACEHOLDER", "clip": ["30", 1]},
                "class_type": "CLIPTextEncode",
            },
            "32": {
                "inputs": {"detection_threshold": SAM3_VIDEO_DETECTION_THRESHOLD, "max_objects": 2, "detect_interval": SAM3_VIDEO_DETECT_INTERVAL, "images": ["3", 0], "model": ["30", 0], "conditioning": ["31", 0]},
                "class_type": "SAM3_VideoTrack",
            },
            "40": {
                "inputs": {
                    "seed": 0, "cfg": 1, "mode": "replacement", "advanced": True, "long_video_mode": "chunk",
                    "max_frames": 0, "chunk_frames": 81, "overlap_frames": 5, "color_correction": False,
                    "context_frames": 81, "context_overlap_frames": 20,
                    "model": ["13", 0], "positive": ["17", 0], "negative": ["18", 0], "vae": ["19", 0],
                    "sampler": ["15", 0], "sigmas": ["14", 0],
                    "reference_image": ["51", 0], "pose_video": ["3", 0], "clip_vision": ["20", 0],
                    "driving_track_data": ["32", 0],
                },
                "class_type": "SCAIL2SimpleVideo",
            },
            "43": {
                "inputs": {"frame_rate": 24, "loop_count": 0, "filename_prefix": "SCAIL2/multi_ref", "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19, "save_metadata": True, "pingpong": False, "save_output": True, "images": ["40", 0], "audio": ["2", 2]},
                "class_type": "VHS_VideoCombine",
            },
            "50": {
                "inputs": {"subject_count": 1, "reference_count": 0, "subject_1_image": ["68", 0]},
                "class_type": "SCAIL2ReferencePack",
            },
            "51": {
                "inputs": {"detection_threshold": SAM3_REFERENCE_DETECTION_THRESHOLD, "max_objects": 2, "detect_interval": SAM3_REFERENCE_DETECT_INTERVAL, "reference_pack": ["50", 0], "sam_model": ["30", 0], "conditioning": ["31", 0]},
                "class_type": "SCAIL2ReferenceSAMBuilder",
            },
            "65": {
                "inputs": {"lora_name": "SCAIL-2/wan2.1_SCAIL_2_DPO_lora_bf16.safetensors", "strength_model": 0.8, "model": ["11", 0]},
                "class_type": "LoraLoaderModelOnly",
            },
            "68": {
                "inputs": {"image": "IMG1_PLACEHOLDER"},
                "class_type": "LoadImage",
            },
            "77": {
                "inputs": {"image": "IMG2_PLACEHOLDER"},
                "class_type": "LoadImage",
            },
            "83": {
                "inputs": {"image": "IMG3_PLACEHOLDER"},
                "class_type": "LoadImage",
            },
        }

    def _build_colored_mask_template(self) -> dict:
        """Advanced SCAIL-2 template with explicit reference and driving masks."""
        return {
            "2": {
                "inputs": {
                    "video": "VIDEO_PLACEHOLDER",
                    "force_rate": 24,
                    "custom_width": 0,
                    "custom_height": 0,
                    "frame_load_cap": 81,
                    "skip_first_frames": 0,
                    "select_every_nth": 1,
                    "format": "AnimateDiff",
                },
                "class_type": "VHS_LoadVideo",
            },
            "3": {
                "inputs": {
                    "resolution": "custom",
                    "custom_width": 512,
                    "custom_height": 896,
                    "video": ["2", 0],
                },
                "class_type": "SCAIL2FitVideo",
            },
            "10": {
                "inputs": {
                    "model_name": "wan2.1_14B_SCAIL_2_fp8_scaled.safetensors",
                    "weight_dtype": "default",
                    "compute_dtype": "default",
                    "patch_cublaslinear": False,
                    "sage_attention": "auto",
                    "enable_fp16_accumulation": True,
                },
                "class_type": "DiffusionModelLoaderKJ",
            },
            "11": {
                "inputs": {"chunks": 2, "dim_threshold": 4096, "model": ["10", 0]},
                "class_type": "WanChunkFeedForward",
            },
            "65": {
                "inputs": {
                    "lora_name": "SCAIL-2/wan2.1_SCAIL_2_DPO_lora_bf16.safetensors",
                    "strength_model": 0.8,
                    "model": ["11", 0],
                },
                "class_type": "LoraLoaderModelOnly",
            },
            "12": {
                "inputs": {
                    "lora_name": "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
                    "strength_model": 0.8,
                    "model": ["65", 0],
                },
                "class_type": "LoraLoaderModelOnly",
            },
            "13": {
                "inputs": {"shift": 5, "model": ["12", 0]},
                "class_type": "ModelSamplingSD3",
            },
            "14": {
                "inputs": {"scheduler": "simple", "steps": 8, "denoise": 1.0, "model": ["12", 0]},
                "class_type": "BasicScheduler",
            },
            "15": {
                "inputs": {"sampler_name": "euler"},
                "class_type": "KSamplerSelect",
            },
            "16": {
                "inputs": {
                    "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                    "type": "wan",
                    "device": "default",
                },
                "class_type": "CLIPLoader",
            },
            "17": {
                "inputs": {"text": "POSITIVE_PLACEHOLDER", "clip": ["16", 0]},
                "class_type": "CLIPTextEncode",
            },
            "18": {
                "inputs": {
                    "text": (
                        "bad video, blurry, low quality, distorted, artifacts, new background, changed scene, "
                        "added theater stage, added black curtain, invented studio backdrop, added runway, "
                        "unrelated dance floor, invented showroom, unrelated dancing, new choreography, "
                        "default background, beautified face, de-aged face, wrong age, wrong face shape, "
                        "slimmed body, skinny parent, thin model body, changed body type, altered body proportions"
                    ),
                    "clip": ["16", 0],
                },
                "class_type": "CLIPTextEncode",
            },
            "19": {
                "inputs": {"vae_name": "Wan2_1_VAE_bf16.safetensors"},
                "class_type": "VAELoader",
            },
            "20": {
                "inputs": {"clip_name": "clip_vision_vit_h.safetensors"},
                "class_type": "CLIPVisionLoader",
            },
            "21": {
                "inputs": {"crop": "none", "clip_vision": ["20", 0], "image": ["ref_resize", 0]},
                "class_type": "CLIPVisionEncode",
            },
            "30": {
                "inputs": {"ckpt_name": "sam3.1_multiplex_fp16.safetensors"},
                "class_type": "CheckpointLoaderSimple",
            },
            "31": {
                "inputs": {"text": "SAM_TEXT_PLACEHOLDER", "clip": ["30", 1]},
                "class_type": "CLIPTextEncode",
            },
            "32": {
                "inputs": {
                    "detection_threshold": SAM3_VIDEO_DETECTION_THRESHOLD,
                    "max_objects": 3,
                    "detect_interval": SAM3_VIDEO_DETECT_INTERVAL,
                    "images": ["3", 0],
                    "model": ["30", 0],
                    "conditioning": ["31", 0],
                },
                "class_type": "SAM3_VideoTrack",
            },
            "33": {
                "inputs": {
                    "detection_threshold": SAM3_REFERENCE_DETECTION_THRESHOLD,
                    "max_objects": 3,
                    "detect_interval": SAM3_REFERENCE_DETECT_INTERVAL,
                    "images": ["ref_resize", 0],
                    "model": ["30", 0],
                    "conditioning": ["31", 0],
                },
                "class_type": "SAM3_VideoTrack",
            },
            "34": {
                "inputs": {
                    "object_indices": "",
                    "sort_by": "left_to_right",
                    "replacement_mode": True,
                    "driving_track_data": ["32", 0],
                    "ref_track_data": ["33", 0],
                },
                "class_type": "SCAIL2ColoredMask",
            },
            "40": {
                "inputs": {
                    "width": 512,
                    "height": 896,
                    "length": 81,
                    "batch_size": 1,
                    "pose_strength": 1.0,
                    "pose_start": 0.0,
                    "pose_end": 1.0,
                    "video_frame_offset": 0,
                    "previous_frame_count": 5,
                    "replacement_mode": True,
                    "positive": ["17", 0],
                    "negative": ["18", 0],
                    "vae": ["19", 0],
                    "pose_video": ["3", 0],
                    "pose_video_mask": ["34", 0],
                    "reference_image": ["ref_resize", 0],
                    "reference_image_mask": ["34", 1],
                    "clip_vision_output": ["21", 0],
                },
                "class_type": "WanSCAILToVideo",
            },
            "41": {
                "inputs": {
                    "add_noise": True,
                    "noise_seed": 0,
                    "cfg": 1,
                    "model": ["13", 0],
                    "positive": ["40", 0],
                    "negative": ["40", 1],
                    "sampler": ["15", 0],
                    "sigmas": ["14", 0],
                    "latent_image": ["40", 2],
                },
                "class_type": "SamplerCustom",
            },
            "42": {
                "inputs": {"samples": ["41", 1], "vae": ["19", 0]},
                "class_type": "VAEDecode",
            },
            "43": {
                "inputs": {
                    "frame_rate": 24,
                    "loop_count": 0,
                    "filename_prefix": "SCAIL2/colored_mask",
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 19,
                    "save_metadata": True,
                    "pingpong": False,
                    "save_output": True,
                    "images": ["42", 0],
                    "audio": ["2", 2],
                },
                "class_type": "VHS_VideoCombine",
            },
            "mask_pose_video": {
                "inputs": {
                    "frame_rate": 24,
                    "loop_count": 0,
                    "filename_prefix": "SCAIL2/debug_pose_mask",
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 17,
                    "save_metadata": True,
                    "pingpong": False,
                    "save_output": True,
                    "images": ["34", 0],
                },
                "class_type": "VHS_VideoCombine",
            },
            "mask_reference_save": {
                "inputs": {
                    "filename_prefix": "SCAIL2/debug_reference_mask",
                    "images": ["34", 1],
                },
                "class_type": "SaveImage",
            },
            "mask_prefix_save": {
                "inputs": {
                    "filename_prefix": "SCAIL2/debug_prefix_mask",
                    "images": ["34", 2],
                },
                "class_type": "SaveImage",
            },
            "ref_load": {
                "inputs": {"image": "REFERENCE_COLLAGE_PLACEHOLDER"},
                "class_type": "LoadImage",
            },
            "ref_resize": self._reference_resize_node("ref_load", 512, 896),
        }

    def _build_wananimate_scail2_template(self) -> dict:
        """WanAnimatePlus SCAIL-2 template using isolated motion and colored identities."""
        return {
            "2": {
                "inputs": {
                    "video": "VIDEO_PLACEHOLDER",
                    "force_rate": 24,
                    "custom_width": 0,
                    "custom_height": 0,
                    "frame_load_cap": 81,
                    "skip_first_frames": 0,
                    "select_every_nth": 1,
                    "format": "AnimateDiff",
                },
                "class_type": "VHS_LoadVideo",
            },
            "3": {
                "inputs": {
                    "resolution": "custom",
                    "custom_width": 512,
                    "custom_height": 896,
                    "video": ["2", 0],
                },
                "class_type": "SCAIL2FitVideo",
            },
            "30": {
                "inputs": {"ckpt_name": "sam3.1_multiplex_fp16.safetensors"},
                "class_type": "CheckpointLoaderSimple",
            },
            "31": {
                "inputs": {"text": "SAM_TEXT_PLACEHOLDER", "clip": ["30", 1]},
                "class_type": "CLIPTextEncode",
            },
            "32": {
                "inputs": {
                    "detection_threshold": SAM3_VIDEO_DETECTION_THRESHOLD,
                    "max_objects": 3,
                    "detect_interval": SAM3_VIDEO_DETECT_INTERVAL,
                    "images": ["3", 0],
                    "model": ["30", 0],
                    "conditioning": ["31", 0],
                },
                "class_type": "SAM3_VideoTrack",
            },
            "33": {
                "inputs": {
                    "detection_threshold": SAM3_REFERENCE_DETECTION_THRESHOLD,
                    "max_objects": 3,
                    "detect_interval": SAM3_REFERENCE_DETECT_INTERVAL,
                    "images": ["ref_resize", 0],
                    "model": ["30", 0],
                    "conditioning": ["31", 0],
                },
                "class_type": "SAM3_VideoTrack",
            },
            "34": {
                "inputs": {
                    "object_indices": "",
                    "sort_by": "left_to_right",
                    "prefix_mask_mode": "Multi Image Single Color",
                    "replacement_mode": False,
                    "render_device": "gpu",
                    "driving_track_data": ["32", 0],
                    "ref_track_data": ["33", 0],
                },
                "class_type": "SCAIL2ColoredMaskV2",
            },
            "source_people_mask": {
                "inputs": {
                    "track_data": ["32", 0],
                    "object_indices": "",
                },
                "class_type": "SAM3_TrackToMask",
            },
            "black_pose_batch": {
                "inputs": {
                    "width": 512,
                    "height": 896,
                    "batch_size": 81,
                    "color": 0,
                },
                "class_type": "EmptyImage",
            },
            "black_pose_video": {
                "inputs": {
                    "destination": ["black_pose_batch", 0],
                    "source": ["3", 0],
                    "mask": ["source_people_mask", 0],
                    "x": 0,
                    "y": 0,
                    "resize_source": False,
                },
                "class_type": "ImageCompositeMasked",
            },
            "clip_loader": {
                "inputs": {"clip_name": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"},
                "class_type": "CLIPVisionLoader",
            },
            "clip_vision": {
                "inputs": {
                    "strength": 1.0,
                    "crop": "center",
                    "combine_embeds": "batch",
                    "force_offload": True,
                    "tiles": 0,
                    "ratio": 0.5,
                    "clip_vision": ["clip_loader", 0],
                    "images": ["clip_ref_batch", 0],
                },
                "class_type": "WanAnimatePlus ClipVisionEncode V2",
            },
            "wan_vae": {
                "inputs": {
                    "model_name": "Wan2_1_VAE_bf16.safetensors",
                    "precision": "bf16",
                    "use_cpu_cache": False,
                    "verbose": False,
                },
                "class_type": "WanAnimatePlus VAELoader",
            },
            "blockswap_args": {
                "inputs": {
                    "blocks_to_swap": 30,
                    "offload_img_emb": False,
                    "offload_txt_emb": False,
                    "use_non_blocking": True,
                    "vace_blocks_to_swap": 1,
                    "prefetch_blocks": 0,
                    "block_swap_debug": False,
                },
                "class_type": "WanAnimatePlus BlockSwap",
            },
            "lora_select": {
                "inputs": {
                    "lora_0": "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors",
                    "strength_0": 1.0,
                    "lora_1": "SCAIL-2/wan2.1_SCAIL_2_DPO_lora_bf16.safetensors",
                    "strength_1": 1.0,
                    "lora_2": "none",
                    "strength_2": 0.0,
                    "lora_3": "none",
                    "strength_3": 0.0,
                    "lora_4": "none",
                    "strength_4": 0.0,
                    "low_mem_load": False,
                    "merge_loras": False,
                },
                "class_type": "WanAnimatePlus LoraSelectMulti",
            },
            "wan_model": {
                "inputs": {
                    "model": "wan2.1_14B_SCAIL_2_fp8_scaled.safetensors",
                    "base_precision": "fp16",
                    "quantization": "disabled",
                    "load_device": "offload_device",
                    "attention_mode": "sageattn",
                    "rms_norm_function": "default",
                },
                "class_type": "WanAnimatePlus ModelLoader",
            },
            "set_loras": {
                "inputs": {"model": ["wan_model", 0], "lora": ["lora_select", 0]},
                "class_type": "WanAnimatePlus SetLoRAs",
            },
            "set_blockswap": {
                "inputs": {"model": ["set_loras", 0], "block_swap_args": ["blockswap_args", 0]},
                "class_type": "WanAnimatePlus SetBlockSwap",
            },
            "text": {
                "inputs": {
                    "model_name": "umt5-xxl-enc-fp8_e4m3fn.safetensors",
                    "precision": "bf16",
                    "positive_prompt": "POSITIVE_PLACEHOLDER",
                    "negative_prompt": (
                        "text, watermark, subtitles, blurry, low quality, jpeg artifacts, "
                        "bad anatomy, deformed limbs, extra fingers, bad hands, bad face, "
                        "duplicate people, wrong clothes, identity swap, gender swap, male to female, "
                        "female to male, new background, changed scene, fantasy scene, extra people, "
                        "added theater stage, added black curtain, invented studio backdrop, added runway, "
                        "unrelated dance floor, invented showroom, unrelated dancing, new choreography, "
                        "default background, beautified face, de-aged face, wrong age, wrong face shape, "
                        "slimmed body, skinny parent, thin model body, changed body type, altered body proportions"
                    ),
                    "quantization": "disabled",
                    "use_disk_cache": False,
                    "device": "gpu",
                },
                "class_type": "WanVideoTextEncodeCached",
            },
            "embeds": {
                "inputs": {
                    "width": 512,
                    "height": 896,
                    "num_frames": 81,
                    "frame_window_size": 81,
                    "force_offload": True,
                    "pose_strength": 1.0,
                    "ref_strength": 1.0,
                    "replacement_mode": False,
                    "tiled_vae": False,
                    "transition_colormatch": "disabled",
                    "loop_colormatch_reference": "previous_matched_frame",
                    "prefix_alpha_crop": False,
                    "preserve_main_ref_background": False,
                    "single_frame_prefix_encoding": True,
                    "vae": ["wan_vae", 0],
                    "clip_embeds": ["clip_vision", 0],
                    "ref_image": ["ref_resize", 0],
                    "pose_images": ["black_pose_video", 0],
                    "pose_image_mask": ["34", 0],
                    "reference_image_mask": ["34", 1],
                },
                "class_type": "WanAnimatePlus SCAIL_2 Embeds",
            },
            "sampler": {
                "inputs": {
                    "steps": 8,
                    "cfg": 1.0,
                    "shift": 5.0,
                    "seed": 0,
                    "force_offload": True,
                    "scheduler": "euler",
                    "riflex_freq_index": 0,
                    "denoise_strength": 1.0,
                    "batched_cfg": False,
                    "rope_function": "comfy",
                    "start_step": 0,
                    "end_step": -1,
                    "add_noise_to_samples": False,
                    "guidance_mode": "cfg",
                    "model": ["set_blockswap", 0],
                    "image_embeds": ["embeds", 0],
                    "text_embeds": ["text", 0],
                },
                "class_type": "WanAnimatePlus SamplerSettings",
            },
            "sample": {
                "inputs": {"sampler_inputs": ["sampler", 0]},
                "class_type": "WanAnimatePlus SamplerFromSettings",
            },
            "decode": {
                "inputs": {
                    "enable_vae_tiling": False,
                    "tile_x": 272,
                    "tile_y": 272,
                    "tile_stride_x": 144,
                    "tile_stride_y": 216,
                    "normalization": "default",
                    "vae": ["wan_vae", 0],
                    "samples": ["sample", 0],
                },
                "class_type": "WanAnimatePlus Decode",
            },
            "43": {
                "inputs": {
                    "frame_rate": 24,
                    "loop_count": 0,
                    "filename_prefix": "SCAIL2/wananimate_v2",
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 19,
                    "save_metadata": True,
                    "pingpong": False,
                    "save_output": True,
                    "images": ["decode", 0],
                    "audio": ["2", 2],
                },
                "class_type": "VHS_VideoCombine",
            },
            "mask_pose_video": {
                "inputs": {
                    "frame_rate": 24,
                    "loop_count": 0,
                    "filename_prefix": "SCAIL2/debug_pose_mask",
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 17,
                    "save_metadata": True,
                    "pingpong": False,
                    "save_output": True,
                    "images": ["34", 0],
                },
                "class_type": "VHS_VideoCombine",
            },
            "mask_reference_save": {
                "inputs": {
                    "filename_prefix": "SCAIL2/debug_reference_mask",
                    "images": ["34", 1],
                },
                "class_type": "SaveImage",
            },
            "ref_load": {
                "inputs": {"image": "REFERENCE_COLLAGE_PLACEHOLDER"},
                "class_type": "LoadImage",
            },
            "ref_resize": self._reference_resize_node("ref_load", 512, 896),
        }

    def _build_bernini_test_template(self, reference_count: int) -> dict:
        """Minimal WanAnimatePlus Bernini rv2v template."""
        reference_count = max(0, min(10, int(reference_count)))
        workflow = {
            "2": {
                "inputs": {
                    "video": "VIDEO_PLACEHOLDER",
                    "force_rate": 24,
                    "custom_width": 0,
                    "custom_height": 0,
                    "frame_load_cap": 81,
                    "skip_first_frames": 0,
                    "select_every_nth": 1,
                    "format": "AnimateDiff",
                },
                "class_type": "VHS_LoadVideo",
            },
            "3": {
                "inputs": {
                    "resolution": "custom",
                    "custom_width": 512,
                    "custom_height": 896,
                    "video": ["2", 0],
                },
                "class_type": "SCAIL2FitVideo",
            },
            "wan_vae": {
                "inputs": {
                    "model_name": "wan2.2_vae.safetensors",
                    "precision": "bf16",
                    "use_cpu_cache": False,
                    "verbose": False,
                },
                "class_type": "WanAnimatePlus VAELoader",
            },
            "blockswap_args": {
                "inputs": {
                    "blocks_to_swap": 30,
                    "offload_img_emb": False,
                    "offload_txt_emb": False,
                    "use_non_blocking": True,
                    "vace_blocks_to_swap": 1,
                    "prefetch_blocks": 0,
                    "block_swap_debug": False,
                },
                "class_type": "WanAnimatePlus BlockSwap",
            },
            "wan_model": {
                "inputs": {
                    "model": "Wan_2.2_ComfyUI_Repackaged/wan2.2_ti2v_5B_fp16.safetensors",
                    "base_precision": "fp16",
                    "quantization": "disabled",
                    "load_device": "offload_device",
                    "attention_mode": "sageattn",
                    "rms_norm_function": "default",
                },
                "class_type": "WanAnimatePlus ModelLoader",
            },
            "set_blockswap": {
                "inputs": {"model": ["wan_model", 0], "block_swap_args": ["blockswap_args", 0]},
                "class_type": "WanAnimatePlus SetBlockSwap",
            },
            "text": {
                "inputs": {
                    "model_name": "umt5-xxl-enc-fp8_e4m3fn.safetensors",
                    "precision": "bf16",
                    "positive_prompt": "POSITIVE_PLACEHOLDER",
                    "negative_prompt": (
                        "text, watermark, subtitles, blurry, low quality, jpeg artifacts, "
                        "bad anatomy, deformed limbs, extra fingers, bad hands, bad face, "
                        "duplicate people, identity not changed, wrong identity, wrong clothes, "
                        "gender swap, changed background, added stage, new choreography"
                    ),
                    "quantization": "disabled",
                    "use_disk_cache": False,
                    "device": "gpu",
                },
                "class_type": "WanVideoTextEncodeCached",
            },
            "bernini": {
                "inputs": {
                    "task_type": "rv2v",
                    "width": 512,
                    "height": 896,
                    "num_frames": 81,
                    "ref_max_size": 848,
                    "force_offload": True,
                    "tiled_vae": False,
                    "vae": ["wan_vae", 0],
                    "source_video": ["3", 0],
                },
                "class_type": "WanAnimatePlus Bernini",
            },
            "sampler": {
                "inputs": {
                    "steps": 8,
                    "cfg": 1.0,
                    "shift": 5.0,
                    "seed": 0,
                    "force_offload": True,
                    "scheduler": "euler",
                    "riflex_freq_index": 0,
                    "denoise_strength": 1.0,
                    "batched_cfg": False,
                    "rope_function": "comfy",
                    "start_step": 0,
                    "end_step": -1,
                    "add_noise_to_samples": False,
                    "guidance_mode": "cfg_chain",
                    "chain_omega_V": 1.25,
                    "chain_omega_I": 4.5,
                    "chain_omega_TI": 4.0,
                    "model": ["set_blockswap", 0],
                    "image_embeds": ["bernini", 0],
                    "text_embeds": ["text", 0],
                },
                "class_type": "WanAnimatePlus SamplerSettings",
            },
            "sample": {
                "inputs": {"sampler_inputs": ["sampler", 0]},
                "class_type": "WanAnimatePlus SamplerFromSettings",
            },
            "decode": {
                "inputs": {
                    "enable_vae_tiling": False,
                    "tile_x": 272,
                    "tile_y": 272,
                    "tile_stride_x": 144,
                    "tile_stride_y": 216,
                    "normalization": "default",
                    "vae": ["wan_vae", 0],
                    "samples": ["sample", 0],
                },
                "class_type": "WanAnimatePlus Decode",
            },
            "43": {
                "inputs": {
                    "frame_rate": 24,
                    "loop_count": 0,
                    "filename_prefix": "SCAIL2/bernini_test",
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 19,
                    "save_metadata": True,
                    "pingpong": False,
                    "save_output": True,
                    "images": ["decode", 0],
                    "audio": ["2", 2],
                },
                "class_type": "VHS_VideoCombine",
            },
        }
        for index in range(reference_count):
            load_id = f"subject_load_{index}"
            workflow[load_id] = {"inputs": {"image": f"IMG{index + 1}_PLACEHOLDER"}, "class_type": "LoadImage"}
            workflow["bernini"]["inputs"][f"reference_image_{index + 1}"] = [load_id, 0]
        return workflow

    def _patch_workflow(
        self,
        wf: dict,
        vid_name: str,
        subject_names: list[str],
        extra_ref_names: list[str],
        sam_text: str,
        positive_prompt: str,
        width: int,
        height: int,
        video_window: dict | None = None,
        sampler_preset: str = "balanced",
        normalize_reference: bool = True,
        subject_extra_ref_names: list[list[str]] | None = None,
        reference_collage_name: str = "",
        driving_object_indices: list[int] | None = None,
        subject_appearance_hints: list[str] | None = None,
        limit_frame_cap_by_sampler: bool = True,
    ) -> dict:
        """替换工作流中的动态参数"""
        wf = copy.deepcopy(wf)
        subject_count = len(subject_names)
        normalized_driving_indices = list(driving_object_indices or range(subject_count))
        if len(normalized_driving_indices) != subject_count:
            normalized_driving_indices = list(range(subject_count))
        subject_extra_ref_names = list(subject_extra_ref_names or [])
        if len(subject_extra_ref_names) < subject_count:
            subject_extra_ref_names.extend([[] for _ in range(subject_count - len(subject_extra_ref_names))])
        subject_extra_ref_names = subject_extra_ref_names[:subject_count]
        subject_appearance_hints = list(subject_appearance_hints or [])[:subject_count]
        if len(subject_appearance_hints) < subject_count:
            subject_appearance_hints.extend([""] * (subject_count - len(subject_appearance_hints)))
        window = {
            "force_rate": 24,
            "frame_load_cap": 120,
            "skip_first_frames": 0,
            "select_every_nth": 1,
            **(video_window or {}),
        }
        sampler = self._sampler_settings(sampler_preset)
        workflow_uses_colored_mask = any(
            node.get("class_type") in {
                "WanSCAILToVideo",
                "WanAnimatePlus SCAIL_2 Embeds",
                "WanAnimatePlus Bernini",
            }
            for node in wf.values()
        )
        workflow_uses_wananimate = any(
            node.get("class_type") == "WanAnimatePlus SCAIL_2 Embeds" for node in wf.values()
        )
        workflow_uses_bernini = any(
            node.get("class_type") == "WanAnimatePlus Bernini" for node in wf.values()
        )
        effective_frame_cap = (
            min(window["frame_load_cap"], sampler["chunk_frames"])
            if workflow_uses_colored_mask and limit_frame_cap_by_sampler
            else window["frame_load_cap"]
        )

        # subject 是生成后要出现的人；reference 只是额外视角或细节图。
        for i, name in enumerate(subject_names):
            nid = f"subject_load_{i}"
            wf[nid] = {"inputs": {"image": name}, "class_type": "LoadImage"}
            if normalize_reference:
                wf[f"subject_resize_{i}"] = self._reference_resize_node(nid, width, height)
        for i, name in enumerate(extra_ref_names):
            nid = f"extra_ref_load_{i}"
            wf[nid] = {"inputs": {"image": name}, "class_type": "LoadImage"}
            if normalize_reference:
                wf[f"extra_ref_resize_{i}"] = self._reference_resize_node(nid, width, height)
        for subject_index, names in enumerate(subject_extra_ref_names):
            for extra_index, name in enumerate(names[:6]):
                nid = f"subject_extra_load_{subject_index}_{extra_index}"
                wf[nid] = {"inputs": {"image": name}, "class_type": "LoadImage"}
                if normalize_reference:
                    wf[f"subject_extra_resize_{subject_index}_{extra_index}"] = (
                        self._reference_resize_node(nid, width, height)
                    )

        if workflow_uses_wananimate:
            subject_sources: list[list[str | int]] = []
            for subject_index in range(subject_count):
                main_source = (
                    f"subject_resize_{subject_index}"
                    if normalize_reference
                    else f"subject_load_{subject_index}"
                )
                subject_sources.append([main_source, 0])
            wf["subject_ref_batch"] = {
                "class_type": "ImageBatchMultiV2",
                "inputs": {"inputcount": max(2, len(subject_sources))},
            }
            for index, source in enumerate(subject_sources, 1):
                wf["subject_ref_batch"]["inputs"][f"image_{index}"] = source

            # Keep the global CLIP identity pool clean for multi-person replacement.
            # Per-subject extra refs are uploaded and can be used by one-role passes,
            # but mixing all extra views here makes identities bleed across people.
            clip_sources: list[list[str | int]] = [["ref_resize", 0], ["subject_ref_batch", 0]]
            wf["clip_ref_batch"] = {
                "class_type": "ImageBatchMultiV2",
                "inputs": {"inputcount": max(2, len(clip_sources))},
            }
            for index, source in enumerate(clip_sources, 1):
                wf["clip_ref_batch"]["inputs"][f"image_{index}"] = source
        elif workflow_uses_colored_mask and "21" in wf:
            clip_sources = []
            for subject_index in range(subject_count):
                main_source = (
                    f"subject_resize_{subject_index}"
                    if normalize_reference
                    else f"subject_load_{subject_index}"
                )
                clip_sources.append([main_source, 0])
                for extra_index, _ in enumerate(subject_extra_ref_names[subject_index][:6]):
                    extra_source = (
                        f"subject_extra_resize_{subject_index}_{extra_index}"
                        if normalize_reference
                        else f"subject_extra_load_{subject_index}_{extra_index}"
                    )
                    clip_sources.append([extra_source, 0])
            if len(clip_sources) >= 2:
                wf["clip_ref_batch"] = {
                    "class_type": "ImageBatchMulti",
                    "inputs": {"inputcount": len(clip_sources)},
                }
                for index, source in enumerate(clip_sources, 1):
                    wf["clip_ref_batch"]["inputs"][f"image_{index}"] = source
                wf["21"]["inputs"]["image"] = ["clip_ref_batch", 0]

        for nid, node in wf.items():
            # 视频节点
            if node.get("class_type") == "VHS_LoadVideo":
                node["inputs"]["video"] = vid_name
                for key in ("force_rate", "frame_load_cap", "skip_first_frames", "select_every_nth"):
                    if key in node["inputs"]:
                        node["inputs"][key] = effective_frame_cap if key == "frame_load_cap" else window[key]

            if node.get("class_type") in {"SCAIL2FitVideo", "ImageResizeKJv2"}:
                inputs = node.get("inputs", {})
                if node.get("class_type") == "SCAIL2FitVideo" and "resolution" in inputs:
                    inputs["resolution"] = "custom"
                for key in ("custom_width", "width"):
                    if key in inputs:
                        inputs[key] = width
                for key in ("custom_height", "height"):
                    if key in inputs:
                        inputs[key] = height
                if nid == "ref_resize":
                    inputs["width"] = self._reference_collage_width(subject_count, width)
                    inputs["height"] = height

            if node.get("class_type") == "VHS_VideoCombine" and "frame_rate" in node.get("inputs", {}):
                node["inputs"]["frame_rate"] = window["force_rate"]

            if node.get("class_type") == "BasicScheduler":
                node["inputs"]["steps"] = sampler["steps"]
                if "denoise" in node["inputs"]:
                    node["inputs"]["denoise"] = sampler["denoise"]
            if node.get("class_type") == "WanAnimatePlus SamplerSettings":
                node["inputs"]["steps"] = sampler["steps"]
                node["inputs"]["seed"] = int(time.time() * 1000) % 2_147_483_647

            if node.get("class_type") == "EmptyImage":
                node["inputs"]["width"] = width
                node["inputs"]["height"] = height
                node["inputs"]["batch_size"] = effective_frame_cap

            if node.get("class_type") == "SCAIL2ColoredMaskV2":
                node["inputs"]["replacement_mode"] = False
                if subject_count > 1 and "object_indices" in node.get("inputs", {}):
                    node["inputs"]["object_indices"] = ",".join(map(str, normalized_driving_indices))
            if node.get("class_type") == "SCAIL2ColoredMask":
                if subject_count > 1 and "object_indices" in node.get("inputs", {}):
                    node["inputs"]["object_indices"] = ",".join(map(str, normalized_driving_indices))

            # 正向提示词
            if node.get("class_type") == "CLIPTextEncode" and nid == "17":
                node["inputs"]["text"] = positive_prompt
            if node.get("class_type") == "WanVideoTextEncodeCached":
                node["inputs"]["positive_prompt"] = positive_prompt

            # SAM3 跟踪文本
            if node.get("class_type") == "CLIPTextEncode" and nid == "31":
                node["inputs"]["text"] = sam_text

            # 输出分辨率
            if node.get("class_type") == "SCAIL2SimpleVideo":
                for key in ("chunk_frames", "context_frames", "overlap_frames", "context_overlap_frames"):
                    if key in node.get("inputs", {}):
                        node["inputs"][key] = sampler[key]
                if node.get("inputs", {}).get("width") is not None:
                    node["inputs"]["width"] = width
                    node["inputs"]["height"] = height
            if node.get("class_type") == "WanSCAILToVideo":
                node["inputs"]["width"] = width
                node["inputs"]["height"] = height
                node["inputs"]["length"] = effective_frame_cap
                node["inputs"]["pose_strength"] = float(sampler["pose_strength"])
                node["inputs"]["previous_frame_count"] = max(
                    1, int(node["inputs"].get("previous_frame_count") or 5)
                )
                node["inputs"]["replacement_mode"] = True
            if node.get("class_type") == "WanAnimatePlus SCAIL_2 Embeds":
                node["inputs"]["width"] = width
                node["inputs"]["height"] = height
                node["inputs"]["num_frames"] = effective_frame_cap
                node["inputs"]["frame_window_size"] = min(
                    effective_frame_cap, int(sampler["chunk_frames"])
                )
                node["inputs"]["pose_strength"] = float(sampler["pose_strength"])
                node["inputs"]["ref_strength"] = float(sampler["ref_strength"])
                node["inputs"]["replacement_mode"] = False
            if node.get("class_type") == "WanAnimatePlus Bernini":
                node["inputs"]["width"] = width
                node["inputs"]["height"] = height
                node["inputs"]["num_frames"] = effective_frame_cap
            if node.get("class_type") == "SamplerCustom":
                node["inputs"]["noise_seed"] = int(time.time() * 1000) % 2_147_483_647
                node["inputs"]["cfg"] = 1
            if nid == "ref_load" and reference_collage_name:
                node["inputs"]["image"] = reference_collage_name

        # SCAIL2ReferencePack 的主体与补充参考必须分开，不能拿原角色图做主体。
        for node in wf.values():
            if node.get("class_type") == "SCAIL2ReferencePack":
                inputs = node["inputs"]
                for key in list(inputs):
                    if key.startswith("subject_") or key.startswith("reference_"):
                        inputs.pop(key)
                inputs["subject_count"] = subject_count
                inputs["reference_count"] = len(extra_ref_names)
                for i in range(1, subject_count + 1):
                    source = f"subject_resize_{i-1}" if normalize_reference else f"subject_load_{i-1}"
                    inputs[f"subject_{i}_image"] = [source, 0]
                for i in range(1, len(extra_ref_names) + 1):
                    source = f"extra_ref_resize_{i-1}" if normalize_reference else f"extra_ref_load_{i-1}"
                    inputs[f"reference_{i}"] = [source, 0]

            if node.get("class_type") == "SAM3_VideoTrack":
                # 只跟踪要替换的人数，避免额外路人占用没有主体图的颜色。
                node["inputs"]["max_objects"] = max(1, subject_count)
                if "detection_threshold" in node["inputs"]:
                    images_input = node["inputs"].get("images")
                    node["inputs"]["detection_threshold"] = (
                        SAM3_REFERENCE_DETECTION_THRESHOLD
                        if images_input == ["ref_resize", 0]
                        else SAM3_VIDEO_DETECTION_THRESHOLD
                    )
                if workflow_uses_colored_mask:
                    images_input = node["inputs"].get("images")
                    if images_input == ["3", 0]:
                        node["inputs"]["detect_interval"] = SAM3_VIDEO_DETECT_INTERVAL
                    if images_input == ["ref_resize", 0]:
                        node["inputs"]["detect_interval"] = SAM3_REFERENCE_DETECT_INTERVAL
                elif subject_count > 1 and node.get("inputs", {}).get("images") == ["3", 0]:
                    node["inputs"]["detect_interval"] = SAM3_VIDEO_DETECT_INTERVAL
            if node.get("class_type") == "SCAIL2ReferenceSAMBuilder":
                # ReferencePack has one clean person per subject image, but the
                # builder must create one reference mask for every subject.
                node["inputs"]["max_objects"] = max(1, subject_count)
                if "detection_threshold" in node["inputs"]:
                    node["inputs"]["detection_threshold"] = SAM3_REFERENCE_DETECTION_THRESHOLD

        if subject_count > 2 and any(
            node.get("class_type") == "SCAIL2SimpleVideo" for node in wf.values()
        ):
            self._add_interior_role_repairs(
                wf,
                subject_count=subject_count,
                normalize_reference=normalize_reference,
                driving_object_indices=driving_object_indices,
                subject_appearance_hints=subject_appearance_hints,
                subject_extra_ref_names=subject_extra_ref_names,
            )

        return wf

    @staticmethod
    def _add_interior_role_repairs(
        wf: dict,
        *,
        subject_count: int,
        normalize_reference: bool,
        driving_object_indices: list[int] | None = None,
        subject_appearance_hints: list[str] | None = None,
        subject_extra_ref_names: list[list[str]] | None = None,
    ) -> None:
        base_node_id = next(
            node_id
            for node_id, node in wf.items()
            if node.get("class_type") == "SCAIL2SimpleVideo"
        )
        driving_track_id = next(
            node_id
            for node_id, node in wf.items()
            if node.get("class_type") == "SAM3_VideoTrack"
            and node.get("inputs", {}).get("images") == ["3", 0]
        )
        normalized_indices = list(driving_object_indices or range(subject_count))
        if len(normalized_indices) != subject_count:
            normalized_indices = list(range(subject_count))
        normalized_hints = list(subject_appearance_hints or [])
        if len(normalized_hints) < subject_count:
            normalized_hints.extend([""] * (subject_count - len(normalized_hints)))
        composite_source: list[str | int] = [base_node_id, 0]

        for subject_index in range(1, subject_count - 1):
            prefix = f"role_repair_{subject_index}"
            reference_source = (
                f"subject_resize_{subject_index}"
                if normalize_reference
                else f"subject_load_{subject_index}"
            )
            reference_image_source: list[str | int] = [reference_source, 0]
            raw_index = normalized_indices[subject_index]
            wf[f"{prefix}_all_frames_mask"] = {
                "class_type": "SAM3_TrackToMask",
                "inputs": {
                    "track_data": [driving_track_id, 0],
                    "object_indices": str(raw_index),
                },
            }
            wf[f"{prefix}_initial_mask"] = {
                "class_type": "VHS_SelectMasks",
                "inputs": {
                    "mask": [f"{prefix}_all_frames_mask", 0],
                    "indexes": "0",
                    "err_if_missing": True,
                    "err_if_empty": True,
                },
            }
            wf[f"{prefix}_driving_track"] = {
                "class_type": "SAM3_VideoTrack",
                "inputs": {
                    "detection_threshold": SAM3_VIDEO_DETECTION_THRESHOLD,
                    "max_objects": 1,
                    "detect_interval": 1,
                    "images": ["3", 0],
                    "model": ["30", 0],
                    "conditioning": ["31", 0],
                    "initial_mask": [f"{prefix}_initial_mask", 0],
                },
            }
            wf[f"{prefix}_reference_track"] = {
                "class_type": "SAM3_VideoTrack",
                "inputs": {
                    "detection_threshold": SAM3_REFERENCE_DETECTION_THRESHOLD,
                    "max_objects": 1,
                    "detect_interval": 1,
                    "images": [reference_source, 0],
                    "model": ["30", 0],
                    "conditioning": ["31", 0],
                },
            }
            wf[f"{prefix}_positive"] = {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": (
                        _INTERIOR_REPAIR_PROMPT
                        + " Preserve the target subject identity from the single full-body reference; "
                        + "do not import background or scale from any auxiliary face reference."
                        + normalized_hints[subject_index]
                    ),
                    "clip": ["16", 0],
                },
            }
            wf[f"{prefix}_scheduler"] = {
                "class_type": "BasicScheduler",
                "inputs": {
                    "scheduler": "simple",
                    "steps": 6,
                    "denoise": 1.0,
                    "model": ["12", 0],
                },
            }

            repair_inputs = copy.deepcopy(wf[base_node_id]["inputs"])
            repair_inputs.update(
                {
                    "positive": [f"{prefix}_positive", 0],
                    "sigmas": [f"{prefix}_scheduler", 0],
                    "reference_image": reference_image_source,
                    "driving_track_data": [f"{prefix}_driving_track", 0],
                    "reference_track_data": [f"{prefix}_reference_track", 0],
                    "mode": "replacement",
                    "chunk_frames": 49,
                    "context_frames": 49,
                    "context_overlap_frames": 12,
                }
            )
            wf[f"{prefix}_generate"] = {
                "class_type": "SCAIL2SimpleVideo",
                "inputs": repair_inputs,
            }
            wf[f"{prefix}_tracked_mask"] = {
                "class_type": "SAM3_TrackToMask",
                "inputs": {
                    "track_data": [f"{prefix}_driving_track", 0],
                    "object_indices": "",
                },
            }
            # The single-person tracker can drift during a close interaction.
            # Keep every repair inside its original multi-person track region.
            wf[f"{prefix}_constrained_mask"] = {
                "class_type": "MaskComposite",
                "inputs": {
                    "destination": [f"{prefix}_all_frames_mask", 0],
                    "source": [f"{prefix}_tracked_mask", 0],
                    "x": 0,
                    "y": 0,
                    "operation": "multiply",
                },
            }
            wf[f"{prefix}_soft_mask"] = {
                "class_type": "GrowMaskWithBlur",
                "inputs": {
                    "mask": [f"{prefix}_constrained_mask", 0],
                    "expand": 2,
                    "incremental_expandrate": 0.0,
                    "tapered_corners": True,
                    "flip_input": False,
                    "blur_radius": 3.0,
                    "lerp_alpha": 1.0,
                    "decay_factor": 1.0,
                    "fill_holes": False,
                },
            }
            wf[f"{prefix}_composite"] = {
                "class_type": "ImageCompositeMasked",
                "inputs": {
                    "destination": composite_source,
                    "source": [f"{prefix}_generate", 0],
                    "mask": [f"{prefix}_soft_mask", 0],
                    "x": 0,
                    "y": 0,
                    "resize_source": False,
                },
            }
            composite_source = [f"{prefix}_composite", 0]

        wf["43"]["inputs"]["images"] = composite_source

    def _missing_advanced_nodes(self) -> list[str]:
        try:
            from .comfy_inventory import load_inventory

            cached = load_inventory(self.comfy)
            if cached:
                available = set((cached.get("nodes") or {}).keys())
                missing = sorted(_ADVANCED_SCAIL2_NODES - available)
                if not missing:
                    return []
        except Exception:
            pass

        missing: list[str] = []
        for class_name in sorted(_ADVANCED_SCAIL2_NODES):
            try:
                response = self.session.get(f"{self.comfy}/object_info/{class_name}", timeout=10)
                if response.status_code == 404:
                    missing.append(class_name)
                    continue
                response.raise_for_status()
            except requests.RequestException:
                missing.append(class_name)
        return missing

    def _missing_wananimate_nodes(self) -> list[str]:
        try:
            from .comfy_inventory import load_inventory

            cached = load_inventory(self.comfy)
            if cached:
                available = set((cached.get("nodes") or {}).keys())
                missing = sorted(_WANANIMATE_SCAIL2_NODES - available)
                if not missing:
                    return []
        except Exception:
            pass

        missing: list[str] = []
        for class_name in sorted(_WANANIMATE_SCAIL2_NODES):
            try:
                response = self.session.get(f"{self.comfy}/object_info/{class_name}", timeout=10)
                if response.status_code == 404:
                    missing.append(class_name)
                    continue
                response.raise_for_status()
            except requests.RequestException:
                missing.append(class_name)
        return missing

    def _missing_wananimate_mask_only_nodes(self) -> list[str]:
        try:
            from .comfy_inventory import load_inventory

            cached = load_inventory(self.comfy)
            if cached:
                available = set((cached.get("nodes") or {}).keys())
                missing = sorted(_WANANIMATE_MASK_ONLY_NODES - available)
                if not missing:
                    return []
        except Exception:
            pass

        missing: list[str] = []
        for class_name in sorted(_WANANIMATE_MASK_ONLY_NODES):
            try:
                response = self.session.get(f"{self.comfy}/object_info/{class_name}", timeout=10)
                if response.status_code == 404:
                    missing.append(class_name)
                    continue
                response.raise_for_status()
            except requests.RequestException:
                missing.append(class_name)
        return missing

    def _missing_vda_white_mask_nodes(self) -> list[str]:
        missing: list[str] = []
        for class_name in sorted(_VDA_WHITE_MASK_NODES):
            try:
                response = self.session.get(f"{self.comfy}/object_info/{class_name}", timeout=10)
                if response.status_code == 404:
                    missing.append(class_name)
                    continue
                response.raise_for_status()
            except requests.RequestException:
                missing.append(class_name)
        return missing

    def _missing_dwpose_video_nodes(self) -> list[str]:
        missing: list[str] = []
        for class_name in sorted(_DWPOSE_VIDEO_NODES):
            try:
                response = self.session.get(f"{self.comfy}/object_info/{class_name}", timeout=10)
                if response.status_code == 404:
                    missing.append(class_name)
                    continue
                response.raise_for_status()
            except requests.RequestException:
                missing.append(class_name)
        return missing

    def _missing_bernini_nodes(self) -> list[str]:
        try:
            from .comfy_inventory import load_inventory

            cached = load_inventory(self.comfy)
            if cached:
                available = set((cached.get("nodes") or {}).keys())
                missing = sorted(_BERNINI_TEST_NODES - available)
                if not missing:
                    return []
        except Exception:
            pass

        missing: list[str] = []
        for class_name in sorted(_BERNINI_TEST_NODES):
            try:
                response = self.session.get(f"{self.comfy}/object_info/{class_name}", timeout=10)
                if response.status_code == 404:
                    missing.append(class_name)
                    continue
                response.raise_for_status()
            except requests.RequestException:
                missing.append(class_name)
        return missing

    def _create_reference_collage(
        self,
        ref_images: list[str],
        *,
        output_dir: Path,
        width: int,
        height: int,
        role_names: list[str],
    ) -> Path:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required to build SCAIL-2 reference collage") from exc

        subject_count = len(ref_images)
        canvas_width = self._reference_collage_width(subject_count, width)
        canvas_height = max(16, int(height))
        slot_width = max(16, canvas_width // max(1, subject_count))
        margin_x = max(4, int(slot_width * 0.01))
        margin_y = 0
        canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))

        for index, image_path in enumerate(ref_images):
            with Image.open(image_path) as raw:
                image = raw.convert("RGBA")
                max_w = max(16, slot_width - margin_x * 2)
                max_h = max(16, canvas_height - margin_y * 2)
                scale = min(max_w / max(1, image.width), max_h / max(1, image.height))
                resized = image.resize(
                    (
                        max(1, int(round(image.width * scale))),
                        max(1, int(round(image.height * scale))),
                    ),
                    Image.Resampling.LANCZOS,
                )
                layer = Image.new("RGBA", (slot_width, canvas_height), (255, 255, 255, 255))
                x = (slot_width - resized.width) // 2
                y = canvas_height - margin_y - resized.height
                if y < margin_y:
                    y = (canvas_height - resized.height) // 2
                layer.alpha_composite(resized, (x, y))
                canvas.paste(layer.convert("RGB"), (index * slot_width, 0))

        role_part = "_".join(_safe_slug(name) for name in role_names if name) or "roles"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"scail2_reference_collage_{role_part}_{stamp}.png"
        canvas.save(path)
        return path

    def _create_source_position_reference_collage(
        self,
        ref_images: list[str],
        *,
        output_dir: Path,
        width: int,
        height: int,
        role_names: list[str],
        source_positions: list[float | None] | None = None,
    ) -> Path:
        """Place reference people near their source-video x positions.

        The colored-mask generation path treats the reference image as a strong
        spatial hint. A side-by-side reference collage can make the result split
        people apart, so this variant keeps the reference identities near the
        original source layout.
        """
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required to build SCAIL-2 reference collage") from exc

        subject_count = len(ref_images)
        canvas_width = self._reference_collage_width(subject_count, width)
        canvas_height = max(16, int(height))
        positions = list(source_positions or [])
        if len(positions) < subject_count:
            positions.extend([None] * (subject_count - len(positions)))
        fallback_step = 1.0 / max(1, subject_count + 1)
        canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
        max_subject_width = max(96, int(canvas_width * (0.42 if subject_count <= 2 else 0.34)))
        max_subject_height = max(16, int(canvas_height * 0.98))

        for index, image_path in enumerate(ref_images):
            raw_position = positions[index]
            if isinstance(raw_position, (int, float)) and 0.0 <= float(raw_position) <= 1.0:
                center_x = int(round(float(raw_position) * canvas_width))
            else:
                center_x = int(round((index + 1) * fallback_step * canvas_width))

            with Image.open(image_path) as raw:
                image = raw.convert("RGBA")
                scale = min(
                    max_subject_width / max(1, image.width),
                    max_subject_height / max(1, image.height),
                )
                resized = image.resize(
                    (
                        max(1, int(round(image.width * scale))),
                        max(1, int(round(image.height * scale))),
                    ),
                    Image.Resampling.LANCZOS,
                )
                x = max(0, min(canvas_width - resized.width, center_x - resized.width // 2))
                y = canvas_height - resized.height
                layer = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 0))
                layer.alpha_composite(resized, (x, y))
                canvas.paste(layer.convert("RGB"), (0, 0), layer.getchannel("A"))

        role_part = "_".join(_safe_slug(name) for name in role_names if name) or "roles"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"scail2_reference_sourcepos_{role_part}_{stamp}.png"
        canvas.save(path)
        return path

    def _create_reference_collage_from_pose_mask(
        self,
        ref_images: list[str],
        *,
        pose_mask_path: str,
        background_image_path: str = "",
        output_dir: Path,
        width: int,
        height: int,
        role_names: list[str],
    ) -> Path:
        try:
            from PIL import Image, ImageStat
        except ImportError as exc:
            raise RuntimeError("Pillow is required to build SCAIL-2 reference collage") from exc

        def trim_rgba(image: Image.Image) -> Image.Image:
            rgba = image.convert("RGBA")
            alpha = rgba.getchannel("A")
            bbox = alpha.getbbox()
            if bbox:
                return rgba.crop(bbox)
            rgb = rgba.convert("RGB")
            bg = Image.new("RGB", rgb.size, (255, 255, 255))
            diff = Image.new("RGB", rgb.size)
            # White-background fallback: keep any non-near-white pixels.
            pixels = rgb.load()
            min_x = rgb.width
            min_y = rgb.height
            max_x = -1
            max_y = -1
            for yy in range(rgb.height):
                for xx in range(rgb.width):
                    r, g, b = pixels[xx, yy]
                    if max(r, g, b) < 245 or (255 - min(r, g, b)) > 24:
                        min_x = min(min_x, xx)
                        min_y = min(min_y, yy)
                        max_x = max(max_x, xx)
                        max_y = max(max_y, yy)
            if max_x >= min_x and max_y >= min_y:
                return rgba.crop((min_x, min_y, max_x + 1, max_y + 1))
            return rgba

        def mask_regions(mask_image: Image.Image) -> list[tuple[int, int, int, int]]:
            rgb = mask_image.convert("RGB")
            pixels = rgb.load()
            width_px, height_px = rgb.size
            boxes = {
                "blue": [width_px, height_px, -1, -1],
                "red": [width_px, height_px, -1, -1],
            }
            for yy in range(height_px):
                for xx in range(width_px):
                    r, g, b = pixels[xx, yy]
                    if b > 120 and r < 120 and g < 120:
                        box = boxes["blue"]
                    elif r > 120 and g < 120 and b < 120:
                        box = boxes["red"]
                    else:
                        continue
                    box[0] = min(box[0], xx)
                    box[1] = min(box[1], yy)
                    box[2] = max(box[2], xx)
                    box[3] = max(box[3], yy)
            regions: list[tuple[int, int, int, int]] = []
            for key in ("blue", "red"):
                x1, y1, x2, y2 = boxes[key]
                if x2 >= x1 and y2 >= y1:
                    regions.append((x1, y1, x2 + 1, y2 + 1))
            if len(regions) >= len(ref_images):
                return regions[: len(ref_images)]
            # Fallback: split by connected x clusters if color detection missed.
            return regions

        with Image.open(pose_mask_path) as pose_mask_raw:
            pose_mask = pose_mask_raw.convert("RGB")

        regions = mask_regions(pose_mask)
        if len(regions) < len(ref_images):
            return self._create_source_position_reference_collage(
                ref_images,
                output_dir=output_dir,
                width=width,
                height=height,
                role_names=role_names,
                source_positions=[None] * len(ref_images),
            )

        if background_image_path and Path(background_image_path).exists():
            with Image.open(background_image_path) as background_raw:
                canvas = background_raw.convert("RGB").resize(pose_mask.size, Image.Resampling.LANCZOS)
        else:
            canvas = Image.new("RGB", pose_mask.size, (255, 255, 255))
        for index, image_path in enumerate(ref_images):
            region = regions[index]
            target_w = max(32, region[2] - region[0])
            target_h = max(32, region[3] - region[1])
            with Image.open(image_path) as raw:
                person = trim_rgba(raw)
                scale = min(
                    target_w / max(1, person.width),
                    target_h / max(1, person.height),
                )
                resized = person.resize(
                    (
                        max(1, int(round(person.width * scale))),
                        max(1, int(round(person.height * scale))),
                    ),
                    Image.Resampling.LANCZOS,
                )
                x = region[0] + max(0, (target_w - resized.width) // 2)
                y = region[1] + max(0, target_h - resized.height)
                layer = Image.new("RGBA", pose_mask.size, (255, 255, 255, 0))
                layer.alpha_composite(resized, (x, y))
                canvas.paste(layer.convert("RGB"), (0, 0), layer.getchannel("A"))

        role_part = "_".join(_safe_slug(name) for name in role_names if name) or "roles"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"scail2_reference_pose_layout_{role_part}_{stamp}.png"
        canvas.save(path)
        return path

    def _analyze_pose_mask_segments(
        self,
        pose_mask_path: str,
        *,
        output_dir: Path,
        expected_subjects: int,
        frame_rate: int | float,
        max_segment_frames: int,
        log: Callable[[str], None],
    ) -> list[dict]:
        if not pose_mask_path or not Path(pose_mask_path).exists():
            return []
        frames = self._extract_video_frames(
            pose_mask_path,
            output_dir=output_dir,
            prefix=f"pose_mask_frames_{Path(pose_mask_path).stem}_{uuid.uuid4().hex[:6]}",
        )
        if not frames:
            return []
        total = len(frames)
        max_segment_frames = max(8, int(max_segment_frames or total))
        min_segment_frames = max(8, min(16, int(round(float(frame_rate or 24) * 0.45))))
        if expected_subjects < 2:
            boundaries = [0, total]
        else:
            distances: list[float | None] = []
            for frame in frames:
                try:
                    from PIL import Image

                    with Image.open(frame) as image:
                        regions = self._colored_mask_regions(image, expected_subjects)
                except Exception:  # noqa: BLE001
                    regions = []
                if len(regions) >= 2:
                    width_px = max(1, regions[0]["image_width"])
                    centers = [
                        (region["box"][0] + region["box"][2]) / 2.0 / width_px
                        for region in regions[:2]
                    ]
                    distances.append(abs(centers[1] - centers[0]))
                else:
                    distances.append(None)

            valid = [value for value in distances if value is not None]
            boundaries = [0]
            if valid and max(valid) - min(valid) >= 0.14 and total >= min_segment_frames * 2:
                threshold = (max(valid) + min(valid)) / 2.0
                states: list[int] = []
                last_state = 0 if valid[0] < threshold else 1
                for value in distances:
                    if value is not None:
                        last_state = 0 if value < threshold else 1
                    states.append(last_state)

                run_start = 0
                run_state = states[0]
                for index, state in enumerate(states[1:], 1):
                    if state == run_state:
                        continue
                    run_length = index - run_start
                    remaining = total - index
                    if (
                        run_length >= min_segment_frames
                        and remaining >= min_segment_frames
                        and index - boundaries[-1] >= min_segment_frames
                    ):
                        boundaries.append(index)
                    run_start = index
                    run_state = state
            boundaries.append(total)

        capped_boundaries = [0]
        for boundary in boundaries[1:]:
            while boundary - capped_boundaries[-1] > max_segment_frames:
                capped_boundaries.append(capped_boundaries[-1] + max_segment_frames)
            if boundary > capped_boundaries[-1]:
                capped_boundaries.append(boundary)

        segments: list[dict] = []
        for start, end in zip(capped_boundaries, capped_boundaries[1:]):
            if end <= start:
                continue
            segments.append({
                "start_frame": start,
                "frame_count": end - start,
                "reference_frame": str(frames[start]),
            })
        if not segments:
            return []
        if len(segments) == 1:
            log(f"SCAIL-2 colored auto segments: 1 segment ({total} frames)")
        else:
            desc = ", ".join(
                f"{item['start_frame']}-{item['start_frame'] + item['frame_count'] - 1}"
                for item in segments
            )
            log(f"SCAIL-2 colored auto segments: {len(segments)} segments: {desc}")
        return segments

    @staticmethod
    def _extract_source_reference_frame(
        video_path: str,
        *,
        output_dir: Path,
        frame_index: int,
        video_window: dict,
        role_names: list[str],
        segment_index: int,
    ) -> Path | None:
        try:
            from .ffmpeg_tools import extract_frame

            force_rate = max(1, int(video_window.get("force_rate") or 24))
            nth = max(1, int(video_window.get("select_every_nth") or 1))
            skip = max(0, int(video_window.get("skip_first_frames") or 0))
            source_frame_index = skip + max(0, int(frame_index)) * nth
            time_seconds = source_frame_index / force_rate
            role_part = "_".join(_safe_slug(name) for name in role_names if name) or "roles"
            target = output_dir / (
                f"scail2_reference_source_frame_{role_part}_seg{segment_index:02d}_{source_frame_index:04d}.jpg"
            )
            extract_frame(video_path, time_seconds, target)
            return target
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _extract_video_frames(
        video_path: str,
        *,
        output_dir: Path,
        prefix: str,
    ) -> list[Path]:
        from .ffmpeg_tools import ffmpeg_path, run_command

        frame_dir = output_dir / f"_{_safe_slug(prefix, 60)}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for stale in frame_dir.glob("frame_*.png"):
            stale.unlink(missing_ok=True)
        pattern = frame_dir / "frame_%04d.png"
        run_command([
            ffmpeg_path(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-start_number",
            "0",
            str(pattern),
        ])
        return sorted(frame_dir.glob("frame_*.png"))

    @staticmethod
    def _colored_mask_regions(mask_image, expected_subjects: int) -> list[dict]:
        rgb = mask_image.convert("RGB")
        pixels = rgb.load()
        width_px, height_px = rgb.size
        colors = [
            ("blue", lambda r, g, b: b > 120 and r < 120 and g < 120),
            ("red", lambda r, g, b: r > 120 and g < 120 and b < 120),
            ("green", lambda r, g, b: g > 120 and r < 140 and b < 140),
            ("magenta", lambda r, g, b: r > 120 and b > 120 and g < 120),
            ("cyan", lambda r, g, b: g > 120 and b > 120 and r < 120),
            ("yellow", lambda r, g, b: r > 120 and g > 120 and b < 120),
        ][: max(1, expected_subjects)]
        boxes = {
            name: [width_px, height_px, -1, -1, 0]
            for name, _ in colors
        }
        for yy in range(height_px):
            for xx in range(width_px):
                r, g, b = pixels[xx, yy]
                for name, predicate in colors:
                    if predicate(r, g, b):
                        box = boxes[name]
                        box[0] = min(box[0], xx)
                        box[1] = min(box[1], yy)
                        box[2] = max(box[2], xx)
                        box[3] = max(box[3], yy)
                        box[4] += 1
                        break
        regions: list[dict] = []
        min_area = max(32, int(width_px * height_px * 0.0005))
        for name, _ in colors:
            x1, y1, x2, y2, count = boxes[name]
            if x2 >= x1 and y2 >= y1 and count >= min_area:
                regions.append({
                    "color": name,
                    "box": (x1, y1, x2 + 1, y2 + 1),
                    "area": count,
                    "image_width": width_px,
                    "image_height": height_px,
                })
        return regions

    @staticmethod
    def _concat_video_segments(
        segment_paths: list[Path],
        output_path: Path,
        *,
        log: Callable[[str], None],
    ) -> None:
        from .ffmpeg_tools import FfmpegError, ffmpeg_path, run_command

        if not segment_paths:
            raise RuntimeError("no colored segments to concatenate")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        list_path = output_path.with_suffix(".concat.txt")
        with list_path.open("w", encoding="utf-8") as handle:
            for path in segment_paths:
                escaped = str(path.resolve()).replace("\\", "/").replace("'", "'\\''")
                handle.write(f"file '{escaped}'\n")
        log(f"SCAIL-2 colored auto segments: stitching {len(segment_paths)} clips...")
        try:
            run_command([
                ffmpeg_path(),
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                str(output_path),
            ])
        except FfmpegError:
            run_command([
                ffmpeg_path(),
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                str(output_path),
            ])

    @staticmethod
    def _reference_collage_width(subject_count: int, width: int) -> int:
        per_subject = max(512, width)
        desired = max(width, per_subject * max(1, subject_count))
        return max(16, int(math.ceil(min(desired, 1920) / 16)) * 16)

    @staticmethod
    def _save_workflow_debug(
        wf: dict,
        *,
        output_dir: Path,
        video_path: str,
        role_names: list[str],
    ) -> Path:
        role_part = "_".join(_safe_slug(name) for name in role_names if name)
        if not role_part:
            role_part = "roles"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"scail2_workflow_{Path(video_path).stem}_{role_part}_{stamp}.json"
        path.write_text(json.dumps(wf, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def _reference_resize_node(source_node: str, width: int, height: int) -> dict:
        return {
            "class_type": "ImageResizeKJv2",
            "inputs": {
                "image": [source_node, 0],
                "width": width,
                "height": height,
                "upscale_method": "lanczos",
                "keep_proportion": "crop",
                "pad_color": "0, 0, 0",
                "crop_position": "center",
                "divisible_by": 16,
                "device": "cpu",
            },
        }

    @staticmethod
    def _reference_clothing_hint(image_path: str) -> str:
        """Describe a reference outfit's dominant color without hard-coding a role."""
        try:
            import colorsys
            from PIL import Image

            with Image.open(image_path) as image:
                image = image.convert("RGB")
                width, height = image.size
                crop = image.crop((
                    int(width * 0.32),
                    int(height * 0.32),
                    int(width * 0.68),
                    int(height * 0.72),
                )).resize((64, 64))

            scores = {
                "red": 0.0,
                "orange": 0.0,
                "yellow": 0.0,
                "green": 0.0,
                "blue": 0.0,
                "purple": 0.0,
                "pink": 0.0,
            }
            pixels = (
                crop.get_flattened_data()
                if hasattr(crop, "get_flattened_data")
                else crop.getdata()
            )
            for red, green, blue in pixels:
                hue, saturation, value = colorsys.rgb_to_hsv(
                    red / 255.0,
                    green / 255.0,
                    blue / 255.0,
                )
                if saturation < 0.32 or value < 0.18:
                    continue
                if hue < 0.045 or hue >= 0.965:
                    label = "red"
                elif hue < 0.105:
                    label = "orange"
                elif hue < 0.19:
                    label = "yellow"
                elif hue < 0.47:
                    label = "green"
                elif hue < 0.72:
                    label = "blue"
                elif hue < 0.88:
                    label = "purple"
                else:
                    label = "pink"
                scores[label] += saturation * (0.5 + value)

            color, score = max(scores.items(), key=lambda item: item[1])
            if score < 100.0:
                return ""
            return (
                f" The reference subject wears a {color} outfit; reproduce the "
                f"same {color} clothing exactly."
            )
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _resolve_video_window(meta, requested: dict | None) -> dict[str, int]:
        requested = requested or {}
        force_rate = int(requested.get("force_rate") or round(meta.fps or 24))
        force_rate = max(1, min(force_rate, 60))
        skip = max(0, int(requested.get("skip_first_frames") or 0))
        nth = max(1, min(int(requested.get("select_every_nth") or 1), 12))
        available = max(1, int(math.ceil(max(0.0, meta.duration) * force_rate / nth)) - skip)
        cap = int(requested.get("frame_load_cap") or available)
        return {
            "force_rate": force_rate,
            "frame_load_cap": max(1, min(cap, available, 241)),
            "skip_first_frames": skip,
            "select_every_nth": nth,
        }

    @staticmethod
    def _normalized_size(source_width: int, source_height: int, width: int, height: int) -> tuple[int, int]:
        if source_width <= 0 or source_height <= 0:
            return width, height
        target_long = max(width, height)
        scale = target_long / max(source_width, source_height)
        normalized_width = max(16, int(round(source_width * scale / 16)) * 16)
        normalized_height = max(16, int(round(source_height * scale / 16)) * 16)
        return normalized_width, normalized_height

    @staticmethod
    def _sampler_settings(preset: str) -> dict[str, int | float]:
        presets = {
            "fast": {
                "steps": 6, "denoise": 1.0, "pose_strength": 1.0, "ref_strength": 1.0, "chunk_frames": 49,
                "context_frames": 49, "overlap_frames": 5, "context_overlap_frames": 12,
            },
            "balanced": {
                "steps": 8, "denoise": 1.0, "pose_strength": 1.0, "ref_strength": 1.0, "chunk_frames": 81,
                "context_frames": 81, "overlap_frames": 5, "context_overlap_frames": 20,
            },
            "quality": {
                "steps": 12, "denoise": 1.0, "pose_strength": 1.0, "ref_strength": 1.0, "chunk_frames": 81,
                "context_frames": 81, "overlap_frames": 9, "context_overlap_frames": 24,
            },
        }
        if preset not in presets:
            raise ValueError("invalid_sampler_preset")
        return presets[preset]


def _safe_slug(value: str, max_length: int = 32) -> str:
    slug = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", str(value or "")).strip("_")
    return (slug or "role")[:max_length]


def _content_addressed_name(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    suffix = path.suffix.lower() or ".bin"
    stem = _safe_slug(path.stem, max_length=48)
    return f"{stem}_{digest.hexdigest()[:12]}{suffix}"


def test():
    """连通性测试"""
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    client = Scail2Client()
    resp = client.session.get(f"{COMFY_URL}/system_stats", timeout=10)
    data = resp.json()
    print(f"ComfyUI: {data['system']['comfyui_version']}")
    print(f"GPU: {data['devices'][0]['name']}")
    print(f"VRAM free: {data['devices'][0]['vram_free'] / 1024**3:.1f} GB")
    print("OK")


if __name__ == "__main__":
    test()
