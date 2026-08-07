"""MiniMax H3 生成后端（Mode 2「原视频→白膜→结果视频」链路的新执行器）。

封装远程 ComfyUI 上已验证的两个 MiniMax H3 API 工作流：

- 白膜生成（R2V：参考视频 → 彩色人偶白膜），模板 comfy_workflows/h3_whitemask_api.json
- 白膜换人（白膜视频 + 人物参考图 → 结果视频），模板 comfy_workflows/h3_charswap_api.json

独立于 Mode 1 的传输/渲染流程，仅复用通用 ComfyClient 的 HTTP 能力。
base_url 默认取环境变量 H3_COMFY_URL，缺省为远程 8189 ComfyUI。
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import secrets
import uuid
from pathlib import Path
from typing import Any, Callable

from spvideo.comfy_client import ComfyClient

DEFAULT_H3_COMFY_URL = "https://8189-cpod-1tpdn4punkor-s1.pod.compshare.cn"  # 5090 独占机（2026-08-06 起）

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "comfy_workflows"
WHITEMASK_TEMPLATE = TEMPLATE_DIR / "h3_whitemask_api.json"
CHARSWAP_TEMPLATE = TEMPLATE_DIR / "h3_charswap_api.json"

# API prompt 模板里的关键节点 id
NODE_LOAD_VIDEO = "140"
NODE_H3 = "136"
NODE_SCHEDULER = "124"
NODE_GUIDER = "126"
NODE_UNET = "127"
NODE_NOISE = "129"
NODE_SAVE_VIDEO = "92"
NODE_SAMPLER = "125"
NODE_KSAMPLER_SELECT = "123"
NODE_VIDEO_DECODE = "122"
NODE_AUDIO_DECODE = "121"
NODE_CREATE_VIDEO = "130"
NODE_SAGE = "204"
NODE_BLOCKCACHE = "205"
NODE_FLASHVSR = "206"
LOAD_IMAGE_BASE_ID = 150  # 150/151/152...

# 5090 生产链路固定为：裁剪 INT8 UNet -> SageAttentionPatch -> BlockCache -> 标准 20 步采样，
# 视频解码后再经 FlashVSR 2x 输出。模型、步数和缓存阈值均不可覆盖；缺少任一生产
# 节点时直接报错，不切换模型或回退到其他加速路线。
SAGE_CLASS = "MiniMaxH3MemoryEfficientSageAttentionPatch"
BLOCKCACHE_CLASS = "MiniMaxH3BlockCacheT8"
FLASHVSR_CLASS = "FlashVSRNode"
BLOCKCACHE_THRESHOLD = 0.12
H3_PRODUCTION_STEPS = 20
PRUNED_UNET_NAME = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
FLASHVSR_MODEL = "FlashVSR-v1.1"
FLASHVSR_MODE = "tiny"
FLASHVSR_SCALE = 2

H3_TARGET_PIXELS = 400_000
H3_SIZE_MULTIPLE = 32
H3_SIZE_AREA_TOLERANCE = 0.05

MAX_CHAR_IMAGES = 3

# H3 输出帧率固定 24fps，且帧数必须落在 17k+5 网格上（5/22/39/.../107/124）
H3_FPS = 24
H3_FRAME_STEP = 17
H3_FRAME_BASE = 5

_FULL_FFMPEG_PATH: str | None = None


def _resolve_h3_base_url(explicit: str | None = None) -> str:
    """确定 H3 ComfyUI 地址：显式传入 > 健康的环境变量 > 默认值。

    H3 的 pod 是临时的（compshare），旧 pod 停止后环境变量指向的地址会 404。
    这里做一次存活探测：环境变量指向的地址不可用就回落到默认值，避免任务
    全部打到死 pod 上失败。探测结果缓存 60 秒，避免每次提交都多一次网络请求。
    """
    if explicit:
        return explicit
    env_url = os.environ.get("H3_COMFY_URL")
    if env_url and _h3_url_alive(env_url):
        return env_url
    return DEFAULT_H3_COMFY_URL


_h3_url_health_cache: dict[str, bool] = {}
_h3_url_health_checked_at: float = 0.0
_H3_URL_HEALTH_TTL = 60.0


def _h3_url_alive(base_url: str) -> bool:
    global _h3_url_health_checked_at
    import time

    now = time.time()
    if base_url in _h3_url_health_cache and now - _h3_url_health_checked_at < _H3_URL_HEALTH_TTL:
        return _h3_url_health_cache[base_url]
    alive = False
    try:
        import requests

        response = requests.get(base_url.rstrip("/") + "/system_stats", timeout=8)
        alive = response.status_code == 200
    except Exception:  # noqa: BLE001
        alive = False
    _h3_url_health_cache[base_url] = alive
    _h3_url_health_checked_at = now
    return alive


def snap_frame_count(frames: int) -> int:
    """把帧数向上吸附到 17k+5 网格，最少 5 帧。

    必须向上取整：向下会把源片尾部内容截掉（4.68s 源片被截成 4.46s 的教训），
    向上则用末帧克隆（tpad）补齐，内容完整；多出的定格尾可在下游按源时长裁回。
    """
    frames = max(H3_FRAME_BASE, int(frames))
    k = max(0, math.ceil((frames - H3_FRAME_BASE) / H3_FRAME_STEP))
    return H3_FRAME_STEP * k + H3_FRAME_BASE


def derive_h3_base_size(
    source_width: int,
    source_height: int,
    target_pixels: int = H3_TARGET_PIXELS,
) -> tuple[int, int]:
    """按源宽高比推导约 0.4MP、宽高均为 32 倍数的 H3 基础尺寸。"""
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"source dimensions must be positive, got {source_width}x{source_height}")
    if target_pixels <= 0:
        raise ValueError(f"target_pixels must be positive, got {target_pixels}")

    source_ratio = source_width / source_height
    target_units = target_pixels / (H3_SIZE_MULTIPLE**2)
    min_product = max(1, math.ceil(target_units * (1.0 - H3_SIZE_AREA_TOLERANCE)))
    max_product = max(min_product, math.floor(target_units * (1.0 + H3_SIZE_AREA_TOLERANCE)))

    best: tuple[tuple[float, float], int, int] | None = None
    for width_units in range(1, max_product + 1):
        min_height_units = max(1, math.ceil(min_product / width_units))
        max_height_units = math.floor(max_product / width_units)
        for height_units in range(min_height_units, max_height_units + 1):
            candidate_ratio = width_units / height_units
            ratio_error = abs(math.log(candidate_ratio / source_ratio))
            area_error = abs(width_units * height_units - target_units) / target_units
            candidate = ((ratio_error, area_error), width_units, height_units)
            if best is None or candidate[0] < best[0]:
                best = candidate

    if best is None:
        raise RuntimeError("unable to derive H3 base dimensions")
    return best[1] * H3_SIZE_MULTIPLE, best[2] * H3_SIZE_MULTIPLE


def _resolve_h3_run_size(
    video_path: Path,
    width: int,
    height: int,
    log: Callable[[str], None],
) -> tuple[int, int]:
    from spvideo import ffmpeg_tools

    meta = ffmpeg_tools.probe_video(video_path)
    source_width = int(meta.width or 0)
    source_height = int(meta.height or 0)
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"H3 source video has invalid dimensions: {source_width}x{source_height}")
    if width <= 0 or height <= 0:
        width, height = derive_h3_base_size(source_width, source_height)
    else:
        width, height = int(width), int(height)
    log(
        f"> H3 尺寸: source {source_width}x{source_height} -> base {width}x{height} "
        f"-> FlashVSR 2x {width * FLASHVSR_SCALE}x{height * FLASHVSR_SCALE}"
    )
    return width, height


def _full_ffmpeg() -> str:
    """挑一个带 libx264 编码器和 tpad 滤镜的完整版 ffmpeg。

    remotion 自带的精简版 ffmpeg 没有滤镜和 libx264（whitematte_recipe 踩过坑），
    归一化必须用完整版；结果缓存。
    """
    global _FULL_FFMPEG_PATH
    if _FULL_FFMPEG_PATH:
        return _FULL_FFMPEG_PATH
    from spvideo import ffmpeg_tools

    candidates = [
        str(Path(__file__).resolve().parent.parent / "assets" / "ffmpeg-full" / "ffmpeg.exe"),
        *ffmpeg_tools.FFMPEG_CANDIDATES,
    ]
    checked: set[str] = set()
    for candidate in candidates:
        try:
            resolved = ffmpeg_tools.find_binary([candidate])
        except ffmpeg_tools.FfmpegError:
            continue
        if resolved in checked:
            continue
        checked.add(resolved)
        try:
            encoders = ffmpeg_tools.run_command([resolved, "-hide_banner", "-encoders"]).stdout
            filters = ffmpeg_tools.run_command([resolved, "-hide_banner", "-filters"]).stdout
        except ffmpeg_tools.FfmpegError:
            continue
        if "libx264" in encoders and "tpad" in filters:
            _FULL_FFMPEG_PATH = resolved
            return resolved
    raise ffmpeg_tools.FfmpegError("h3_normalize_no_full_ffmpeg: 找不到带 libx264/tpad 的完整版 ffmpeg")


def normalize_reference_video(
    clip_path: str | Path,
    work_dir: str | Path,
    *,
    length: int | None = None,
    audio_path: str | Path | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[Path, int]:
    """发送前时长对齐：把参考视频转成 24fps、帧数吸附到 17k+5 网格。

    H3 输出帧数只能是 17k+5 且帧率固定 24fps；参考视频不先对齐，输出时长就会和
    源片错位（2 秒源片被拉成 3 秒就是这么来的）。源片不足目标帧数时用末帧克隆
    补齐（tpad），保证内容完整且帧数精确。返回 (归一化视频路径, 对齐后的帧数)。

    音轨处理（H3 的 ref_video_audios 输入要求视频必须带音轨，用于驱动口型）：
    - audio_path 给定时（如外语配音），用该音轨替换原音轨，并按视频时长截断；
    - 未给定时保留源片原音轨；源片本身无音轨时补一条静音轨兜底。
    """
    from spvideo import ffmpeg_tools

    logger = log or (lambda _message: None)
    clip = Path(clip_path)
    meta = ffmpeg_tools.probe_video(clip)
    source_frames = int(round(meta.duration * H3_FPS)) if meta.duration > 0 else 0
    if length is not None:
        target_frames = snap_frame_count(length)
    elif source_frames >= H3_FRAME_BASE:
        target_frames = snap_frame_count(source_frames)
    else:
        target_frames = snap_frame_count(124)
    audio = Path(audio_path).resolve() if audio_path else None
    if audio is not None and not audio.is_file():
        raise FileNotFoundError(f"h3 audio not found: {audio}")
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    audio_tag = ""
    if audio is not None:
        audio_tag = "_a" + hashlib.md5(str(audio).encode("utf-8")).hexdigest()[:6]
    out_path = work / f"{clip.stem}_h3ref_{target_frames}f{audio_tag}.mp4"
    logger(
        f"> H3 时长对齐: 源 {meta.fps:.2f}fps {meta.duration:.2f}s"
        f" → 24fps {target_frames} 帧 ({target_frames / H3_FPS:.2f}s)"
    )
    if audio is not None:
        logger(f"> H3 口型音轨: {audio.name}")
    if out_path.is_file() and out_path.stat().st_size > 0:
        logger(f"> H3 时长对齐: 复用已归一化文件 {out_path.name}")
        return out_path, target_frames
    seconds = target_frames / H3_FPS
    args = [_full_ffmpeg(), "-y", "-i", str(clip)]
    if audio is not None:
        args += ["-i", str(audio), "-map", "0:v", "-map", "1:a"]
    elif meta.audio_codec:
        args += ["-map", "0:v", "-map", "0:a:0"]
    else:
        args += [
            "-f", "lavfi", "-t", f"{seconds:.3f}", "-i", "anullsrc=r=44100:cl=stereo",
            "-map", "0:v", "-map", "1:a",
        ]
    args += [
        "-vf",
        f"fps={H3_FPS},tpad=stop_mode=clone:stop={H3_FRAME_STEP}",
        "-frames:v",
        str(target_frames),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-t",
        f"{seconds:.3f}",
        str(out_path),
    ]
    ffmpeg_tools.run_command(args)
    if not out_path.is_file() or out_path.stat().st_size <= 0:
        raise ffmpeg_tools.FfmpegError(f"h3_normalize_output_missing: {out_path}")
    return out_path, target_frames


def _load_template(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _random_seed() -> int:
    return secrets.randbelow(2**53)


def _require_production_steps(value: int) -> int:
    try:
        steps = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"H3 production workflow steps must be {H3_PRODUCTION_STEPS}, got {value!r}") from exc
    if steps != H3_PRODUCTION_STEPS:
        raise ValueError(f"H3 production workflow steps must be {H3_PRODUCTION_STEPS}, got {steps}")
    return H3_PRODUCTION_STEPS


def _validate_production_workflow(workflow: dict[str, Any]) -> None:
    """拒绝模板漂移，确保实际采样和输出都经过固定生产链路。"""
    expected_classes = {
        NODE_UNET: "UNETLoader",
        NODE_SAGE: SAGE_CLASS,
        NODE_BLOCKCACHE: BLOCKCACHE_CLASS,
        NODE_SCHEDULER: "BasicScheduler",
        NODE_GUIDER: "BasicGuider",
        NODE_KSAMPLER_SELECT: "KSamplerSelect",
        NODE_SAMPLER: "SamplerCustomAdvanced",
        NODE_VIDEO_DECODE: "VAEDecode",
        NODE_AUDIO_DECODE: "VAEDecodeAudio",
        NODE_FLASHVSR: FLASHVSR_CLASS,
        NODE_CREATE_VIDEO: "CreateVideo",
    }
    for node_id, class_type in expected_classes.items():
        node = workflow.get(node_id)
        if not isinstance(node, dict) or node.get("class_type") != class_type:
            actual = node.get("class_type") if isinstance(node, dict) else None
            raise ValueError(
                f"H3 production workflow node {node_id} must be {class_type}, got {actual}"
            )

    expected_inputs = {
        (NODE_UNET, "unet_name"): PRUNED_UNET_NAME,
        (NODE_SAGE, "model"): [NODE_UNET, 0],
        (NODE_BLOCKCACHE, "model"): [NODE_SAGE, 0],
        (NODE_BLOCKCACHE, "residual_diff_threshold"): BLOCKCACHE_THRESHOLD,
        (NODE_BLOCKCACHE, "start_percent"): 0.08,
        (NODE_BLOCKCACHE, "end_percent"): 0.95,
        (NODE_BLOCKCACHE, "max_consecutive_hits"): 2,
        (NODE_BLOCKCACHE, "cache_device"): "cpu",
        (NODE_BLOCKCACHE, "metric_stride"): 8,
        (NODE_BLOCKCACHE, "verbose"): False,
        (NODE_SCHEDULER, "model"): [NODE_BLOCKCACHE, 0],
        (NODE_SCHEDULER, "scheduler"): "simple",
        (NODE_SCHEDULER, "steps"): H3_PRODUCTION_STEPS,
        (NODE_GUIDER, "model"): [NODE_BLOCKCACHE, 0],
        (NODE_KSAMPLER_SELECT, "sampler_name"): "res_multistep",
        (NODE_SAMPLER, "guider"): [NODE_GUIDER, 0],
        (NODE_SAMPLER, "sampler"): [NODE_KSAMPLER_SELECT, 0],
        (NODE_SAMPLER, "sigmas"): [NODE_SCHEDULER, 0],
        (NODE_VIDEO_DECODE, "samples"): [NODE_SAMPLER, 0],
        (NODE_AUDIO_DECODE, "samples"): [NODE_SAMPLER, 0],
        (NODE_FLASHVSR, "frames"): [NODE_VIDEO_DECODE, 0],
        (NODE_FLASHVSR, "model"): FLASHVSR_MODEL,
        (NODE_FLASHVSR, "mode"): FLASHVSR_MODE,
        (NODE_FLASHVSR, "scale"): FLASHVSR_SCALE,
        (NODE_FLASHVSR, "tiled_vae"): True,
        (NODE_FLASHVSR, "tiled_dit"): True,
        (NODE_FLASHVSR, "unload_dit"): False,
        (NODE_FLASHVSR, "seed"): 0,
        (NODE_CREATE_VIDEO, "images"): [NODE_FLASHVSR, 0],
        (NODE_CREATE_VIDEO, "audio"): [NODE_AUDIO_DECODE, 0],
    }
    for (node_id, input_name), expected in expected_inputs.items():
        actual = workflow[node_id].get("inputs", {}).get(input_name)
        if actual != expected:
            raise ValueError(
                f"H3 production workflow node {node_id}.{input_name} must be {expected!r}, got {actual!r}"
            )


def _apply_production_workflow(workflow: dict[str, Any]) -> dict[str, Any]:
    """把模板规范化为 5090 上验证过的唯一生产链路。"""
    required_base_nodes = (
        NODE_UNET,
        NODE_SCHEDULER,
        NODE_GUIDER,
        NODE_KSAMPLER_SELECT,
        NODE_SAMPLER,
        NODE_VIDEO_DECODE,
        NODE_AUDIO_DECODE,
        NODE_CREATE_VIDEO,
    )
    missing = [node_id for node_id in required_base_nodes if node_id not in workflow]
    if missing:
        raise ValueError(f"H3 workflow template missing required nodes: {', '.join(missing)}")

    workflow[NODE_UNET].setdefault("inputs", {})["unet_name"] = PRUNED_UNET_NAME
    workflow[NODE_SAGE] = {
        "class_type": SAGE_CLASS,
        "inputs": {"model": [NODE_UNET, 0]},
    }
    workflow[NODE_BLOCKCACHE] = {
        "class_type": BLOCKCACHE_CLASS,
        "inputs": {
            "model": [NODE_SAGE, 0],
            "residual_diff_threshold": BLOCKCACHE_THRESHOLD,
            "start_percent": 0.08,
            "end_percent": 0.95,
            "max_consecutive_hits": 2,
            "cache_device": "cpu",
            "metric_stride": 8,
            "verbose": False,
        },
    }
    workflow[NODE_SCHEDULER]["inputs"]["model"] = [NODE_BLOCKCACHE, 0]
    workflow[NODE_SCHEDULER]["inputs"]["scheduler"] = "simple"
    workflow[NODE_SCHEDULER]["inputs"]["steps"] = H3_PRODUCTION_STEPS
    workflow[NODE_GUIDER]["inputs"]["model"] = [NODE_BLOCKCACHE, 0]
    workflow[NODE_KSAMPLER_SELECT]["inputs"]["sampler_name"] = "res_multistep"
    workflow[NODE_SAMPLER]["inputs"]["guider"] = [NODE_GUIDER, 0]
    workflow[NODE_SAMPLER]["inputs"]["sampler"] = [NODE_KSAMPLER_SELECT, 0]
    workflow[NODE_SAMPLER]["inputs"]["sigmas"] = [NODE_SCHEDULER, 0]
    workflow[NODE_VIDEO_DECODE]["inputs"]["samples"] = [NODE_SAMPLER, 0]
    workflow[NODE_AUDIO_DECODE]["inputs"]["samples"] = [NODE_SAMPLER, 0]
    workflow[NODE_FLASHVSR] = {
        "class_type": FLASHVSR_CLASS,
        "inputs": {
            "frames": [NODE_VIDEO_DECODE, 0],
            "model": FLASHVSR_MODEL,
            "mode": FLASHVSR_MODE,
            "scale": FLASHVSR_SCALE,
            "tiled_vae": True,
            "tiled_dit": True,
            "unload_dit": False,
            "seed": 0,
        },
    }
    workflow[NODE_CREATE_VIDEO]["inputs"]["images"] = [NODE_FLASHVSR, 0]
    workflow[NODE_CREATE_VIDEO]["inputs"]["audio"] = [NODE_AUDIO_DECODE, 0]
    _validate_production_workflow(workflow)
    return workflow


def _apply_common_overrides(
    workflow: dict[str, Any],
    *,
    video_file: str,
    prompt: str | None,
    width: int,
    height: int,
    length: int,
    steps: int,
    seed: int | None,
    filename_prefix: str | None,
) -> dict[str, Any]:
    """覆盖运行时字段，并验证 124 steps 仍是固定生产值。"""
    workflow[NODE_LOAD_VIDEO]["inputs"]["file"] = video_file
    h3_inputs = workflow[NODE_H3]["inputs"]
    if prompt:
        h3_inputs["prompt"] = prompt
    h3_inputs["width"] = int(width)
    h3_inputs["height"] = int(height)
    h3_inputs["length"] = int(length)
    workflow[NODE_SCHEDULER]["inputs"]["steps"] = _require_production_steps(steps)
    workflow[NODE_NOISE]["inputs"]["noise_seed"] = int(seed) if seed is not None else _random_seed()
    if filename_prefix:
        workflow[NODE_SAVE_VIDEO]["inputs"]["filename_prefix"] = filename_prefix
    return _apply_production_workflow(workflow)


def build_white_mask_workflow(
    video_file: str,
    *,
    prompt: str | None = None,
    width: int = 480,
    height: int = 864,
    length: int = 124,
    steps: int = H3_PRODUCTION_STEPS,
    seed: int | None = None,
    filename_prefix: str | None = None,
) -> dict[str, Any]:
    """离线构造白膜生成工作流（video_file 为已上传到 ComfyUI input 目录的文件名）。"""
    workflow = copy.deepcopy(_load_template(WHITEMASK_TEMPLATE))
    return _apply_common_overrides(
        workflow,
        video_file=video_file,
        prompt=prompt,
        width=width,
        height=height,
        length=length,
        steps=steps,
        seed=seed,
        filename_prefix=filename_prefix,
    )


def build_charswap_workflow(
    video_file: str,
    char_image_files: list[str],
    *,
    prompt: str | None = None,
    width: int = 480,
    height: int = 864,
    length: int = 124,
    steps: int = H3_PRODUCTION_STEPS,
    seed: int | None = None,
    filename_prefix: str | None = None,
) -> dict[str, Any]:
    """离线构造白膜换人工作流，按实际人物参考图数量增删 LoadImage 节点。"""
    if not 1 <= len(char_image_files) <= MAX_CHAR_IMAGES:
        raise ValueError(f"char_image_files count must be 1..{MAX_CHAR_IMAGES}, got {len(char_image_files)}")
    workflow = copy.deepcopy(_load_template(CHARSWAP_TEMPLATE))
    _apply_common_overrides(
        workflow,
        video_file=video_file,
        prompt=prompt,
        width=width,
        height=height,
        length=length,
        steps=steps,
        seed=seed,
        filename_prefix=filename_prefix,
    )
    # 移除模板里固化的 LoadImage 节点和 136 的 ref_images.* 引用，按实际数量重建
    for node_id in [
        nid for nid, node in workflow.items() if isinstance(node, dict) and node.get("class_type") == "LoadImage"
    ]:
        del workflow[node_id]
    h3_inputs = workflow[NODE_H3]["inputs"]
    for key in [key for key in h3_inputs if str(key).startswith("ref_images.")]:
        del h3_inputs[key]
    for index, image_file in enumerate(char_image_files):
        node_id = str(LOAD_IMAGE_BASE_ID + index)
        workflow[node_id] = {"class_type": "LoadImage", "inputs": {"image": image_file}}
        h3_inputs[f"ref_images.ref_image_{index}"] = [node_id, 0]
    return workflow


class MiniMaxH3Client:
    """MiniMax H3 工作流执行器，复用通用 ComfyClient 与远程 ComfyUI 通信。"""

    def __init__(self, base_url: str | None = None):
        self.base_url = _resolve_h3_base_url(base_url).rstrip("/")
        self._comfy = ComfyClient(self.base_url)
        self._remote_classes: set[str] | None = None

    @property
    def comfy(self) -> ComfyClient:
        return self._comfy

    def _remote_has_node(self, class_type: str) -> bool:
        """远程 ComfyUI 是否注册了某节点类（/object_info，结果缓存）。"""
        if self._remote_classes is None:
            try:
                import requests

                resp = requests.get(self.base_url + "/object_info", timeout=15)
                self._remote_classes = set(resp.json().keys()) if resp.status_code == 200 else set()
            except Exception:  # noqa: BLE001
                self._remote_classes = set()
        return class_type in self._remote_classes

    def _ensure_workflow_compat(self, workflow: dict[str, Any], log: Callable[[str], None]) -> dict[str, Any]:
        """确认远程具备固定生产链路；缺节点时禁止静默降级。"""
        _validate_production_workflow(workflow)
        required_classes = (SAGE_CLASS, BLOCKCACHE_CLASS, FLASHVSR_CLASS)
        missing = [class_type for class_type in required_classes if not self._remote_has_node(class_type)]
        if missing:
            raise RuntimeError(
                "H3 production workflow unavailable: remote ComfyUI "
                f"{self.base_url} is missing required node classes: {', '.join(missing)}. "
                "The fixed pruned-INT8/Sage/BlockCache/FlashVSR route has no fallback."
            )
        log("> H3 生产节点检查: SageAttention / BlockCache / FlashVSR 均可用")
        return workflow

    def run_white_mask(
        self,
        clip_path: str | Path,
        *,
        prompt: str | None = None,
        width: int = 0,
        height: int = 0,
        length: int | None = None,
        steps: int = H3_PRODUCTION_STEPS,
        out_dir: str | Path,
        seed: int | None = None,
        out_name: str = "whitemask.mp4",
        audio_path: str | Path | None = None,
        background_replace: bool = False,
        background_path: str | Path | None = "",
        log: Callable[[str], None] | None = None,
        on_submitted: Callable[[str, str], None] | None = None,
    ) -> Path:
        """参考视频 → 彩色人偶白膜视频，返回本地产物路径。

        width 或 height 非正数时按源视频宽高比自动推导约 0.4MP 的基础尺寸。
        length 为 None 时按源片时长自动对齐到 H3 的 17k+5 帧网格（发送前完成）。
        audio_path 给定时（如外语配音）替换参考视频音轨，H3 会按该音轨驱动口型。
        background_replace=True 时把 prompt 中的保留版背景句替换为替换版背景句；
        background_path 目前仅透传记录，不接 Comfy 节点（节点能力未确认）。
        """
        logger = log or (lambda _message: None)
        steps = _require_production_steps(steps)
        clip = Path(clip_path)
        if not clip.is_file():
            raise FileNotFoundError(f"h3 clip not found: {clip}")
        width, height = _resolve_h3_run_size(clip, width, height, logger)
        if background_replace:
            from spvideo.white_mask_contract import background_block

            keep_block = background_block(False)
            replace_block = background_block(True)
            if prompt and keep_block in prompt:
                prompt = prompt.replace(keep_block, replace_block)
            elif prompt:
                prompt = prompt + replace_block
            else:
                prompt = replace_block
            logger("> H3 背景: 已启用背景替换，prompt 背景句切换为替换版")
            if background_path:
                logger(f"> H3 背景: background_path 仅记录透传（未接 Comfy 节点）: {background_path}")
        normalized, aligned_length = normalize_reference_video(
            clip, Path(out_dir) / "h3_norm", length=length, audio_path=audio_path, log=logger
        )
        logger(f"> H3 上传参考视频: {normalized.name}")
        video_file = self._comfy.upload_file(normalized)
        workflow = build_white_mask_workflow(
            video_file,
            prompt=prompt,
            width=width,
            height=height,
            length=aligned_length,
            steps=steps,
            seed=seed,
            filename_prefix=f"h3_whitemask_{uuid.uuid4().hex[:8]}",
        )
        logger(
            f"> H3 生产链路: pruned INT8 -> SageAttention -> BlockCache "
            f"threshold={BLOCKCACHE_THRESHOLD} -> {steps} steps -> FlashVSR 2x"
        )
        self._ensure_workflow_compat(workflow, logger)
        prompt_id, history = self._comfy.run_workflow(
            workflow,
            log=logger,
            on_submitted=on_submitted,
        )
        return self._download_result(prompt_id, history, Path(out_dir), out_name)

    def run_charswap(
        self,
        mask_video_path: str | Path,
        char_image_paths: list[str | Path],
        *,
        prompt: str | None = None,
        width: int = 0,
        height: int = 0,
        length: int | None = None,
        steps: int = H3_PRODUCTION_STEPS,
        out_dir: str | Path,
        seed: int | None = None,
        out_name: str = "charswap.mp4",
        audio_path: str | Path | None = None,
        log: Callable[[str], None] | None = None,
        on_submitted: Callable[[str, str], None] | None = None,
    ) -> Path:
        """白膜视频 + 1~3 张人物参考图 → 换人结果视频，返回本地产物路径。

        width 或 height 非正数时按白膜视频宽高比自动推导约 0.4MP 的基础尺寸。
        length 为 None 时按白膜视频时长自动对齐到 H3 的 17k+5 帧网格（发送前完成）。
        audio_path 给定时（如外语配音）替换白膜视频音轨，H3 会按该音轨驱动口型。
        """
        logger = log or (lambda _message: None)
        steps = _require_production_steps(steps)
        mask_video = Path(mask_video_path)
        if not mask_video.is_file():
            raise FileNotFoundError(f"h3 mask video not found: {mask_video}")
        width, height = _resolve_h3_run_size(mask_video, width, height, logger)
        images = [Path(p) for p in char_image_paths]
        if not 1 <= len(images) <= MAX_CHAR_IMAGES:
            raise ValueError(f"char_image_paths count must be 1..{MAX_CHAR_IMAGES}, got {len(images)}")
        for image in images:
            if not image.is_file():
                raise FileNotFoundError(f"h3 char image not found: {image}")
        normalized, aligned_length = normalize_reference_video(
            mask_video, Path(out_dir) / "h3_norm", length=length, audio_path=audio_path, log=logger
        )
        logger(f"> H3 上传白膜视频: {normalized.name}")
        video_file = self._comfy.upload_file(normalized)
        image_files = []
        for image in images:
            logger(f"> H3 上传人物参考图: {image.name}")
            image_files.append(self._comfy.upload_file(image))
        workflow = build_charswap_workflow(
            video_file,
            image_files,
            prompt=prompt,
            width=width,
            height=height,
            length=aligned_length,
            steps=steps,
            seed=seed,
            filename_prefix=f"h3_charswap_{uuid.uuid4().hex[:8]}",
        )
        logger(
            f"> H3 生产链路: pruned INT8 -> SageAttention -> BlockCache "
            f"threshold={BLOCKCACHE_THRESHOLD} -> {steps} steps -> FlashVSR 2x"
        )
        self._ensure_workflow_compat(workflow, logger)
        prompt_id, history = self._comfy.run_workflow(
            workflow,
            log=logger,
            on_submitted=on_submitted,
        )
        return self._download_result(prompt_id, history, Path(out_dir), out_name)

    def resume_result(
        self,
        prompt_id: str,
        *,
        out_dir: str | Path,
        out_name: str,
        log: Callable[[str], None] | None = None,
    ) -> Path:
        """继续等待已提交的远程任务，并下载固定节点 92 的视频产物。"""
        history = self._comfy.resume_workflow(prompt_id, log=log)
        return self._download_result(prompt_id, history, Path(out_dir), out_name)

    def _download_result(
        self,
        prompt_id: str,
        history: dict[str, Any],
        out_dir: Path,
        out_name: str,
    ) -> Path:
        item = history.get(prompt_id) if isinstance(history, dict) else None
        outputs = item.get("outputs") if isinstance(item, dict) else None
        node_output = outputs.get(NODE_SAVE_VIDEO) if isinstance(outputs, dict) else None
        asset = ComfyClient.first_output_asset(node_output)
        if asset is None:
            raise RuntimeError(
                "MiniMax H3 workflow produced no SaveVideo output: "
                + ComfyClient.history_debug_summary(item)
            )
        target = out_dir / out_name
        self._comfy.download_output_asset(asset, target)
        return target
