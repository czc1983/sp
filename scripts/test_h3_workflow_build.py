"""离线验证 MiniMax H3 两套生产工作流，不提交 ComfyUI。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spvideo.minimax_h3_client import (  # noqa: E402
    MiniMaxH3Client,
    _resolve_h3_run_size,
    build_charswap_workflow,
    build_white_mask_workflow,
    derive_h3_base_size,
)

PRUNED_UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"


def _assert_production_chain(workflow: dict) -> None:
    assert workflow["127"] == {
        "class_type": "UNETLoader",
        "inputs": {"unet_name": PRUNED_UNET, "weight_dtype": "default"},
    }
    assert workflow["204"] == {
        "class_type": "MiniMaxH3MemoryEfficientSageAttentionPatch",
        "inputs": {"model": ["127", 0]},
    }
    assert workflow["205"] == {
        "class_type": "MiniMaxH3BlockCacheT8",
        "inputs": {
            "model": ["204", 0],
            "residual_diff_threshold": 0.12,
            "start_percent": 0.08,
            "end_percent": 0.95,
            "max_consecutive_hits": 2,
            "cache_device": "cpu",
            "metric_stride": 8,
            "verbose": False,
        },
    }

    assert workflow["123"] == {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": "res_multistep"},
    }
    scheduler = workflow["124"]
    assert scheduler["class_type"] == "BasicScheduler"
    assert scheduler["inputs"]["model"] == ["205", 0]
    assert scheduler["inputs"]["scheduler"] == "simple"
    assert scheduler["inputs"]["steps"] == 20
    assert scheduler["inputs"]["denoise"] == 1.0
    assert workflow["126"]["class_type"] == "BasicGuider"
    assert workflow["126"]["inputs"]["model"] == ["205", 0]

    sampler = workflow["125"]
    assert sampler["class_type"] == "SamplerCustomAdvanced"
    assert sampler["inputs"]["guider"] == ["126", 0]
    assert sampler["inputs"]["sampler"] == ["123", 0]
    assert sampler["inputs"]["sigmas"] == ["124", 0]
    assert sampler["inputs"]["latent_image"] == ["136", 1]

    assert workflow["122"]["class_type"] == "VAEDecode"
    assert workflow["122"]["inputs"]["samples"] == ["125", 0]
    assert workflow["121"]["class_type"] == "VAEDecodeAudio"
    assert workflow["121"]["inputs"]["samples"] == ["125", 0]
    assert workflow["206"] == {
        "class_type": "FlashVSRNode",
        "inputs": {
            "frames": ["122", 0],
            "model": "FlashVSR-v1.1",
            "mode": "tiny",
            "scale": 2,
            "tiled_vae": True,
            "tiled_dit": True,
            "unload_dit": False,
            "seed": 0,
        },
    }
    assert workflow["130"]["class_type"] == "CreateVideo"
    assert workflow["130"]["inputs"]["images"] == ["206", 0]
    assert workflow["130"]["inputs"]["audio"] == ["121", 0]

    class_types = {str(node.get("class_type") or "") for node in workflow.values() if isinstance(node, dict)}
    forbidden_fragments = ("Lora", "DualClock", "HyperStep", "SolAttn")
    assert not any(fragment in class_type for class_type in class_types for fragment in forbidden_fragments)
    assert "203" not in workflow


def _assert_size_derivation() -> None:
    assert derive_h3_base_size(1080, 1920) == (480, 864)
    assert derive_h3_base_size(1920, 1080) == (864, 480)
    assert derive_h3_base_size(1000, 1000) == (640, 640)

    source_width, source_height = 1373, 777
    width, height = derive_h3_base_size(source_width, source_height)
    assert width % 32 == 0 and height % 32 == 0
    assert abs(width * height - 400_000) / 400_000 <= 0.05
    source_ratio = source_width / source_height
    assert abs(width / height - source_ratio) / source_ratio <= 0.03

    for bad_width, bad_height in ((0, 1080), (1920, 0), (-1, 1080)):
        try:
            derive_h3_base_size(bad_width, bad_height)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad_width}x{bad_height}")
    print("[ok] size: 常见横竖屏、方形、任意比例和非法输入")


def _assert_runtime_size_selection() -> None:
    from spvideo import ffmpeg_tools

    original_probe = ffmpeg_tools.probe_video
    ffmpeg_tools.probe_video = lambda _path: SimpleNamespace(width=1920, height=1080)
    logs: list[str] = []
    try:
        assert _resolve_h3_run_size(Path("source.mp4"), 0, 720, logs.append) == (864, 480)
        assert _resolve_h3_run_size(Path("source.mp4"), 512, 896, logs.append) == (512, 896)
    finally:
        ffmpeg_tools.probe_video = original_probe
    assert logs == [
        "> H3 尺寸: source 1920x1080 -> base 864x480 -> FlashVSR 2x 1728x960",
        "> H3 尺寸: source 1920x1080 -> base 512x896 -> FlashVSR 2x 1024x1792",
    ]
    print("[ok] runtime size: 任一非正维度触发自动推导，显式正尺寸保持不变")


def _assert_remote_dependencies() -> None:
    workflow = build_white_mask_workflow("a.mp4")
    client = MiniMaxH3Client("http://example.invalid")
    required = {
        "MiniMaxH3MemoryEfficientSageAttentionPatch",
        "MiniMaxH3BlockCacheT8",
        "FlashVSRNode",
    }
    client._remote_classes = set(required)
    assert client._ensure_workflow_compat(workflow, lambda _message: None) is workflow

    client._remote_classes.remove("FlashVSRNode")
    try:
        client._ensure_workflow_compat(workflow, lambda _message: None)
    except RuntimeError as exc:
        assert "FlashVSRNode" in str(exc) and "no fallback" in str(exc)
    else:
        raise AssertionError("expected missing FlashVSRNode to fail without fallback")
    print("[ok] remote nodes: 三个生产插件缺一即明确报错")


def main() -> None:
    os.environ.pop("H3_BLOCKCACHE_THRESHOLD", None)
    os.environ["H3_BLOCKCACHE"] = "off"

    for template_name in ("h3_whitemask_api.json", "h3_charswap_api.json"):
        template = json.loads((ROOT / "comfy_workflows" / template_name).read_text(encoding="utf-8"))
        _assert_production_chain(template)
    print("[ok] templates: 两套 API JSON 均为固定生产链路")

    white = build_white_mask_workflow(
        "my_clip.mp4",
        prompt="自定义提示词",
        width=512,
        height=896,
        length=100,
        seed=12345,
    )
    _assert_production_chain(white)
    assert white["140"]["inputs"]["file"] == "my_clip.mp4"
    h3 = white["136"]["inputs"]
    assert h3["prompt"] == "自定义提示词"
    assert (h3["width"], h3["height"], h3["length"]) == (512, 896, 100)
    assert h3["ref_videos.ref_video_0"] == ["141", 0]
    assert not any(key.startswith("ref_images.") for key in h3)
    assert white["129"]["inputs"]["noise_seed"] == 12345
    assert white["92"]["inputs"]["filename_prefix"].startswith("h3_whitemask")

    white_default = build_white_mask_workflow("a.mp4")
    _assert_production_chain(white_default)
    assert "彩色树脂关节人偶" in white_default["136"]["inputs"]["prompt"]
    assert isinstance(white_default["129"]["inputs"]["noise_seed"], int)
    os.environ["H3_BLOCKCACHE_THRESHOLD"] = "0.2"
    assert build_white_mask_workflow("threshold.mp4")["205"]["inputs"]["residual_diff_threshold"] == 0.12
    os.environ.pop("H3_BLOCKCACHE_THRESHOLD", None)
    try:
        build_white_mask_workflow("invalid_steps.mp4", steps=30)
    except ValueError as exc:
        assert "must be 20" in str(exc) and "30" in str(exc)
    else:
        raise AssertionError("expected white-mask steps=30 to be rejected")
    print("[ok] white-mask: 固定 20 步、BlockCache 0.12 与运行字段覆盖")

    charswap = build_charswap_workflow(
        "mask.mp4",
        ["char1.png", "char2.png"],
        width=512,
        height=896,
        length=97,
        seed=999,
    )
    _assert_production_chain(charswap)
    assert charswap["140"]["inputs"]["file"] == "mask.mp4"
    h3_swap = charswap["136"]["inputs"]
    assert (h3_swap["width"], h3_swap["height"], h3_swap["length"]) == (512, 896, 97)
    assert charswap["129"]["inputs"]["noise_seed"] == 999
    assert charswap["92"]["inputs"]["filename_prefix"].startswith("h3_wmswap")

    for count in range(1, 4):
        images = [f"char_{index}.png" for index in range(count)]
        workflow = build_charswap_workflow("mask.mp4", images)
        h3_inputs = workflow["136"]["inputs"]
        load_image_ids = {
            node_id for node_id, node in workflow.items() if node.get("class_type") == "LoadImage"
        }
        assert load_image_ids == {str(150 + index) for index in range(count)}
        for index, image in enumerate(images):
            node_id = str(150 + index)
            assert workflow[node_id]["inputs"]["image"] == image
            assert h3_inputs[f"ref_images.ref_image_{index}"] == [node_id, 0]
    try:
        build_charswap_workflow("mask.mp4", ["char.png"], steps=25)
    except ValueError as exc:
        assert "must be 20" in str(exc) and "25" in str(exc)
    else:
        raise AssertionError("expected charswap steps=25 to be rejected")
    print("[ok] charswap: 1-3 张人物图重建、固定 20 步与运行字段覆盖")

    for invalid_images in ([], ["a", "b", "c", "d"]):
        try:
            build_charswap_workflow("mask.mp4", invalid_images)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {len(invalid_images)} images")

    _assert_size_derivation()
    _assert_runtime_size_selection()
    _assert_remote_dependencies()
    os.environ.pop("H3_BLOCKCACHE", None)
    print("\n全部断言通过。")


if __name__ == "__main__":
    main()
