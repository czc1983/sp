from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


INVENTORY_ROOT = Path(__file__).resolve().parent.parent / ".server_inventory"
NON_MODEL_RESOURCE_FOLDERS = {"custom_nodes", "VHS_video_formats", "kjnodes_fonts"}

SCAIL2_REQUIRED_NODES = {
    "BasicScheduler",
    "CLIPLoader",
    "CLIPTextEncode",
    "CLIPVisionLoader",
    "CheckpointLoaderSimple",
    "DiffusionModelLoaderKJ",
    "KSamplerSelect",
    "LoadImage",
    "LoraLoaderModelOnly",
    "GrowMaskWithBlur",
    "ImageCompositeMasked",
    "ImageBatchMulti",
    "ImageToMask",
    "MaskToImage",
    "MaskComposite",
    "ModelSamplingSD3",
    "SAM3_TrackToMask",
    "SAM3_VideoTrack",
    "SaveImage",
    "SCAIL2FitVideo",
    "SCAIL2ColoredMask",
    "SCAIL2ReferencePack",
    "SCAIL2ReferenceSAMBuilder",
    "SCAIL2SimpleVideo",
    "VAELoader",
    "VHS_LoadVideo",
    "VHS_SelectMasks",
    "VHS_VideoCombine",
    "WanChunkFeedForward",
}
SCAIL2_OPTIONAL_NODES = {"ImageResizeKJv2"}
SCAIL2_REQUIRED_MODELS = {
    "diffusion_models": {"wan2.1_14B_SCAIL_2_fp8_scaled.safetensors"},
    "loras": {
        "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
        "SCAIL-2/wan2.1_SCAIL_2_DPO_lora_bf16.safetensors",
    },
    "text_encoders": {"umt5_xxl_fp8_e4m3fn_scaled.safetensors"},
    "vae": {"Wan2_1_VAE_bf16.safetensors"},
    "clip_vision": {"clip_vision_vit_h.safetensors"},
    "checkpoints": {"sam3.1_multiplex_fp16.safetensors"},
}


def inventory_path(comfy_url: str) -> Path:
    parsed = urlparse(comfy_url)
    host = parsed.netloc or parsed.path
    safe_host = re.sub(r"[^A-Za-z0-9._-]+", "_", host).strip("_") or "comfyui"
    return INVENTORY_ROOT / f"{safe_host}.json"


def load_inventory(comfy_url: str) -> dict[str, Any] | None:
    path = inventory_path(comfy_url)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def fetch_inventory(
    comfy_url: str,
    *,
    timeout: float = 30.0,
    save: bool = True,
) -> dict[str, Any]:
    base = comfy_url.rstrip("/")
    session = requests.Session()
    system = _get_json(session, f"{base}/system_stats", timeout)
    object_info = _get_json(session, f"{base}/object_info", max(timeout, 60.0))
    model_folders = _get_json(session, f"{base}/models", timeout)
    extensions = _get_json(session, f"{base}/extensions", timeout)

    if not isinstance(object_info, dict):
        raise ValueError("invalid_comfy_object_info")
    if not isinstance(model_folders, list):
        model_folders = []
    if not isinstance(extensions, list):
        extensions = []

    models: dict[str, list[str]] = {}
    model_errors: dict[str, str] = {}

    def fetch_model_folder(folder: str) -> tuple[str, list[str]]:
        values = _get_json(session, f"{base}/models/{folder}", timeout)
        if not isinstance(values, list):
            raise ValueError("invalid_model_list")
        return folder, sorted(str(value) for value in values)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(fetch_model_folder, str(folder)): str(folder)
            for folder in model_folders
        }
        for future in as_completed(futures):
            folder = futures[future]
            try:
                key, values = future.result()
                models[key] = values
            except Exception as exc:  # noqa: BLE001
                models[folder] = []
                model_errors[folder] = str(exc)

    nodes: dict[str, dict[str, Any]] = {}
    plugins: dict[str, dict[str, Any]] = {}
    for class_name, definition in object_info.items():
        if not isinstance(definition, dict):
            continue
        module = str(definition.get("python_module") or "")
        plugin = _plugin_from_module(module)
        node = {
            "class_type": class_name,
            "display_name": definition.get("display_name") or class_name,
            "category": definition.get("category") or "",
            "python_module": module,
            "plugin": plugin,
            "output_node": bool(definition.get("output_node", False)),
            "inputs": list((definition.get("input") or {}).get("required", {}).keys()),
            "optional_inputs": list((definition.get("input") or {}).get("optional", {}).keys()),
            "outputs": list(definition.get("output_name") or definition.get("output") or []),
        }
        nodes[class_name] = node
        item = plugins.setdefault(
            plugin,
            {
                "name": plugin,
                "python_modules": set(),
                "nodes": [],
                "frontend_extensions": [],
                "api_callable": False,
            },
        )
        if module:
            item["python_modules"].add(module)
        item["nodes"].append(class_name)
        item["api_callable"] = True

    for extension in extensions:
        extension = str(extension)
        plugin = _plugin_from_extension(extension)
        item = plugins.setdefault(
            plugin,
            {
                "name": plugin,
                "python_modules": set(),
                "nodes": [],
                "frontend_extensions": [],
                "api_callable": False,
            },
        )
        item["frontend_extensions"].append(extension)

    normalized_plugins: dict[str, dict[str, Any]] = {}
    for name, item in sorted(plugins.items()):
        normalized_plugins[name] = {
            **item,
            "python_modules": sorted(item["python_modules"]),
            "nodes": sorted(item["nodes"]),
            "frontend_extensions": sorted(item["frontend_extensions"]),
            "node_count": len(item["nodes"]),
        }

    total_resource_files = sum(len(values) for values in models.values())
    total_model_files = sum(
        len(values)
        for folder, values in models.items()
        if folder not in NON_MODEL_RESOURCE_FOLDERS
    )
    inventory = {
        "schema_version": 1,
        "comfy_url": base,
        "fetched_at": time.time(),
        "fetched_at_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "system": system,
        "summary": {
            "node_count": len(nodes),
            "plugin_count": len(normalized_plugins),
            "model_folder_count": len(models),
            "model_file_count": total_model_files,
            "resource_file_count": total_resource_files,
            "frontend_extension_count": len(extensions),
        },
        "profile_checks": {
            "scail2": evaluate_scail2_requirements(nodes.keys(), models),
        },
        "nodes": dict(sorted(nodes.items())),
        "plugins": normalized_plugins,
        "models": dict(sorted(models.items())),
        "model_errors": model_errors,
        "extensions": sorted(str(value) for value in extensions),
        "api_notes": {
            "workflow_nodes": "Nodes listed in object_info can be invoked through POST /prompt.",
            "frontend_only": "Extensions without nodes are UI extensions and are not standalone workflow APIs.",
            "manager": "Core inventory APIs are available; legacy ComfyUI-Manager list endpoints returned 404.",
        },
    }
    if save:
        path = inventory_path(base)
        path.parent.mkdir(parents=True, exist_ok=True)
        inventory["inventory_path"] = str(path)
        path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    return inventory


def evaluate_scail2_requirements(
    node_names: Any,
    models: dict[str, list[str]],
) -> dict[str, Any]:
    available_nodes = set(str(value) for value in node_names)
    missing_nodes = sorted(SCAIL2_REQUIRED_NODES - available_nodes)
    missing_optional_nodes = sorted(SCAIL2_OPTIONAL_NODES - available_nodes)
    missing_models: list[str] = []
    for folder, required in SCAIL2_REQUIRED_MODELS.items():
        available = set(models.get(folder, []))
        for model in sorted(required - available):
            missing_models.append(f"{folder}/{model}")
    return {
        "ready": not missing_nodes and not missing_models,
        "missing_nodes": missing_nodes,
        "missing_optional_nodes": missing_optional_nodes,
        "missing_models": missing_models,
    }


def check_scail2_server(
    comfy_url: str,
    *,
    timeout: float = 20.0,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    base = comfy_url.rstrip("/")
    http = session or requests.Session()
    missing_nodes: list[str] = []
    missing_optional_nodes: list[str] = []
    try:
        for class_name in sorted(SCAIL2_REQUIRED_NODES | SCAIL2_OPTIONAL_NODES):
            response = http.get(f"{base}/object_info/{class_name}", timeout=timeout)
            if response.status_code == 404:
                if class_name in SCAIL2_OPTIONAL_NODES:
                    missing_optional_nodes.append(class_name)
                else:
                    missing_nodes.append(class_name)
                continue
            response.raise_for_status()

        missing_models: list[str] = []
        for folder, required in SCAIL2_REQUIRED_MODELS.items():
            response = http.get(f"{base}/models/{folder}", timeout=timeout)
            response.raise_for_status()
            available = set(str(value) for value in response.json())
            for model in sorted(required - available):
                missing_models.append(f"{folder}/{model}")
        return {
            "ready": not missing_nodes and not missing_models,
            "missing_nodes": missing_nodes,
            "missing_optional_nodes": missing_optional_nodes,
            "missing_models": missing_models,
            "source": "live",
        }
    except requests.RequestException as exc:
        cached = load_inventory(base)
        if not cached:
            raise
        result = dict(cached.get("profile_checks", {}).get("scail2") or {})
        if not result:
            result = evaluate_scail2_requirements(
                (cached.get("nodes") or {}).keys(),
                cached.get("models") or {},
            )
        if missing_nodes:
            result["missing_nodes"] = sorted(set(result.get("missing_nodes", [])) | set(missing_nodes))
            result["ready"] = False
        if missing_optional_nodes:
            result["missing_optional_nodes"] = sorted(
                set(result.get("missing_optional_nodes", [])) | set(missing_optional_nodes)
            )
        result["source"] = "cache"
        result["warning"] = f"live_check_unavailable: {exc}"
        return result


def _get_json(session: requests.Session, url: str, timeout: float) -> Any:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _plugin_from_module(module: str) -> str:
    if module.startswith("custom_nodes."):
        parts = module.split(".")
        return parts[1] if len(parts) > 1 else "custom_nodes"
    if module.startswith("comfy_extras.") or module in {"nodes", "comfy_api_nodes"}:
        return "ComfyUI Core"
    return module.split(".", 1)[0] if module else "Unknown"


def _plugin_from_extension(extension: str) -> str:
    parts = extension.strip("/").split("/")
    if len(parts) >= 2 and parts[0] == "extensions":
        return "ComfyUI Core" if parts[1] == "core" else parts[1]
    return "Unknown frontend"
