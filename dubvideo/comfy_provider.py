from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import requests


LIPSYNC_PROFILES: dict[str, dict[str, set[str]]] = {
    "kling_audio": {
        "nodes": {"LoadVideo", "LoadAudio", "KlingLipSyncAudioToVideoNode", "SaveVideo"},
        "models": set(),
    },
    "kling_text": {
        "nodes": {"LoadVideo", "KlingLipSyncTextToVideoNode", "SaveVideo"},
        "models": set(),
    },
    "wan_multitalk": {
        "nodes": {"MultiTalkModelLoader", "MultiTalkWav2VecEmbeds", "WanVideoImageToVideoMultiTalk"},
        "models": {"audio_encoders/wav2vec2_large_english_fp16.safetensors"},
    },
    "infinitetalk": {
        "nodes": {
            "MultiTalkModelLoader",
            "MultiTalkWav2VecEmbeds",
            "Wav2VecModelLoader",
            "WanVideoModelLoader",
            "WanVideoImageToVideoMultiTalk",
            "WanVideoSampler",
            "WanVideoTextEncodeCached",
            "WanVideoClipVisionEncode",
            "WanVideoEncode",
            "WanVideoDecode",
            "WanVideoVAELoader",
            "WanVideoLoraSelect",
            "WanVideoBlockSwap",
            "ImageResizeKJv2",
            "GetImageRangeFromBatch",
            "VHS_LoadVideo",
            "VHS_VideoCombine",
            "CLIPVisionLoader",
            "LoadAudio",
        },
        "models": {
            "diffusion_models/WanVideo/InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors",
            "diffusion_models/kijai/WanVideo_comfy/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors",
            "loras/WanVideo/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
            "vae/Wan2_1_VAE_bf16.safetensors",
        },
    },
    "latent_sync": {"nodes": {"LatentSync"}, "models": set()},
    "musetalk": {"nodes": {"MuseTalk"}, "models": set()},
    "wav2lip": {"nodes": {"Wav2Lip"}, "models": set()},
}


INFINITETALK_MODEL = "kijai/WanVideo_comfy/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors"
INFINITETALK_PATCH = "WanVideo/InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors"
INFINITETALK_LORA = "WanVideo/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
INFINITETALK_VAE = "Wan2_1_VAE_bf16.safetensors"
INFINITETALK_T5 = "umt5-xxl-enc-bf16.safetensors"
INFINITETALK_CLIP_VISION = "clip_vision_h.safetensors"
INFINITETALK_WAV2VEC = "wav2vec2-chinese-base_fp16.safetensors"
INFINITETALK_NEGATIVE_PROMPT = (
    "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
    "fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
)


class ComfyLipSyncProvider:
    def __init__(self, base_url: str, timeout: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def test_connection(self) -> dict[str, Any]:
        try:
            response = self.session.get(f"{self.base_url}/system_stats", timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            system = data.get("system") or {}
            devices = data.get("devices") or []
            device = devices[0] if devices else {}
            return {
                "ok": True,
                "base_url": self.base_url,
                "comfyui_version": system.get("comfyui_version"),
                "python_version": system.get("python_version"),
                "pytorch_version": system.get("pytorch_version"),
                "ram_total": system.get("ram_total"),
                "gpu": device.get("name"),
                "vram_total": device.get("vram_total"),
                "vram_free": device.get("vram_free"),
            }
        except Exception as exc:
            return {"ok": False, "base_url": self.base_url, "error": str(exc)}

    def fetch_inventory(self, cache_root: str | Path, refresh: bool = False) -> dict[str, Any]:
        cache_path = _inventory_path(cache_root, self.base_url)
        if cache_path.exists() and not refresh:
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8-sig"))
                cached["source"] = "cache"
                return cached
            except Exception:
                pass
        session = requests.Session()
        system = _get_json(session, f"{self.base_url}/system_stats", self.timeout)
        object_info = _get_json(session, f"{self.base_url}/object_info", self.timeout)
        folders = _get_json(session, f"{self.base_url}/models", self.timeout)
        models: dict[str, list[str]] = {}
        if isinstance(folders, list):
            for folder in folders:
                try:
                    values = _get_json(session, f"{self.base_url}/models/{folder}", self.timeout)
                    models[str(folder)] = [str(value) for value in values] if isinstance(values, list) else []
                except Exception:
                    models[str(folder)] = []
        inventory = {
            "ok": True,
            "source": "live",
            "base_url": self.base_url,
            "fetched_at": time.time(),
            "summary": {
                "node_count": len(object_info) if isinstance(object_info, dict) else 0,
                "model_folder_count": len(models),
                "model_file_count": sum(len(values) for values in models.values()),
            },
            "nodes": object_info if isinstance(object_info, dict) else {},
            "models": models,
            "profiles": {
                name: self.check_lipsync_profile({"nodes": object_info, "models": models}, name)
                for name in LIPSYNC_PROFILES
            },
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
        return inventory

    def check_lipsync_profile(self, inventory: dict[str, Any], profile: str) -> dict[str, Any]:
        spec = LIPSYNC_PROFILES.get(profile)
        if not spec:
            return {"ready": False, "missing_nodes": [], "missing_models": [], "warnings": ["unknown_profile"]}
        nodes = set((inventory.get("nodes") or {}).keys())
        models = inventory.get("models") or {}
        available_models = {f"{folder}/{value}" for folder, values in models.items() for value in values}
        missing_nodes = sorted(spec["nodes"] - nodes)
        missing_models = sorted(spec["models"] - available_models)
        warnings: list[str] = []
        if profile.startswith("kling"):
            warnings.append("Kling 是 partner API 节点，仍需服务器侧凭证可用。")
        if profile in {"latent_sync", "musetalk", "wav2lip"}:
            warnings.append("当前仅检查节点名；具体模型文件需按安装方案补充。")
        return {
            "profile": profile,
            "ready": not missing_nodes and not missing_models,
            "missing_nodes": missing_nodes,
            "missing_models": missing_models,
            "warnings": warnings,
            "candidate_nodes": sorted(spec["nodes"]),
        }

    def upload_file(self, path: str | Path, subfolder: str = "foreign_lipsync") -> str:
        local_path = Path(path)
        if not local_path.is_file():
            raise FileNotFoundError(f"comfy_upload_file_not_found: {local_path}")
        filename = _content_addressed_name(local_path)
        with local_path.open("rb") as handle:
            response = self.session.post(
                f"{self.base_url}/upload/image",
                files={"image": (filename, handle)},
                data={"subfolder": subfolder, "overwrite": "true"},
                timeout=180,
            )
        response.raise_for_status()
        payload = response.json()
        name = str(payload.get("name") or filename)
        returned_subfolder = str(payload.get("subfolder") or subfolder or "").strip("/")
        return f"{returned_subfolder}/{name}" if returned_subfolder else name

    def build_kling_audio_workflow(
        self,
        *,
        video_name: str,
        audio_name: str,
        voice_language: str = "en",
        filename_prefix: str = "foreign_lipsync/lipsync",
    ) -> dict[str, Any]:
        language = "zh" if str(voice_language).lower().startswith("zh") else "en"
        return {
            "1": {
                "class_type": "LoadVideo",
                "inputs": {"file": video_name},
            },
            "2": {
                "class_type": "LoadAudio",
                "inputs": {"audio": audio_name},
            },
            "3": {
                "class_type": "KlingLipSyncAudioToVideoNode",
                "inputs": {
                    "video": ["1", 0],
                    "audio": ["2", 0],
                    "voice_language": language,
                },
            },
            "4": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["3", 0],
                    "filename_prefix": filename_prefix,
                    "format": "mp4",
                    "codec": "h264",
                },
            },
        }

    def run_kling_audio_lipsync(
        self,
        *,
        video_path: str | Path,
        audio_path: str | Path,
        output_dir: str | Path,
        voice_language: str = "en",
        timeout_seconds: int = 5400,
        progress: Callable[[str, int], None] | None = None,
    ) -> dict[str, Any]:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        project_slug = _safe_slug(Path(video_path).stem, 24)
        subfolder = f"foreign_lipsync/{project_slug}_{uuid.uuid4().hex[:8]}"
        if progress:
            progress("上传原视频到 Comfy", 8)
        video_name = self.upload_file(video_path, subfolder=subfolder)
        if progress:
            progress("上传整轨配音到 Comfy", 16)
        audio_name = self.upload_file(audio_path, subfolder=subfolder)
        workflow = self.build_kling_audio_workflow(
            video_name=video_name,
            audio_name=audio_name,
            voice_language=voice_language,
            filename_prefix=f"foreign_lipsync/{project_slug}_kling",
        )
        if progress:
            progress("提交 Kling 口型工作流", 20)
        prompt_id, client_id = self.submit_workflow(workflow)
        history = self.wait_for_completion(prompt_id, timeout_seconds=timeout_seconds, progress=progress)
        self.raise_for_history_error(prompt_id, history)
        asset = first_output_asset(history.get(prompt_id, {}).get("outputs"))
        if not asset:
            raise RuntimeError("comfy_lipsync_output_not_found")
        target = output_root / ("lipsync_kling" + (Path(str(asset.get("filename") or "")).suffix or ".mp4"))
        view_url = self.download_output_asset(asset, target)
        return {
            "ok": True,
            "prompt_id": prompt_id,
            "client_id": client_id,
            "output_video_path": str(target),
            "remote_asset": asset,
            "remote_view_url": view_url,
            "workflow": workflow,
        }

    def build_infinitetalk_workflow(
        self,
        *,
        video_name: str,
        audio_name: str,
        width: int,
        height: int,
        fps: float,
        num_frames: int,
        positive_prompt: str,
        filename_prefix: str = "foreign_lipsync/infinitetalk",
    ) -> dict[str, Any]:
        return {
            "1": {"class_type": "MultiTalkModelLoader", "inputs": {"model": INFINITETALK_PATCH}},
            "2": {"class_type": "WanVideoBlockSwap", "inputs": {
                "blocks_to_swap": 20, "offload_img_emb": False, "offload_txt_emb": False,
                "use_non_blocking": True, "vace_blocks_to_swap": 0, "prefetch_blocks": 1,
                "block_swap_debug": False,
            }},
            "3": {"class_type": "WanVideoLoraSelect", "inputs": {
                "lora": INFINITETALK_LORA, "strength": 1.0, "low_mem_load": False, "merge_loras": False,
            }},
            "4": {"class_type": "WanVideoModelLoader", "inputs": {
                "model": INFINITETALK_MODEL, "base_precision": "fp16_fast", "quantization": "disabled",
                "load_device": "offload_device", "attention_mode": "sageattn",
                "block_swap_args": ["2", 0], "lora": ["3", 0], "multitalk_model": ["1", 0],
            }},
            "5": {"class_type": "WanVideoVAELoader", "inputs": {"model_name": INFINITETALK_VAE, "precision": "bf16"}},
            "6": {"class_type": "WanVideoTextEncodeCached", "inputs": {
                "model_name": INFINITETALK_T5, "precision": "bf16",
                "positive_prompt": positive_prompt or "a person is talking, natural lip movement, clear face",
                "negative_prompt": INFINITETALK_NEGATIVE_PROMPT,
                "quantization": "disabled", "use_disk_cache": False, "device": "gpu",
            }},
            "7": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": INFINITETALK_CLIP_VISION}},
            "8": {"class_type": "VHS_LoadVideo", "inputs": {
                "video": video_name, "force_rate": 0, "custom_width": 0, "custom_height": 0,
                "frame_load_cap": 0, "skip_first_frames": 0, "select_every_nth": 1, "format": "Wan",
            }},
            "9": {"class_type": "ImageResizeKJv2", "inputs": {
                "image": ["8", 0], "width": width, "height": height, "upscale_method": "lanczos",
                "keep_proportion": "crop", "pad_color": "0, 0, 0", "crop_position": "center",
                "divisible_by": 16, "device": "cpu",
            }},
            "10": {"class_type": "WanVideoEncode", "inputs": {
                "vae": ["5", 0], "image": ["9", 0], "enable_vae_tiling": True,
                "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128,
                "noise_aug_strength": 0, "latent_strength": 1,
            }},
            "11": {"class_type": "GetImageRangeFromBatch", "inputs": {
                "images": ["9", 0], "start_index": 0, "num_frames": 1,
            }},
            "12": {"class_type": "WanVideoClipVisionEncode", "inputs": {
                "clip_vision": ["7", 0], "image_1": ["11", 0], "strength_1": 1.0, "strength_2": 1.0,
                "crop": "center", "combine_embeds": "average", "force_offload": True,
                "tiles": 0, "ratio": 0.5,
            }},
            "13": {"class_type": "WanVideoImageToVideoMultiTalk", "inputs": {
                "vae": ["5", 0], "start_image": ["11", 0], "clip_embeds": ["12", 0],
                "width": width, "height": height, "frame_window_size": 81, "motion_frame": 9,
                "force_offload": False, "colormatch": "disabled", "tiled_vae": True, "mode": "infinitetalk",
            }},
            "14": {"class_type": "Wav2VecModelLoader", "inputs": {
                "model": INFINITETALK_WAV2VEC, "base_precision": "fp16", "load_device": "main_device",
            }},
            "15": {"class_type": "LoadAudio", "inputs": {"audio": audio_name}},
            "16": {"class_type": "MultiTalkWav2VecEmbeds", "inputs": {
                "wav2vec_model": ["14", 0], "audio_1": ["15", 0], "normalize_loudness": True,
                "num_frames": num_frames, "fps": float(fps), "audio_scale": 1.5,
                "audio_cfg_scale": 1.0, "multi_audio_type": "para",
            }},
            "17": {"class_type": "WanVideoSampler", "inputs": {
                "model": ["4", 0], "image_embeds": ["13", 0], "text_embeds": ["6", 0],
                "samples": ["10", 0], "multitalk_embeds": ["16", 0],
                "steps": 4, "cfg": 1.0, "shift": 11.0, "seed": 2, "force_offload": True,
                "scheduler": "dpm++_sde", "riflex_freq_index": 0, "denoise_strength": 1.0,
                "batched_cfg": False, "rope_function": "comfy",
                "start_step": 2, "end_step": -1, "add_noise_to_samples": True,
            }},
            "18": {"class_type": "WanVideoDecode", "inputs": {
                "vae": ["5", 0], "samples": ["17", 0], "enable_vae_tiling": True,
                "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128,
                "normalization": "default",
            }},
            "19": {"class_type": "VHS_VideoCombine", "inputs": {
                "images": ["18", 0], "frame_rate": float(fps), "loop_count": 0,
                "filename_prefix": filename_prefix, "format": "video/h264-mp4",
                "pingpong": False, "save_output": True,
            }},
        }

    def run_infinitetalk_lipsync(
        self,
        *,
        video_path: str | Path,
        audio_path: str | Path,
        output_dir: str | Path,
        ffmpeg_configured: str = "",
        positive_prompt: str = "",
        chunk_seconds: float = 24.0,
        timeout_seconds: int = 5400,
        progress: Callable[[str, int], None] | None = None,
    ) -> dict[str, Any]:
        import shutil
        import subprocess

        from .ffmpeg_tools import find_binary, FFMPEG_CANDIDATES, FFPROBE_CANDIDATES

        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        video = Path(video_path)
        audio = Path(audio_path)
        meta = _probe_video(video, audio, ffmpeg_configured, find_binary, FFPROBE_CANDIDATES)
        fps = meta["fps"]
        target_w, target_h = _infinitetalk_target_size(meta["width"], meta["height"])
        total_seconds = max(1.0, min(meta["duration"] or meta["audio_seconds"], meta["audio_seconds"]))
        if total_seconds <= chunk_seconds * 1.25:
            chunks = [(0.0, total_seconds)]
        else:
            count = max(2, round(total_seconds / chunk_seconds))
            length = total_seconds / count
            chunks = [(i * length, min(length, total_seconds - i * length)) for i in range(count)]
        total_chunks = len(chunks)
        if progress:
            progress(
                f"InfiniteTalk 参数：{target_w}x{target_h} @ {fps:.3g}fps，全长 {total_seconds:.1f}s，分 {total_chunks} 段跑",
                5,
            )
        ffmpeg = find_binary(ffmpeg_configured, FFMPEG_CANDIDATES)

        def run_ffmpeg(args: list[str]) -> None:
            result = subprocess.run(
                [ffmpeg, *args, "-loglevel", "error"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg_failed: {result.stderr[:400]}")

        work = output_root / f"it_parts_{uuid.uuid4().hex[:6]}"
        work.mkdir(parents=True, exist_ok=True)
        project_slug = _safe_slug(video.stem, 24)
        part_outputs: list[Path] = []
        prompt_ids: list[str] = []
        for index, (start, seconds) in enumerate(chunks):
            base = 10 + int(index / total_chunks * 78)
            if progress:
                progress(f"分段 {index + 1}/{total_chunks}：截取 {seconds:.1f}s 片段", base)
            part_video = work / f"part_{index:03d}_video.mp4"
            part_audio = work / f"part_{index:03d}_audio.wav"
            run_ffmpeg(["-y", "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}", "-i", str(video),
                        "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-pix_fmt", "yuv420p", "-an", str(part_video)])
            run_ffmpeg(["-y", "-ss", f"{start:.3f}", "-t", f"{seconds:.3f}", "-i", str(audio),
                        "-c:a", "pcm_s16le", str(part_audio)])
            part_meta = _probe_video(part_video, part_audio, ffmpeg_configured, find_binary, FFPROBE_CANDIDATES)
            num_frames = max(25, min(part_meta["video_frames"], int(part_meta["audio_seconds"] * fps + 0.5)))
            if progress:
                progress(f"分段 {index + 1}/{total_chunks}：上传并提交工作流（{num_frames} 帧）", base + 2)
            subfolder = f"foreign_lipsync/{project_slug}_{uuid.uuid4().hex[:8]}"
            video_name = self.upload_file(part_video, subfolder=subfolder)
            audio_name = self.upload_file(part_audio, subfolder=subfolder)
            workflow = self.build_infinitetalk_workflow(
                video_name=video_name,
                audio_name=audio_name,
                width=target_w,
                height=target_h,
                fps=fps,
                num_frames=num_frames,
                positive_prompt=positive_prompt,
                filename_prefix=f"foreign_lipsync/{project_slug}_it{index:03d}",
            )
            prompt_id, _client_id = self.submit_workflow(workflow)
            prompt_ids.append(prompt_id)

            def chunk_progress(message: str, percent: int, _base: int = base, _index: int = index) -> None:
                if progress:
                    span = max(8, int(70 / total_chunks))
                    mapped = _base + 3 + int(percent / 100 * span)
                    progress(f"[分段 {_index + 1}/{total_chunks}] {message}", min(93, mapped))

            history = self.wait_for_completion(prompt_id, timeout_seconds=timeout_seconds, progress=chunk_progress)
            self.raise_for_history_error(prompt_id, history)
            asset = first_output_asset(history.get(prompt_id, {}).get("outputs"))
            if not asset:
                raise RuntimeError(f"comfy_infinitetalk_output_not_found: part {index}")
            part_out = work / f"part_{index:03d}_lipsync.mp4"
            self.download_output_asset(asset, part_out)
            part_outputs.append(part_out)
            if progress:
                progress(f"分段 {index + 1}/{total_chunks} 完成", min(94, base + max(8, int(70 / total_chunks))))

        final = output_root / "lipsync_infinitetalk.mp4"
        if len(part_outputs) == 1:
            shutil.copyfile(part_outputs[0], final)
        else:
            if progress:
                progress("拼接分段结果", 96)
            list_file = work / "concat.txt"
            list_file.write_text("".join(f"file '{p.as_posix()}'\n" for p in part_outputs), encoding="utf-8")
            run_ffmpeg(["-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(final)])
        return {
            "ok": True,
            "prompt_id": prompt_ids[-1] if prompt_ids else "",
            "prompt_ids": prompt_ids,
            "output_video_path": str(final),
            "parts": [str(p) for p in part_outputs],
            "width": target_w,
            "height": target_h,
            "fps": fps,
            "chunks": total_chunks,
        }

    def submit_workflow(self, workflow: dict[str, Any]) -> tuple[str, str]:
        client_id = uuid.uuid4().hex
        response = self.session.post(
            f"{self.base_url}/prompt",
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

    def wait_for_completion(
        self,
        prompt_id: str,
        *,
        timeout_seconds: int,
        progress: Callable[[str, int], None] | None = None,
    ) -> dict[str, Any]:
        started = time.time()
        last_log = 0.0
        while True:
            if time.time() - started > timeout_seconds:
                raise TimeoutError(f"comfy_lipsync_timeout: {prompt_id}")
            history = self.fetch_history(prompt_id)
            if prompt_id in history:
                if progress:
                    progress("Comfy 口型工作流完成", 92)
                return history
            if progress and time.time() - last_log >= 10:
                last_log = time.time()
                elapsed = int(time.time() - started)
                percent = min(90, 20 + elapsed // 6)
                progress(f"等待 Comfy 口型结果 {elapsed}s", percent)
            time.sleep(5)

    def fetch_history(self, prompt_id: str) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}/history/{prompt_id}", timeout=60)
        if response.status_code in {408, 429, 500, 502, 503, 504}:
            return {}
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    def download_output_asset(self, asset: dict[str, Any], target: Path) -> str:
        target.parent.mkdir(parents=True, exist_ok=True)
        response = self.session.get(
            f"{self.base_url}/view",
            params={
                "filename": asset["filename"],
                "subfolder": asset.get("subfolder", ""),
                "type": asset.get("type", "output"),
            },
            timeout=180,
        )
        response.raise_for_status()
        target.write_bytes(response.content)
        return response.url

    @staticmethod
    def raise_for_history_error(prompt_id: str, history: dict[str, Any]) -> None:
        item = history.get(prompt_id) if isinstance(history, dict) else None
        if not isinstance(item, dict):
            raise RuntimeError("comfy_history_empty")
        status = item.get("status") if isinstance(item.get("status"), dict) else {}
        if status.get("completed") is False or str(status.get("status_str") or "").lower() == "error":
            raise RuntimeError("ComfyUI workflow failed: " + json.dumps(status, ensure_ascii=False)[:2000])


def _get_json(session: requests.Session, url: str, timeout: float) -> Any:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _inventory_path(cache_root: str | Path, base_url: str) -> Path:
    parsed = urlparse(base_url)
    host = parsed.netloc or parsed.path
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", host).strip("_") or "comfyui"
    return Path(cache_root) / f"{safe}.json"


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


def _probe_video(video_path: Path, audio_path: Path, ffmpeg_configured: str, find_binary, ffprobe_candidates) -> dict[str, Any]:
    import subprocess

    ffprobe = find_binary(ffmpeg_configured, ffprobe_candidates)

    def probe(path: Path) -> dict[str, Any]:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe_failed: {path}: {result.stderr[:300]}")
        return json.loads(result.stdout or "{}")

    video = probe(video_path)
    stream = next((s for s in video.get("streams") or [] if s.get("codec_type") == "video"), {})
    rate = str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "25/1")
    num, _, den = rate.partition("/")
    fps = float(num) / max(float(den or 1), 1e-6) if num else 25.0
    fps = fps if 1 <= fps <= 120 else 25.0
    duration = float(video.get("format", {}).get("duration") or stream.get("duration") or 0.0)
    nb_frames = int(float(stream.get("nb_frames") or 0)) or int(duration * fps + 0.5)
    audio = probe(audio_path)
    audio_seconds = float(audio.get("format", {}).get("duration") or 0.0)
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": fps,
        "duration": duration,
        "video_frames": nb_frames,
        "audio_seconds": audio_seconds or duration,
    }


def _infinitetalk_target_size(width: int, height: int, max_side: int = 832) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        return 480, 832
    if height >= width:
        target_h = max_side
        target_w = round(width * max_side / height / 16) * 16
    else:
        target_w = max_side
        target_h = round(height * max_side / width / 16) * 16
    return max(240, target_w), max(240, target_h)


def first_output_asset(outputs: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(outputs, dict):
        return None
    assets: list[dict[str, Any]] = []
    collect_output_assets(outputs, assets)
    video_assets = [
        asset for asset in assets
        if Path(str(asset.get("filename") or "")).suffix.lower() in {".mp4", ".webm", ".mov", ".mkv", ".gif"}
    ]
    return (video_assets or assets or [None])[0]


def collect_output_assets(value: Any, assets: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        if value.get("filename"):
            assets.append(value)
            return
        for child in value.values():
            collect_output_assets(child, assets)
    elif isinstance(value, list):
        for child in value:
            collect_output_assets(child, assets)
