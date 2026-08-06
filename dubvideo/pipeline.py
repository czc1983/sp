from __future__ import annotations

from pathlib import Path
from typing import Any

import re
import subprocess

from .asr_client import (
    AsrSettings,
    DashScopeFileTransAsrClient,
    OpenAICompatibleAsrClient,
    supports_diarization,
    DIARIZATION_FALLBACK_MODEL,
)
from .comfy_provider import ComfyLipSyncProvider
import json

from .ffmpeg_tools import FFMPEG_CANDIDATES, extract_audio_for_asr, extract_speaker_sample, extract_video_frame_crop, find_binary, mask_burned_subtitles, merge_manifest_audio, mix_voice_bgm_to_video, replace_video_audio
from .jobs import append_log, should_cancel, update_job
from .models import TtsSegment
from .emotion_client import EmotionSettings, QwenAudioEmotionClient
from .ocr_client import OcrSettings, QwenOcrClient, choose_source_text, merge_ocr_texts, throttle
from .project_store import invalidate_downstream_outputs, load_project, save_project
from .settings_store import load_settings
from .subtitle_tools import save_srt
from .translate_client import OpenAICompatibleTranslator, TranslateSettings
from .tts_client_minimax import MiniMaxTtsClient, MiniMaxTtsSettings
from .videoretalk_client import VideoRetalkClient
from .vocal_separator import ensure_stems, separate_vocals
from .voice_clone_client import MiniMaxVoiceCloneClient, VoiceCloneSettings, generate_voice_id


MIN_CLONE_SOURCE_SECONDS = 2.0
MIN_CLONE_SAMPLE_SECONDS = 10.5
MAX_CLONE_SAMPLE_SECONDS = 120.0


def run_voice_clone_job(job_id: str, payload: dict[str, Any]) -> None:
    """从原片音轨中按说话人截取片段，调用 MiniMax 音色复刻，得到 speaker -> voice_id 映射。"""
    project_id = str(payload.get("project_id") or "").strip()
    if not project_id:
        raise ValueError("missing_project_id")
    project = load_project(project_id)
    root = Path(str(project["paths"]["project_dir"]))
    raw_segments = project.get("source_segments") or project.get("segments") or []
    if not raw_segments:
        raise ValueError("请先自动识别原文字幕，得到每个说话人的时间轴。")

    settings = load_settings()
    tts_dict = settings["foreign_dub"]["tts_minimax"]
    clone_settings = VoiceCloneSettings.from_tts_settings(tts_dict)
    if not clone_settings.api_key:
        raise ValueError("MiniMax API Key 未配置，无法复刻音色。")
    ffmpeg_configured = str(settings["foreign_dub"]["paths"].get("ffmpeg") or "")

    def progress(message: str, percent: int) -> None:
        append_log(job_id, "> " + message)
        update_job(job_id, progress=max(1, min(99, percent)))
        if should_cancel(job_id):
            raise RuntimeError("job_cancelled")

    progress("正在准备原片音轨", 5)
    outputs = project.setdefault("outputs", {})
    audio_path = Path(str(outputs.get("source_audio") or ""))
    if not audio_path.is_file():
        video_path = Path(str(project.get("paths", {}).get("source_video") or ""))
        if not video_path.is_file():
            raise FileNotFoundError("请先上传视频，再提取音色。")
        audio_path = extract_audio_for_asr(
            video_path,
            root / "audio" / "source_asr.mp3",
            ffmpeg_configured=ffmpeg_configured,
        )
        outputs["source_audio"] = str(audio_path)
        save_project(project)

    # 先把人声从背景音乐中分离出来，避免复刻出的音色带 BGM
    sample_source = audio_path
    vocals_path = Path(str(outputs.get("vocals_audio") or ""))
    if not vocals_path.is_file():
        try:
            progress("正在分离人声（去除背景音乐）", 7)
            vocals_path = separate_vocals(
                audio_path,
                root / "audio",
                log=lambda message: append_log(job_id, f"> {message}"),
            )
            outputs["vocals_audio"] = str(vocals_path)
            inst_matches = sorted((root / "audio").glob(f"{audio_path.stem}_*(Instrumental)*.wav"))
            if inst_matches:
                outputs["instrumental_audio"] = str(inst_matches[0])
            save_project(project)
            append_log(job_id, f"> 人声分离完成：{vocals_path.name}")
        except Exception as exc:  # noqa: BLE001 - 分离失败时退回原始音轨
            append_log(job_id, f"> 人声分离失败，改用原始音轨提取音色: {exc}")
    if vocals_path.is_file():
        sample_source = vocals_path

    speaker_ranges: dict[str, list[tuple[float, float]]] = {}
    speaker_order: list[str] = []
    for index, item in enumerate(raw_segments, start=1):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or item.get("segment_type") or "dialogue").strip().lower()
        if role in {"ignore", "mute", "silent", "none"}:
            continue
        speaker = _friendly_speaker(item.get("speaker"), index)
        start = float(item.get("start") or 0.0)
        end = max(start + 0.05, float(item.get("end") or (start + 1.0)))
        if speaker not in speaker_ranges:
            speaker_ranges[speaker] = []
            speaker_order.append(speaker)
        speaker_ranges[speaker].append((start, end))
    if not speaker_order:
        raise ValueError("字幕段里没有可用于提取音色的说话人。")

    client = MiniMaxVoiceCloneClient()
    speaker_voices = {str(k): str(v) for k, v in (project.get("speaker_voices") or {}).items() if str(v or "").strip()}
    cloned: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    total = len(speaker_order)
    for index, speaker in enumerate(speaker_order, start=1):
        if should_cancel(job_id):
            raise RuntimeError("job_cancelled")
        ranges = speaker_ranges[speaker]
        total_seconds = sum(max(0.0, end - start) for start, end in ranges)
        base_percent = 10 + int((index - 1) / max(total, 1) * 85)
        if speaker_voices.get(speaker):
            progress(f"{speaker}：已有复刻音色 {speaker_voices[speaker]}，跳过", base_percent)
            continue
        if total_seconds < MIN_CLONE_SOURCE_SECONDS:
            reason = f"有效原声只有 {total_seconds:.1f}s，太少无法复刻，将使用默认音色"
            skipped.append({"speaker": speaker, "seconds": round(total_seconds, 2), "reason": reason})
            progress(f"{speaker}：{reason}", base_percent)
            continue
        try:
            safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", speaker) or f"speaker_{index}"
            sample_path = root / "audio" / "voice_samples" / f"{index:02d}_{safe_name}.mp3"
            pad_note = "，不足 10s 将循环补足" if total_seconds < MIN_CLONE_SAMPLE_SECONDS else ""
            progress(f"{speaker}：截取原声片段（{total_seconds:.1f}s，{len(ranges)} 段{pad_note}）", base_percent)
            sample = extract_speaker_sample(
                sample_source,
                ranges,
                sample_path,
                max_seconds=MAX_CLONE_SAMPLE_SECONDS,
                min_seconds=MIN_CLONE_SAMPLE_SECONDS,
                ffmpeg_configured=ffmpeg_configured,
            )
            if sample.get("looped"):
                append_log(job_id, f"> {speaker}：原声 {sample.get('source_seconds')}s，已循环补足到 {sample.get('duration')}s")
            progress(f"{speaker}：上传采样并复刻音色", base_percent + 3)
            file_id = client.upload_file(sample["output_path"], clone_settings)
            voice_id = generate_voice_id()
            client.clone_voice(file_id, voice_id, clone_settings)
            speaker_voices[speaker] = voice_id
            cloned.append({
                "speaker": speaker,
                "voice_id": voice_id,
                "sample_path": sample["output_path"],
                "sample_seconds": sample["duration"],
            })
            progress(f"{speaker}：音色复刻完成 → {voice_id}", base_percent + 6)
        except Exception as exc:  # noqa: BLE001
            errors.append({"speaker": speaker, "error": str(exc)})
            append_log(job_id, f"> {speaker} 音色复刻失败: {exc}")

    if not cloned and not speaker_voices:
        detail = "; ".join(item["error"] for item in errors) or "没有说话人有足够的原声（至少 2 秒）"
        raise RuntimeError(f"音色复刻失败：{detail}")

    project["speaker_voices"] = speaker_voices
    manifest_path = root / "audio" / "voice_clones.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    previous_cloned: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            for item in previous.get("cloned") or []:
                if isinstance(item, dict) and item.get("speaker"):
                    previous_cloned[str(item["speaker"])] = item
        except Exception:
            previous_cloned = {}
    for item in cloned:
        previous_cloned[str(item["speaker"])] = item
    manifest_path.write_text(
        json.dumps({"speaker_voices": speaker_voices, "cloned": list(previous_cloned.values()), "skipped": skipped, "errors": errors}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    outputs["voice_clone_manifest"] = str(manifest_path)
    save_project(project)
    update_job(
        job_id,
        result={
            "speaker_voices": speaker_voices,
            "cloned": cloned,
            "skipped": skipped,
            "errors": errors,
            "voice_clone_manifest": str(manifest_path),
        },
        progress=99,
    )


def run_final_mix_job(job_id: str, payload: dict[str, Any]) -> None:
    """合成导出：原视频画面（流拷贝） + 整轨配音 + 分离出的背景音乐，不做口型。"""
    project_id = str(payload.get("project_id") or "").strip()
    if not project_id:
        raise ValueError("missing_project_id")
    project = load_project(project_id)
    root = Path(str(project["paths"]["project_dir"]))
    outputs = project.setdefault("outputs", {})

    settings = load_settings()
    ffmpeg_configured = str(settings["foreign_dub"]["paths"].get("ffmpeg") or "")
    bgm_volume = float(payload.get("bgm_volume") or 0.35)

    def progress(message: str, percent: int) -> None:
        append_log(job_id, "> " + message)
        update_job(job_id, progress=max(1, min(99, percent)))
        if should_cancel(job_id):
            raise RuntimeError("job_cancelled")

    video_path = Path(str(project.get("paths", {}).get("source_video") or ""))
    if not video_path.is_file():
        raise FileNotFoundError("请先上传视频。")
    voice_path = Path(str(outputs.get("translated_full_audio") or ""))
    if not voice_path.is_file():
        raise FileNotFoundError("整轨配音不存在，请先生成配音。")

    # 成片画面先去烧录字幕：原片字幕靠混流压不掉，必须在画面输入端抹除
    progress("正在检测并抹除原片烧录字幕", 5)
    clean_video = mask_burned_subtitles(video_path, ffmpeg_configured=ffmpeg_configured)
    if clean_video != video_path:
        append_log(job_id, f"> 已抹除原片烧录字幕：{clean_video.name}")
        video_path = clean_video
    else:
        append_log(job_id, "> 原片未检测到烧录字幕（或已抹过），使用原画面")

    # 背景音乐：优先复用已分离的伴奏轨；没有就从原片音轨分离
    bgm_path: Path | None = None
    inst = Path(str(outputs.get("instrumental_audio") or ""))
    if inst.is_file():
        bgm_path = inst
    else:
        source_audio = Path(str(outputs.get("source_audio") or ""))
        if not source_audio.is_file():
            source_audio = extract_audio_for_asr(
                video_path,
                root / "audio" / "source_asr.mp3",
                ffmpeg_configured=ffmpeg_configured,
            )
            outputs["source_audio"] = str(source_audio)
            save_project(project)
        try:
            progress("正在分离背景音乐", 15)
            vocals, instrumental = ensure_stems(
                source_audio,
                root / "audio",
                log=lambda message: append_log(job_id, f"> {message}"),
            )
            if vocals and Path(vocals).is_file() and Path(vocals) != source_audio:
                outputs["vocals_audio"] = str(vocals)
            if instrumental and Path(instrumental).is_file():
                outputs["instrumental_audio"] = str(instrumental)
                bgm_path = Path(instrumental)
            save_project(project)
        except Exception as exc:  # noqa: BLE001 - 分离失败则不加 BGM
            append_log(job_id, f"> 背景音乐分离失败，将只使用配音轨: {exc}")
    append_log(job_id, f"> 背景音乐：{bgm_path.name if bgm_path else '无（仅配音轨）'}，音量 {bgm_volume:.2f}")

    # 成片只混配音轨，不混背景音乐：BGM 单独导出保留，避免影响后续生成
    progress("正在混流导出成片", 60 if bgm_path else 40)
    out_path = root / "render" / "final_dub.mp4"
    mix_voice_bgm_to_video(
        video_path,
        voice_path,
        None,
        out_path,
        bgm_volume=bgm_volume,
        ffmpeg_configured=ffmpeg_configured,
    )
    outputs["final_video"] = str(out_path)
    result: dict[str, Any] = {"output_video_path": str(out_path), "bgm_volume": bgm_volume}
    if bgm_path is not None:
        try:
            bgm_export = _export_bgm_audio(bgm_path, root / "render" / "final_bgm.mp3", ffmpeg_configured=ffmpeg_configured)
            outputs["bgm_audio"] = str(bgm_export)
            result["bgm_audio"] = str(bgm_export)
            append_log(job_id, f"> 背景音乐已单独保留：{bgm_export.name}")
        except Exception as exc:  # noqa: BLE001 - 导出失败不影响成片
            append_log(job_id, f"> 背景音乐单独导出失败（成片已不含 BGM）: {exc}")
    save_project(project)
    update_job(
        job_id,
        result=result,
        progress=99,
    )


def _export_bgm_audio(source: Path, output: Path, *, ffmpeg_configured: str = "") -> Path:
    """把分离出的背景音乐轨转成独立 mp3 文件单独保留，供后续使用。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_binary(ffmpeg_configured, FFMPEG_CANDIDATES)
    command = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(source),
        "-vn",
        "-c:a", "libmp3lame",
        "-b:a", "192k",
        str(output),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "bgm_export_failed").strip())
    if not output.exists() or output.stat().st_size <= 0:
        raise RuntimeError("bgm_export_output_missing")
    return output


def _cut_audio_clip(source: Path, start: float, end: float, output: Path, *, ffmpeg_configured: str = "") -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_binary(ffmpeg_configured, FFMPEG_CANDIDATES)
    command = [
        ffmpeg, "-y", "-loglevel", "error",
        "-ss", f"{max(0.0, start):.3f}",
        "-to", f"{max(start + 0.2, end):.3f}",
        "-i", str(source),
        "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "32k",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "cut_audio_failed").strip())
    if not output.exists() or output.stat().st_size <= 0:
        raise RuntimeError("cut_audio_output_missing")
    return output


def _apply_auto_emotion(
    segments: list[TtsSegment],
    project: dict[str, Any],
    root: Path,
    settings: dict[str, Any],
    minimax: MiniMaxTtsSettings,
    log,
) -> None:
    """给每句字幕自动识别情绪（复用翻译通道的百炼 Key 调千问 Audio），失败不影响配音。"""
    if minimax.emotion:
        for segment in segments:
            segment.emotion = minimax.emotion
        log(f"使用全局情绪设置：{minimax.emotion}")
        return
    if not minimax.auto_emotion:
        return
    translate_cfg = settings["foreign_dub"].get("translate") or {}
    emotion_settings = EmotionSettings.from_translate_settings(
        translate_cfg,
        model=str(translate_cfg.get("emotion_model") or "qwen3-omni-flash"),
    )
    if not emotion_settings.configured:
        log("情绪识别模型未配置（需翻译通道的 Base URL 和 API Key），跳过自动情绪")
        return
    outputs = project.get("outputs") or {}
    audio_path = Path(str(outputs.get("vocals_audio") or outputs.get("source_audio") or ""))
    if not audio_path.is_file():
        video_path = Path(str(project.get("paths", {}).get("source_video") or ""))
        if not video_path.is_file():
            log("没有可用音源，跳过自动情绪识别")
            return
        audio_path = extract_audio_for_asr(
            video_path,
            root / "audio" / "source_asr.mp3",
            ffmpeg_configured=str(settings["foreign_dub"]["paths"].get("ffmpeg") or ""),
        )
    ffmpeg_configured = str(settings["foreign_dub"]["paths"].get("ffmpeg") or "")
    client = QwenAudioEmotionClient()
    candidates = [
        segment for segment in segments
        if segment.role in {"dialogue", "lip_sync", "onscreen"} and segment.target_seconds >= 0.4
    ]
    total = len(candidates)
    if not total:
        return
    log(f"开始自动情绪识别：{total} 句（模型 {emotion_settings.model}）")
    detected = 0
    for index, segment in enumerate(candidates, start=1):
        try:
            clip = _cut_audio_clip(
                audio_path, segment.start, segment.end,
                root / "audio" / "emotion_clips" / f"{segment.segment_id}.mp3",
                ffmpeg_configured=ffmpeg_configured,
            )
            segment.emotion = client.analyze(clip, emotion_settings)
            if segment.emotion:
                detected += 1
            log(f"情绪 {index}/{total}：{segment.segment_id} → {segment.emotion or '未识别'}")
        except Exception as exc:  # noqa: BLE001 - 单句失败不影响整体
            segment.emotion = ""
            log(f"情绪 {index}/{total}：{segment.segment_id} 识别失败（不影响配音）: {exc}")
    log(f"自动情绪识别完成：{detected}/{total} 句带情绪配音")


def run_tts_job(job_id: str, payload: dict[str, Any]) -> None:
    project_id = str(payload.get("project_id") or "").strip()
    if not project_id:
        raise ValueError("missing_project_id")
    project = load_project(project_id)
    root = Path(str(project["paths"]["project_dir"]))
    raw_segments = payload.get("segments") or project.get("segments") or []
    if not raw_segments:
        raise ValueError("missing_tts_segments")
    segments = [_segment_from_dict(item, project.get("target_language", "en")) for item in raw_segments]
    settings = load_settings()
    minimax = MiniMaxTtsSettings.from_dict(settings["foreign_dub"]["tts_minimax"])
    project_voices = {
        str(key): str(value)
        for key, value in (project.get("speaker_voices") or {}).items()
        if str(value or "").strip()
    }
    if project_voices:
        merged_map = dict(minimax.speaker_voice_map or {})
        merged_map.update(project_voices)
        minimax.speaker_voice_map = merged_map
        append_log(job_id, f"> 已加载原片复刻音色：{len(project_voices)} 个说话人")
    client = MiniMaxTtsClient()

    def progress(message: str, percent: int) -> None:
        append_log(job_id, "> " + message)
        update_job(job_id, progress=max(1, min(99, percent)))
        if should_cancel(job_id):
            raise RuntimeError("job_cancelled")

    output_dir = root / "audio" / "tts_segments"
    try:
        _apply_auto_emotion(
            segments, project, root, settings, minimax,
            log=lambda message: append_log(job_id, f"> {message}"),
        )
    except Exception as exc:  # noqa: BLE001 - 情绪识别失败不阻塞配音
        append_log(job_id, f"> 自动情绪识别失败（不影响配音）: {exc}")
    result = client.synthesize_segments(segments, output_dir, minimax, progress=progress)
    if should_cancel(job_id):
        raise RuntimeError("job_cancelled")
    append_log(job_id, "> 保存字幕文件")
    translated_srt = save_srt(root / "subtitles" / "translated.srt", [segment.to_dict() for segment in segments])
    append_log(job_id, "> 合成整轨配音")
    update_job(job_id, progress=95)
    full_audio = root / "audio" / "translated_full.wav"
    auto_fit = bool(settings["foreign_dub"].get("defaults", {}).get("auto_fit_translation_duration", True))
    merge_result = merge_manifest_audio(
        result["manifest_path"],
        full_audio,
        ffmpeg_configured=str(settings["foreign_dub"]["paths"].get("ffmpeg") or ""),
        sample_rate=minimax.sample_rate,
        fit_to_target=auto_fit,
    )
    driver_result: dict[str, Any] | None = None
    try:
        driver_result = merge_manifest_audio(
            result["manifest_path"],
            root / "audio" / "lipsync_driver.wav",
            ffmpeg_configured=str(settings["foreign_dub"]["paths"].get("ffmpeg") or ""),
            sample_rate=minimax.sample_rate,
            fit_to_target=auto_fit,
            role_filter={"dialogue", "lip_sync", "onscreen"},
        )
    except ValueError:
        driver_result = None
    project["segments"] = [segment.to_dict() for segment in segments]
    outputs = project.setdefault("outputs", {})
    outputs["tts_manifest"] = result["manifest_path"]
    outputs["translated_srt"] = str(translated_srt)
    outputs["translated_full_audio"] = merge_result["output_path"]
    if driver_result:
        outputs["lipsync_driver_audio"] = driver_result["output_path"]
    else:
        outputs.pop("lipsync_driver_audio", None)
    project["status"] = "tts_done"
    save_project(project)
    result["translated_srt"] = str(translated_srt)
    result["translated_full_audio"] = merge_result["output_path"]
    result["merge"] = merge_result
    if driver_result:
        result["lipsync_driver_audio"] = driver_result["output_path"]
        result["driver_merge"] = driver_result
    update_job(job_id, result=result, progress=99)


def run_asr_job(job_id: str, payload: dict[str, Any]) -> None:
    project_id = str(payload.get("project_id") or "").strip()
    if not project_id:
        raise ValueError("missing_project_id")
    project = load_project(project_id)
    root = Path(str(project["paths"]["project_dir"]))
    video_path = Path(str(payload.get("video_path") or project.get("paths", {}).get("source_video") or ""))
    if not video_path.is_file():
        raise FileNotFoundError("请先上传视频，再自动识别。")

    settings = load_settings()
    asr_settings = AsrSettings.from_dict(settings["foreign_dub"]["asr"])

    def progress(message: str, percent: int) -> None:
        append_log(job_id, "> " + message)
        update_job(job_id, progress=max(1, min(99, percent)))
        if should_cancel(job_id):
            raise RuntimeError("job_cancelled")

    progress("正在从视频提取音频", 8)
    audio_path = extract_audio_for_asr(
        video_path,
        root / "audio" / "source_asr.mp3",
        ffmpeg_configured=str(settings["foreign_dub"]["paths"].get("ffmpeg") or ""),
    )
    project.setdefault("outputs", {})["source_audio"] = str(audio_path)
    save_project(project)

    progress(f"正在调用 ASR：{asr_settings.model}", 20)
    provider = asr_settings.provider.strip().lower()
    if provider == "dashscope_filetrans" or "filetrans" in asr_settings.model or supports_diarization(asr_settings.model):
        if asr_settings.diarization_enabled and not supports_diarization(asr_settings.model):
            append_log(
                job_id,
                f"> 模型 {asr_settings.model} 不支持说话人分离，本次自动改用 {DIARIZATION_FALLBACK_MODEL}（可在设置中固定）",
            )
            asr_settings.model = DIARIZATION_FALLBACK_MODEL
        if asr_settings.diarization_enabled:
            append_log(job_id, "> 已开启说话人分离：ASR 会给每句标注不同说话人")
        result = DashScopeFileTransAsrClient().transcribe_audio(audio_path, asr_settings, progress=progress)
    else:
        result = OpenAICompatibleAsrClient().transcribe_audio(audio_path, asr_settings)
    if should_cancel(job_id):
        raise RuntimeError("job_cancelled")

    source_language = str(
        result.get("language")
        or payload.get("source_language")
        or project.get("source_language")
        or "auto"
    )
    segments: list[dict[str, Any]] = []
    for item in result.get("segments") or []:
        if not isinstance(item, dict):
            continue
        segment = dict(item)
        segment["language"] = str(segment.get("language") or source_language)
        segments.append(segment)
    if not segments:
        raise RuntimeError("ASR 没有识别到字幕段。")
    segments.sort(key=lambda item: (float(item.get("start") or 0.0), float(item.get("end") or 0.0)))
    for index, segment in enumerate(segments, start=1):
        segment["segment_id"] = str(index)
        segment["speaker"] = _friendly_speaker(segment.get("speaker"), index)

    progress("正在保存原文字幕", 88)
    source_srt = save_srt(root / "subtitles" / "source.srt", segments)
    project = invalidate_downstream_outputs(project)
    project["source_segments"] = segments
    project["source_language"] = source_language
    project["source_reviewed"] = False
    project["source_revision_reason"] = "asr"
    project.setdefault("outputs", {})["source_audio"] = str(audio_path)
    project.setdefault("outputs", {})["source_srt"] = str(source_srt)
    project["status"] = "source_script_ready"
    save_project(project)
    update_job(
        job_id,
        result={
            "source_audio": str(audio_path),
            "source_srt": str(source_srt),
            "segments": segments,
            "language": source_language,
            "count": len(segments),
        },
        progress=99,
    )


def run_ocr_job(job_id: str, payload: dict[str, Any]) -> None:
    project_id = str(payload.get("project_id") or "").strip()
    if not project_id:
        raise ValueError("missing_project_id")
    project = load_project(project_id)
    root = Path(str(project["paths"]["project_dir"]))
    video_path = Path(str(payload.get("video_path") or project.get("paths", {}).get("source_video") or ""))
    if not video_path.is_file():
        raise FileNotFoundError("请先上传视频，再做画面字幕 OCR。")
    raw_segments = project.get("source_segments") or []
    if not raw_segments:
        raise ValueError("请先自动识别，得到每句时间轴，再做画面字幕 OCR。")

    settings = load_settings()
    ocr_settings = OcrSettings.from_dict(settings["foreign_dub"].get("ocr", {}))
    crop_ratio = float(payload.get("crop_bottom_ratio") or ocr_settings.crop_bottom_ratio or 0.5)
    frames_per_segment = int(payload.get("frames_per_segment") or ocr_settings.frames_per_segment or 6)
    ffmpeg_configured = str(settings["foreign_dub"]["paths"].get("ffmpeg") or "")
    client = QwenOcrClient()
    frame_dir = root / "ocr_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    prompt = (
        "只输出画面中的中文字幕原文。不要改写，不要补字，不要输出人物、背景、数字编号或解释。"
        "如果没有清晰字幕，输出空字符串。"
    )

    def progress(message: str, percent: int) -> None:
        append_log(job_id, "> " + message)
        update_job(job_id, progress=max(1, min(99, percent)))
        if should_cancel(job_id):
            raise RuntimeError("job_cancelled")

    corrected: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    changed_count = 0
    last_request_started = 0.0
    ordered_segments = [dict(item) for item in raw_segments if isinstance(item, dict)]
    ordered_segments.sort(key=lambda item: (float(item.get("start") or 0.0), float(item.get("end") or 0.0)))
    total = len(ordered_segments)
    for offset, raw in enumerate(ordered_segments):
        index = offset + 1
        segment = dict(raw)
        sid = str(segment.get("segment_id") or index)
        start = float(segment.get("start") or 0.0)
        end = max(start + 0.1, float(segment.get("end") or (start + 1.0)))
        original_text = str(segment.get("text") or "")
        asr_text = str(segment.get("asr_text") or original_text)
        previous_end = None
        next_start = None
        if offset > 0:
            previous_end = float(ordered_segments[offset - 1].get("end") or 0.0)
        if offset + 1 < total:
            next_start = float(ordered_segments[offset + 1].get("start") or 0.0)
        samples: list[dict[str, Any]] = []
        progress(f"OCR 校正字幕 {index}/{total}", int((index - 1) / max(total, 1) * 92))
        sample_times = _sample_times(start, end, frames_per_segment, previous_end=previous_end, next_start=next_start)
        for sample_index, timestamp in enumerate(sample_times, start=1):
            frame_path = extract_video_frame_crop(
                video_path,
                frame_dir / f"{sid}_{sample_index}.png",
                timestamp=timestamp,
                crop_bottom_ratio=crop_ratio,
                ffmpeg_configured=ffmpeg_configured,
            )
            last_request_started = throttle(last_request_started, ocr_settings.request_interval_ms)
            text = client.recognize_image(frame_path, ocr_settings, prompt=prompt)
            samples.append({
                "time": round(timestamp, 3),
                "text": text,
                "frame": str(frame_path),
            })
        merged = merge_ocr_texts([sample["text"] for sample in samples])
        segment["start"] = start
        segment["end"] = end
        segment["timing_source"] = "asr"
        if merged:
            chosen_text, text_source = choose_source_text(asr_text, merged)
            segment["asr_text"] = asr_text
            segment["ocr_text"] = merged
            segment["text"] = chosen_text
            segment["text_source"] = text_source
            segment["needs_review"] = text_source in {"ocr_needs_review", "ocr_asr_conflict"}
            if chosen_text != original_text:
                changed_count += 1
        else:
            segment["asr_text"] = asr_text
            segment["text"] = asr_text or original_text
            segment["text_source"] = segment.get("text_source") or "asr"
            segment["needs_review"] = True
        segment["ocr_samples"] = samples
        corrected.append(segment)
        manifest.append({
            "segment_id": sid,
            "samples": samples,
            "merged_text": merged,
            "chosen_text": segment.get("text") or "",
            "text_source": segment.get("text_source") or "",
            "asr_time": {"start": start, "end": end},
            "ocr_search_time": {
                "start": min(sample_times) if sample_times else start,
                "end": max(sample_times) if sample_times else end,
            },
        })

    source_srt = save_srt(root / "subtitles" / "source.srt", corrected)
    manifest_path = root / "subtitles" / "ocr_manifest.json"
    manifest_path.write_text(json.dumps({"segments": manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    project = invalidate_downstream_outputs(project)
    project["source_segments"] = corrected
    project["source_reviewed"] = False
    project["source_revision_reason"] = "ocr"
    outputs = project.setdefault("outputs", {})
    outputs["source_srt"] = str(source_srt)
    outputs["ocr_manifest"] = str(manifest_path)
    project["status"] = "ocr_corrected"
    save_project(project)
    update_job(
        job_id,
        result={
            "source_srt": str(source_srt),
            "ocr_manifest": str(manifest_path),
            "segments": corrected,
            "count": len(corrected),
            "changed_count": changed_count,
        },
        progress=99,
    )


def run_lipsync_job(job_id: str, payload: dict[str, Any]) -> None:
    project_id = str(payload.get("project_id") or "").strip()
    if not project_id:
        raise ValueError("missing_project_id")
    project = load_project(project_id)
    root = Path(str(project["paths"]["project_dir"]))
    video_path = Path(str(payload.get("video_path") or project.get("paths", {}).get("source_video") or ""))
    outputs = project.get("outputs", {})
    final_audio_path = Path(str(outputs.get("translated_full_audio") or ""))
    audio_path = Path(str(payload.get("audio_path") or outputs.get("lipsync_driver_audio") or ""))
    if not video_path.is_file():
        raise FileNotFoundError(f"source_video_missing: {video_path}")
    if not final_audio_path.is_file():
        raise FileNotFoundError(f"translated_full_audio_missing: {final_audio_path}")

    settings = load_settings()
    comfy_settings = settings["foreign_dub"]["lipsync_comfy"]
    ffmpeg_configured = str(settings["foreign_dub"]["paths"].get("ffmpeg") or "")
    if not audio_path.is_file():
        progress_result = replace_video_audio(
            video_path,
            final_audio_path,
            root / "render" / "foreign_audio_only.mp4",
            ffmpeg_configured=ffmpeg_configured,
        )
        project.setdefault("outputs", {})["lipsync_video"] = progress_result["output_path"]
        project["status"] = "audio_replaced"
        save_project(project)
        update_job(
            job_id,
            result={**progress_result, "mode": "audio_replace_only", "output_video_path": progress_result["output_path"]},
            progress=99,
        )
        return
    client = ComfyLipSyncProvider(
        str(comfy_settings.get("base_url") or ""),
        timeout=float(comfy_settings.get("timeout_seconds") or 5400),
    )

    def progress(message: str, percent: int) -> None:
        append_log(job_id, "> " + message)
        update_job(job_id, progress=max(1, min(99, percent)))
        if should_cancel(job_id):
            raise RuntimeError("job_cancelled")

    cloud = settings["foreign_dub"].get("lipsync_cloud") or {}
    cloud_provider = str(cloud.get("provider") or "").strip().lower()
    if cloud_provider in {"videoretalk", "pixverse_lipsync"}:
        api_key = str(cloud.get("api_key") or settings["foreign_dub"]["asr"].get("api_key") or "").strip()
        if not api_key:
            raise ValueError("百炼 API Key 未配置，无法使用云端口型（设置里填 ASR 或口型云的 Key）。")
        asr_base = str(settings["foreign_dub"]["asr"].get("base_url") or "").strip()
        base_url = str(cloud.get("base_url") or "").strip()
        if cloud_provider == "pixverse_lipsync":
            # 爱诗 PixVerse 是第三方模型，走业务空间（workspace）端点
            model = "pixverse/pixverse-lipsync"
            if not base_url:
                base_url = asr_base or "https://dashscope.aliyuncs.com"
        else:
            model = str(cloud.get("model") or "videoretalk")
            if not base_url:
                base_url = "https://dashscope.aliyuncs.com" if "maas.aliyuncs.com" in asr_base else asr_base
        vr_client = VideoRetalkClient()
        vr_result = vr_client.run(
            video_path=video_path,
            audio_path=audio_path,
            output_path=root / "render" / f"{model.split('/')[-1]}_raw.mp4",
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=int(cloud.get("timeout_seconds") or 3600),
            poll_interval_seconds=float(cloud.get("poll_interval_seconds") or 5),
            progress=progress,
        )
        raw_lipsync_video = vr_result["output_video_path"]
        final_mux = replace_video_audio(
            raw_lipsync_video,
            final_audio_path,
            root / "render" / "lipsync_final_audio.mp4",
            ffmpeg_configured=ffmpeg_configured,
        )
        vr_result["raw_lipsync_video"] = raw_lipsync_video
        vr_result["final_audio_mux"] = final_mux
        vr_result["output_video_path"] = final_mux["output_path"]
        project.setdefault("outputs", {})["raw_lipsync_video"] = raw_lipsync_video
        project.setdefault("outputs", {})["lipsync_video"] = vr_result["output_video_path"]
        if vr_result.get("task_id"):
            project.setdefault("outputs", {})["videoretalk_task_id"] = vr_result["task_id"]
        project["status"] = "lipsync_done"
        save_project(project)
        update_job(job_id, result=vr_result, progress=99)
        return

    profile = str(comfy_settings.get("profile") or "kling_audio").strip()
    if profile == "infinitetalk":
        result = client.run_infinitetalk_lipsync(
            video_path=video_path,
            audio_path=audio_path,
            output_dir=root / "render",
            ffmpeg_configured=ffmpeg_configured,
            timeout_seconds=int(comfy_settings.get("timeout_seconds") or 5400),
            progress=progress,
        )
    else:
        voice_language = str(payload.get("voice_language") or project.get("target_language") or "en")
        result = client.run_kling_audio_lipsync(
            video_path=video_path,
            audio_path=audio_path,
            output_dir=root / "render",
            voice_language=voice_language,
            timeout_seconds=int(comfy_settings.get("timeout_seconds") or 5400),
            progress=progress,
        )
    raw_lipsync_video = result["output_video_path"]
    final_mux = replace_video_audio(
        raw_lipsync_video,
        final_audio_path,
        root / "render" / "lipsync_final_audio.mp4",
        ffmpeg_configured=ffmpeg_configured,
    )
    result["raw_lipsync_video"] = raw_lipsync_video
    result["final_audio_mux"] = final_mux
    result["output_video_path"] = final_mux["output_path"]
    project.setdefault("outputs", {})["raw_lipsync_video"] = raw_lipsync_video
    project.setdefault("outputs", {})["lipsync_video"] = result["output_video_path"]
    project.setdefault("outputs", {})["lipsync_prompt_id"] = result["prompt_id"]
    project["status"] = "lipsync_done"
    save_project(project)
    update_job(job_id, result=result, progress=99)


def run_translate_job(job_id: str, payload: dict[str, Any]) -> None:
    project_id = str(payload.get("project_id") or "").strip()
    if not project_id:
        raise ValueError("missing_project_id")
    project = load_project(project_id)
    raw_segments = (
        payload.get("source_segments")
        or project.get("source_segments")
        or project.get("segments")
        or []
    )
    if not raw_segments:
        raise ValueError("请先自动识别原文字幕，或导入原文 SRT。")
    if not project.get("source_reviewed"):
        raise ValueError("请先到 03 字幕校对保存原文字幕，再自动翻译。")
    source_segments = [_source_segment_dict(item, project.get("source_language", "auto")) for item in raw_segments]
    settings = load_settings()
    translate_settings = TranslateSettings.from_dict(settings["foreign_dub"]["translate"])
    client = OpenAICompatibleTranslator()

    def progress(message: str, percent: int) -> None:
        append_log(job_id, "> " + message)
        update_job(job_id, progress=max(1, min(99, percent)))
        if should_cancel(job_id):
            raise RuntimeError("job_cancelled")

    target_language = str(payload.get("target_language") or project.get("target_language") or "en")
    source_language = str(payload.get("source_language") or project.get("source_language") or "auto")
    translated = client.translate_segments(
        source_segments,
        translate_settings,
        source_language=source_language,
        target_language=target_language,
        progress=progress,
    )
    root = Path(str(project["paths"]["project_dir"]))
    source_srt = save_srt(root / "subtitles" / "source.srt", source_segments)
    translated_srt = save_srt(root / "subtitles" / "translated.srt", translated)
    project["source_segments"] = source_segments
    project["segments"] = translated
    project["source_language"] = source_language
    project["target_language"] = target_language
    project.setdefault("outputs", {})["source_srt"] = str(source_srt)
    project.setdefault("outputs", {})["translated_srt"] = str(translated_srt)
    project["status"] = "translate_done"
    save_project(project)
    update_job(
        job_id,
        result={
            "source_srt": str(source_srt),
            "translated_srt": str(translated_srt),
            "segments": translated,
            "count": len(translated),
        },
        progress=99,
    )


def _segment_from_dict(item: dict[str, Any], default_language: str) -> TtsSegment:
    segment_id = str(item.get("segment_id") or item.get("id") or "").strip()
    if not segment_id:
        raise ValueError("segment_missing_id")
    return TtsSegment(
        segment_id=segment_id,
        start=float(item.get("start") or 0.0),
        end=float(item.get("end") or 0.0),
        text=str(item.get("text") or item.get("translated_text") or item.get("tts_text") or "").strip(),
        speaker=_friendly_speaker(item.get("speaker"), 1),
        language=str(item.get("language") or default_language or "en"),
        role=str(item.get("role") or item.get("segment_type") or "dialogue"),
        emotion=str(item.get("emotion") or ""),
    )


def _source_segment_dict(item: dict[str, Any], default_language: str) -> dict[str, Any]:
    segment_id = str(item.get("segment_id") or item.get("id") or "").strip()
    if not segment_id:
        raise ValueError("segment_missing_id")
    return {
        "segment_id": segment_id,
        "start": float(item.get("start") or 0.0),
        "end": float(item.get("end") or 0.0),
        "text": str(item.get("text") or item.get("source_text") or "").strip(),
        "speaker": _friendly_speaker(item.get("speaker"), 1),
        "language": str(item.get("language") or default_language or "auto"),
        "role": str(item.get("role") or item.get("segment_type") or "dialogue"),
    }


def _friendly_speaker(value: Any, index: int = 1) -> str:
    raw = str(value or "").strip()
    lower = raw.lower()
    if lower.startswith("channel_"):
        try:
            return f"说话人{int(lower.rsplit('_', 1)[-1]) + 1}"
        except ValueError:
            return "说话人1"
    if lower.startswith("speaker_"):
        try:
            return f"说话人{max(1, int(lower.rsplit('_', 1)[-1]))}"
        except ValueError:
            return "说话人1"
    if lower in {"narration", "narrator", "voiceover", "voice_over"}:
        return "旁白"
    return raw or f"说话人{max(1, index)}"


def _sample_times(
    start: float,
    end: float,
    count: int,
    *,
    previous_end: float | None = None,
    next_start: float | None = None,
) -> list[float]:
    duration = max(0.1, end - start)
    lead = min(0.6, max(0.12, duration * 0.22))
    trail = min(0.6, max(0.12, duration * 0.18))
    pre_time = max(0.0, start - lead)
    if previous_end is not None and previous_end > 0:
        pre_time = min(start, max(pre_time, previous_end + 0.05))
    post_time = end + trail
    if next_start is not None and next_start > end:
        post_time = min(post_time, max(end, next_start - 0.05))
    elif next_start is None:
        post_time = end
    if count <= 1:
        candidates = [start]
    elif count == 2:
        candidates = [start, start + duration * 0.75]
    elif count == 3:
        candidates = [start, start + duration * 0.45, start + duration * 0.82]
    elif count == 4:
        candidates = [pre_time, start, start + duration * 0.48, start + duration * 0.82]
    elif count == 5:
        candidates = [
            pre_time,
            start,
            start + min(0.16, duration * 0.08),
            start + duration * 0.55,
            post_time,
        ]
    else:
        candidates = [
            pre_time,
            start,
            start + min(0.16, duration * 0.08),
            start + duration * 0.45,
            start + duration * 0.80,
            post_time,
        ]
    return _unique_times(candidates[: max(1, count)])


def _unique_times(values: list[float]) -> list[float]:
    result: list[float] = []
    for value in values:
        timestamp = round(max(0.0, float(value)), 3)
        if not result or abs(timestamp - result[-1]) >= 0.04:
            result.append(timestamp)
    return result
