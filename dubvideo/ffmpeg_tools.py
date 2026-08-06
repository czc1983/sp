from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


FFMPEG_CANDIDATES = [
    "ffmpeg",
    r"C:\Users\Administrator\Desktop\生成视频ppt\camel-milk-tech-video\node_modules\@remotion\compositor-win32-x64-msvc\ffmpeg.exe",
    r"C:\Program Files\lindong\resources\tools\ffmpeg\ffmpeg.exe",
    r"C:\Program Files\lindong\resources\app.asar.unpacked\node_modules\@ffmpeg-installer\win32-x64\ffmpeg.exe",
]

FFPROBE_CANDIDATES = [
    "ffprobe",
    r"C:\Users\Administrator\Desktop\生成视频ppt\camel-milk-tech-video\node_modules\@remotion\compositor-win32-x64-msvc\ffprobe.exe",
]


def find_binary(configured: str = "", candidates: list[str] | None = None) -> str:
    values = []
    if configured:
        values.append(configured)
    values.extend(candidates or [])
    for candidate in values:
        found = shutil.which(candidate)
        if found:
            return found
        path = Path(candidate)
        if path.exists():
            return str(path)
    raise FileNotFoundError("binary_not_found")


def ffmpeg_status(configured: str = "") -> dict[str, Any]:
    try:
        path = find_binary(configured, FFMPEG_CANDIDATES)
        result = subprocess.run(
            [path, "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
        )
        first = (result.stdout or result.stderr or "").splitlines()[0] if (result.stdout or result.stderr) else ""
        return {"ok": result.returncode == 0, "path": path, "version": first}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def extract_audio_for_asr(
    video_path: str | Path,
    output_path: str | Path,
    *,
    ffmpeg_configured: str = "",
) -> Path:
    video = Path(video_path)
    if not video.is_file():
        raise FileNotFoundError(f"source_video_missing: {video}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_binary(ffmpeg_configured, FFMPEG_CANDIDATES)
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "32k",
        str(output),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "extract_audio_failed").strip())
    if not output.exists() or output.stat().st_size <= 0:
        raise RuntimeError("extract_audio_output_missing")
    return output


def extract_video_frame_crop(
    video_path: str | Path,
    output_path: str | Path,
    *,
    timestamp: float,
    crop_bottom_ratio: float = 0.5,
    scale_width: int = 1280,
    ffmpeg_configured: str = "",
) -> Path:
    video = Path(video_path)
    if not video.is_file():
        raise FileNotFoundError(f"source_video_missing: {video}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_binary(ffmpeg_configured, FFMPEG_CANDIDATES)
    ratio = min(1.0, max(0.15, float(crop_bottom_ratio or 0.5)))
    if ratio >= 0.99:
        vf = f"scale={scale_width}:-1"
    else:
        vf = f"crop=iw:ih*{ratio:.4f}:0:ih-ih*{ratio:.4f},scale={scale_width}:-1"
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, float(timestamp)):.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-vf",
        vf,
        str(output),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "extract_frame_failed").strip())
    if not output.exists() or output.stat().st_size <= 0:
        raise RuntimeError("extract_frame_output_missing")
    return output


def merge_manifest_audio(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    ffmpeg_configured: str = "",
    ffprobe_configured: str = "",
    sample_rate: int = 32000,
    fit_to_target: bool = True,
    role_filter: set[str] | None = None,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"manifest_not_found: {manifest_file}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8-sig"))
    segments = [item for item in manifest.get("segments", []) if isinstance(item, dict)]
    inputs: list[dict[str, Any]] = []
    included_roles = {str(role).strip().lower() for role in role_filter} if role_filter else None
    for segment in sorted(segments, key=lambda item: float(item.get("start") or 0)):
        role = str(segment.get("role") or segment.get("segment_type") or "dialogue").strip().lower()
        if role in {"ignore", "mute", "silent", "none"}:
            continue
        if included_roles is not None and role not in included_roles:
            continue
        audio_file = str(segment.get("audio_file") or "").strip()
        if not audio_file:
            continue
        audio_path = (manifest_file.parent / audio_file).resolve()
        if not audio_path.exists() or audio_path.stat().st_size <= 0:
            continue
        delay_ms = max(0, int(round(float(segment.get("start") or 0) * 1000)))
        target_seconds = float(segment.get("target_seconds") or 0) or max(
            0.0,
            float(segment.get("end") or 0) - float(segment.get("start") or 0),
        )
        inputs.append({
            "audio_path": audio_path,
            "delay_ms": delay_ms,
            "target_seconds": target_seconds,
            "segment_id": str(segment.get("segment_id") or segment.get("id") or ""),
        })
    if not inputs:
        raise ValueError("manifest_has_no_audio_segments")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_binary(ffmpeg_configured, FFMPEG_CANDIDATES)
    ffprobe = find_binary(ffprobe_configured, FFPROBE_CANDIDATES)
    work_dir = output.parent / f".fit_{output.stem}"
    work_dir.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, "-y"]
    prepared_inputs: list[Path] = []
    for index, item in enumerate(inputs, start=1):
        source = Path(item["audio_path"])
        fitted = source
        if fit_to_target and item.get("target_seconds", 0) > 0:
            fitted = _fit_audio_to_target(
                source,
                work_dir / f"{item['segment_id'] or f'seg_{index:03d}'}.wav",
                target_seconds=float(item["target_seconds"]),
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                sample_rate=sample_rate,
            )
        prepared_inputs.append(fitted)
        command.extend(["-i", str(fitted)])

    labels: list[str] = []
    filters: list[str] = []
    for index, item in enumerate(inputs):
        delay_ms = int(item["delay_ms"])
        label = f"a{index}"
        filters.append(f"[{index}:a]adelay={delay_ms}:all=1,aresample={sample_rate}[{label}]")
        labels.append(f"[{label}]")
    if len(labels) == 1:
        filter_complex = filters[0].replace(f"[{labels[0][1:-1]}]", "[out]")
    else:
        filter_complex = (
            ";".join(filters)
            + ";"
            + "".join(labels)
            + f"amix=inputs={len(labels)}:duration=longest:normalize=0,aresample={sample_rate}[out]"
        )

    command.extend(["-filter_complex", filter_complex, "-map", "[out]", "-ac", "1", str(output)])
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "ffmpeg_merge_failed").strip())
    return {
        "ok": True,
        "output_path": str(output),
        "segment_count": len(inputs),
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "fit_to_target": bool(fit_to_target),
    }


def mix_voice_bgm_to_video(
    video_path: str | Path,
    voice_path: str | Path,
    bgm_path: str | Path | None,
    output_path: str | Path,
    *,
    bgm_volume: float = 0.35,
    ffmpeg_configured: str = "",
) -> dict[str, Any]:
    """把配音轨 + 背景音乐轨混成立体声后替换原视频音轨；画面流拷贝，不重编码。

    bgm_path 为 None 时退化为纯配音替换。配音轨先做 loudnorm 响度归一化（-14 LUFS），
    保证人声清晰突出；bgm_volume 取 0~1，默认 0.35，避免背景音乐盖过人声。
    """
    video = Path(video_path)
    voice = Path(voice_path)
    bgm = Path(bgm_path) if bgm_path else None
    if not video.is_file():
        raise FileNotFoundError(f"video_not_found: {video}")
    if not voice.is_file():
        raise FileNotFoundError(f"voice_not_found: {voice}")
    if bgm is not None and not bgm.is_file():
        bgm = None
    if bgm is None:
        return replace_video_audio(video, voice, output_path, ffmpeg_configured=ffmpeg_configured)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    bgm_volume = max(0.0, min(1.0, float(bgm_volume)))
    ffmpeg = find_binary(ffmpeg_configured, FFMPEG_CANDIDATES)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video),
        "-i",
        str(voice),
        "-i",
        str(bgm),
        "-filter_complex",
        f"[1:a]loudnorm=I=-14:TP=-1.5:LRA=11[v];[2:a]volume={bgm_volume:.2f}[m];[v][m]amix=inputs=2:duration=longest:normalize=0[a]",
        "-map",
        "0:v:0",
        "-map",
        "[a]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(output),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg_mix_failed: {(result.stderr or '')[-800:]}")
    return {"output_path": str(output), "bgm_volume": bgm_volume}


def replace_video_audio(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    *,
    ffmpeg_configured: str = "",
) -> dict[str, Any]:
    video = Path(video_path)
    audio = Path(audio_path)
    if not video.is_file():
        raise FileNotFoundError(f"video_not_found: {video}")
    if not audio.is_file():
        raise FileNotFoundError(f"audio_not_found: {audio}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_binary(ffmpeg_configured, FFMPEG_CANDIDATES)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "replace_video_audio_failed").strip())
    return {"ok": True, "output_path": str(output), "ffmpeg": ffmpeg}


def extract_speaker_sample(
    audio_path: str | Path,
    ranges: list[tuple[float, float]],
    output_path: str | Path,
    *,
    max_seconds: float = 120.0,
    min_seconds: float = 0.0,
    ffmpeg_configured: str = "",
    sample_rate: int = 16000,
    bitrate: str = "64k",
) -> dict[str, Any]:
    """从整轨音频中截取某个说话人的所有片段并拼接成一条采样音频（用于音色复刻）。

    min_seconds > 0 时，若片段总长不足，会循环重复该说话人的片段补足到 min_seconds。
    """
    source = Path(audio_path)
    if not source.is_file():
        raise FileNotFoundError(f"audio_not_found: {source}")
    base_ranges = [
        (max(0.0, float(start)), max(float(start) + 0.05, float(end)))
        for start, end in sorted(ranges, key=lambda item: item[0])
    ]
    base_total = sum(end - start for start, end in base_ranges)
    if not base_ranges or base_total <= 0:
        raise ValueError("speaker_sample_ranges_empty")
    picked: list[tuple[float, float]] = []
    total = 0.0
    target = max(float(min_seconds or 0.0), min(base_total, max_seconds))
    target = min(target, max_seconds)
    loops = 0
    max_loops = 6
    while total < target and loops < max_loops:
        for start, end in base_ranges:
            if total >= target:
                break
            duration = end - start
            if total + duration > max_seconds and picked:
                total = max_seconds
                break
            picked.append((start, end))
            total += duration
        loops += 1
    if not picked or total <= 0:
        raise ValueError("speaker_sample_ranges_empty")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_binary(ffmpeg_configured, FFMPEG_CANDIDATES)
    parts: list[str] = []
    labels: list[str] = []
    for index, (start, end) in enumerate(picked):
        label = f"s{index}"
        parts.append(f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=N/SR/TB[{label}]")
        labels.append(f"[{label}]")
    if len(labels) == 1:
        filter_complex = parts[0].replace(f"[{labels[0][1:-1]}]", "[out]")
    else:
        filter_complex = ";".join(parts) + ";" + "".join(labels) + f"concat=n={len(labels)}:v=0:a=1[out]"
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "libmp3lame",
        "-b:a",
        bitrate,
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
        raise RuntimeError((result.stderr or result.stdout or "extract_speaker_sample_failed").strip())
    if not output.exists() or output.stat().st_size <= 0:
        raise RuntimeError("extract_speaker_sample_output_missing")
    return {
        "output_path": str(output),
        "duration": round(total, 3),
        "source_seconds": round(base_total, 3),
        "looped": total > base_total + 0.01,
        "range_count": len(picked),
    }


def probe_duration(path: str | Path, *, ffprobe_configured: str = "") -> float:
    media = Path(path)
    if not media.is_file():
        raise FileNotFoundError(f"media_not_found: {media}")
    ffprobe = find_binary(ffprobe_configured, FFPROBE_CANDIDATES)
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "probe_duration_failed").strip())
    try:
        return float((result.stdout or result.stderr or "0").strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"probe_duration_invalid: {result.stdout[:200]}") from exc


def _fit_audio_to_target(
    source: Path,
    output: Path,
    *,
    target_seconds: float,
    ffmpeg: str,
    ffprobe: str,
    sample_rate: int,
) -> Path:
    duration = probe_duration(source, ffprobe_configured=ffprobe)
    if target_seconds <= 0:
        return source
    output.parent.mkdir(parents=True, exist_ok=True)
    if duration <= 0:
        shutil.copy2(source, output)
        return output
    speed = 1.0
    if duration > target_seconds:
        speed = duration / target_seconds
    filters: list[str] = []
    if speed > 1.0001:
        filters.extend(_atempo_chain(speed))
    effective_duration = duration / speed
    pad = max(0.0, target_seconds - effective_duration)
    if pad > 0.01:
        filters.append(f"apad=pad_dur={pad:.3f}")
    filters.extend([
        f"atrim=0:{target_seconds:.3f}",
        "asetpts=N/SR/TB",
        f"aresample={sample_rate}",
    ])
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-filter:a",
        ",".join(filters),
        str(output),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "fit_audio_failed").strip())
    return output if output.exists() and output.stat().st_size > 0 else source


def _atempo_chain(speed: float) -> list[str]:
    value = max(speed, 0.001)
    filters: list[str] = []
    while value > 2.0:
        filters.append("atempo=2.0")
        value /= 2.0
    while value < 0.5:
        filters.append("atempo=0.5")
        value /= 0.5
    filters.append(f"atempo={value:.6f}")
    return filters


def probe_video_size(path: str | Path, *, ffprobe_configured: str = "") -> tuple[int, int]:
    media = Path(path)
    if not media.is_file():
        raise FileNotFoundError(f"media_not_found: {media}")
    ffprobe = find_binary(ffprobe_configured, FFPROBE_CANDIDATES)
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0",
        str(media),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    line = (result.stdout or "").strip().splitlines()
    if result.returncode != 0 or not line:
        raise RuntimeError((result.stderr or "probe_video_size_failed").strip())
    try:
        width_text, height_text = line[0].split(",")[:2]
        return int(width_text), int(height_text)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"probe_video_size_invalid: {result.stdout[:200]}") from exc


def detect_subtitle_band(
    video_path: str | Path,
    width: int,
    height: int,
    duration: float,
    ffmpeg: str,
) -> tuple[int, int, int, int] | None:
    """抽样帧检测烧录字幕的真实位置，返回 delogo 用的 (x, y, w, h)；无字幕返回 None。

    短剧字幕特征：笔画密集、局部对比强、位置固定、集中在画面下方。
    逐行统计"明暗跳变密集度"（与字色无关，白/黄/粉/黑字通吃）。
    字幕跟台词走，很多片段大半时间没有字幕：抽样要密、投票门槛要低，
    由笔画成对过滤和带宽上限挡住误报。
    """
    if duration <= 0:
        return None
    small_w = 256
    small_h = max(16, round(height * small_w / width))
    fps = min(8.0, max(1.0, 24.0 / duration))
    command = [
        ffmpeg, "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"fps={fps:.4f},scale={small_w}:{small_h},format=gray",
        "-f", "rawvideo", "-",
    ]
    try:
        proc = subprocess.run(command, capture_output=True, timeout=180)
    except Exception:  # noqa: BLE001
        return None
    data = proc.stdout or b""
    frame_size = small_w * small_h
    frame_count = len(data) // frame_size
    if frame_count <= 0:
        return None
    scan_top = int(small_h * 0.55)
    row_votes = [0] * small_h
    # 认笔画不认颜色：文字行的局部对比强、明暗跳变密集，
    # 且笔画成对出现（一跳变附近 6px 内必有另一跳变），可滤掉孤立的画面噪点
    edge_min = 64
    hits_needed = max(8, int(small_w * 0.04))
    for f in range(frame_count):
        frame = data[f * frame_size:(f + 1) * frame_size]
        for y in range(scan_top, small_h):
            row = frame[y * small_w:(y + 1) * small_w]
            edges = [abs(row[x + 1] - row[x - 1]) >= edge_min for x in range(1, small_w - 1)]
            hits = 0
            for i in range(len(edges) - 6):
                if edges[i] and any(edges[i + 1:i + 7]):
                    hits += 1
            if hits >= hits_needed:
                row_votes[y] += 1
    votes_needed = max(2, round(frame_count / 8))
    band_rows = [y for y in range(scan_top, small_h) if row_votes[y] >= votes_needed]
    if not band_rows:
        return None
    pad = 3
    y0 = max(scan_top, min(band_rows) - pad)
    y1 = min(small_h - 1, max(band_rows) + pad)
    band_h = y1 - y0 + 1
    # 过宽说明是误判（比如大片白衣服），宁可不动
    if band_h < 2 or band_h > small_h * 0.22:
        return None
    scale = height / small_h
    y = int(y0 * scale)
    h = max(8, int(band_h * scale))
    x = int(width * 0.02)
    w = int(width * 0.96)
    if y + h > height:
        h = height - y
    return (x, y, w, h)


def mask_burned_subtitles(
    video_path: str | Path,
    *,
    ffmpeg_configured: str = "",
    ffprobe_configured: str = "",
) -> Path:
    """对烧录字幕做 delogo 抹除，返回处理后的视频路径。

    有字幕只抹检测到的窄条，没字幕原样返回不重编码；失败时静默退回原视频，
    不阻断成片导出。产物写到原视频旁边的 *_desub.mp4，已存在且比原视频新则复用。
    """
    source = Path(video_path)
    try:
        width, height = probe_video_size(source, ffprobe_configured=ffprobe_configured)
        duration = probe_duration(source, ffprobe_configured=ffprobe_configured)
        if width <= 0 or height <= 0:
            return source
        # delogo 需要全功能 ffmpeg：remotion 内置版没有 rawvideo/delogo 滤镜，
        # 与 Mode 2 一样优先用项目 assets/ffmpeg-full，找不到再退回常规探测
        full_ffmpeg = Path(__file__).resolve().parent.parent / "assets" / "ffmpeg-full" / "ffmpeg.exe"
        ffmpeg = str(full_ffmpeg) if full_ffmpeg.is_file() else find_binary(ffmpeg_configured, FFMPEG_CANDIDATES)
        band = detect_subtitle_band(source, width, height, duration, ffmpeg)
        if band is None:
            return source
        masked_path = source.with_name(source.stem + "_desub.mp4")
        if masked_path.is_file() and masked_path.stat().st_size > 0 and masked_path.stat().st_mtime >= source.stat().st_mtime:
            return masked_path
        x, y, w, h = band
        command = [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", str(source),
            "-vf", f"delogo=x={x}:y={y}:w={w}:h={h}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            str(masked_path),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,
        )
        if result.returncode != 0 or not masked_path.exists() or masked_path.stat().st_size <= 0:
            return source
        return masked_path
    except Exception:  # noqa: BLE001
        return source
