from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .models import VideoMeta


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


class FfmpegError(RuntimeError):
    pass


def find_binary(candidates: list[str]) -> str:
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found
        path = Path(candidate)
        if path.exists():
            return str(path)
    raise FfmpegError(f"Cannot find binary from candidates: {candidates}")


def ffmpeg_path() -> str:
    return find_binary(FFMPEG_CANDIDATES)


def ffprobe_path() -> str:
    return find_binary(FFPROBE_CANDIDATES)


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **subprocess_no_window_kwargs(),
    )
    if result.returncode != 0:
        raise FfmpegError(result.stderr.strip() or result.stdout.strip() or f"Command failed: {args}")
    return result


def subprocess_no_window_kwargs() -> dict[str, object]:
    """Hide transient console windows for ffmpeg/ffprobe on Windows."""
    if os.name != "nt":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {
        "startupinfo": startupinfo,
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
    }


def _concat_file_line(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/").replace("'", r"'\''")
    return f"file '{value}'\n"


def concat_videos(
    segment_paths: list[str | Path],
    output_path: str | Path,
    *,
    reencode_fallback: bool = True,
    force_reencode: bool = False,
    expected_duration: float | None = None,
    duration_tolerance: float = 0.35,
) -> Path:
    """Concatenate videos in order with FFmpeg concat demuxer.

    The first pass uses stream copy for speed and exact frames. If copy fails,
    or a caller asks for safer timestamp normalization, the fallback re-encodes
    to a stable H.264 MP4.
    """
    if not segment_paths:
        raise FfmpegError("concat_videos_requires_segments")
    inputs = [Path(path) for path in segment_paths]
    missing = [str(path) for path in inputs if not path.exists() or not path.is_file()]
    if missing:
        raise FfmpegError(f"concat_videos_missing_input: {missing[0]}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_list_path = tempfile.mkstemp(prefix="concat_", suffix=".txt", dir=str(output.parent))
    os.close(fd)
    list_path = Path(raw_list_path)
    try:
        list_path.write_text("".join(_concat_file_line(path) for path in inputs), encoding="utf-8")

        def reencode_concat() -> None:
            output.unlink(missing_ok=True)
            reencode_args = [
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
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-movflags",
                "+faststart",
                str(output),
            ]
            run_command(reencode_args)

        if force_reencode:
            reencode_concat()
        else:
            copy_args = [
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
                str(output),
            ]
            try:
                run_command(copy_args)
            except FfmpegError:
                if not reencode_fallback:
                    raise
                reencode_concat()

        if (
            expected_duration is not None
            and expected_duration > 0
            and output.exists()
            and output.stat().st_size > 0
        ):
            actual_duration = float(probe_video(output).duration or 0.0)
            tolerance = max(0.05, float(duration_tolerance))
            if abs(actual_duration - expected_duration) > tolerance:
                if not force_reencode and reencode_fallback:
                    reencode_concat()
                    actual_duration = float(probe_video(output).duration or 0.0)
                if abs(actual_duration - expected_duration) > tolerance:
                    raise FfmpegError(
                        "concat_videos_duration_mismatch: "
                        f"expected={expected_duration:.3f}, actual={actual_duration:.3f}, output={output}"
                    )
        if not output.exists() or output.stat().st_size <= 0:
            raise FfmpegError(f"concat_videos_output_missing: {output}")
        return output
    finally:
        list_path.unlink(missing_ok=True)


def copy_audio_from_source(
    source_video: str | Path,
    generated_video: str | Path,
    output_path: str | Path,
) -> Path:
    """Copy the source audio track onto a generated video when the result is silent."""
    source = Path(source_video)
    generated = Path(generated_video)
    output = Path(output_path)
    if not source.exists() or not generated.exists():
        return generated

    source_meta = probe_video(source)
    generated_meta = probe_video(generated)
    if not source_meta.audio_codec or generated_meta.audio_codec:
        return generated

    output.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.01, float(generated_meta.duration or 0.0))
    copy_args = [
        ffmpeg_path(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(generated),
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(output),
    ]
    try:
        run_command(copy_args)
    except FfmpegError:
        output.unlink(missing_ok=True)
        reencode_args = [
            ffmpeg_path(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(generated),
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(output),
        ]
        run_command(reencode_args)

    if not output.exists() or output.stat().st_size <= 0:
        raise FfmpegError(f"copy_audio_output_missing: {output}")
    return output


def strip_audio_from_video(
    source_video: str | Path,
    output_path: str | Path,
) -> Path:
    """Create a video-only copy for generators that should not receive audio."""
    source = Path(source_video)
    output = Path(output_path)
    if not source.exists():
        raise FfmpegError(f"strip_audio_source_missing: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    copy_args = [
        ffmpeg_path(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
        "-an",
        "-movflags",
        "+faststart",
        str(output),
    ]
    try:
        run_command(copy_args)
    except FfmpegError:
        output.unlink(missing_ok=True)
        reencode_args = [
            ffmpeg_path(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
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
            "-an",
            "-movflags",
            "+faststart",
            str(output),
        ]
        run_command(reencode_args)

    if not output.exists() or output.stat().st_size <= 0:
        raise FfmpegError(f"strip_audio_output_missing: {output}")
    return output


def probe_video(video_path: str | Path) -> VideoMeta:
    args = [
        ffprobe_path(),
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,sample_rate",
        "-of",
        "json",
        str(video_path),
    ]
    result = run_command(args)
    data = json.loads(result.stdout)
    duration = float(data.get("format", {}).get("duration", 0.0))

    video_stream = {}
    audio_stream = {}
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and not video_stream:
            video_stream = stream
        if stream.get("codec_type") == "audio" and not audio_stream:
            audio_stream = stream

    fps = _parse_fraction(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/1")
    return VideoMeta(
        source_path=str(Path(video_path)),
        duration=duration,
        width=int(video_stream.get("width", 0) or 0),
        height=int(video_stream.get("height", 0) or 0),
        fps=fps,
        video_codec=str(video_stream.get("codec_name", "")),
        audio_codec=str(audio_stream.get("codec_name", "")),
        audio_sample_rate=_optional_int(audio_stream.get("sample_rate")),
    )


def extract_frame(video_path: str | Path, time_seconds: float, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    args = [
        ffmpeg_path(),
        "-y",
        "-loglevel", "error",
        "-ss", f"{time_seconds:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-c:v", "mjpeg",
        "-strict", "unofficial",
        "-q:v", "2",
        "-update", "1",
        str(output),
    ]
    run_command(args)
    if not output.exists() or output.stat().st_size <= 0:
        raise FfmpegError(f"extract_frame_output_missing: {output}")


def extract_frames_bulk(
    video_path: str | Path,
    output_pattern: str | Path,
    *,
    frames_per_second: float,
) -> None:
    """Decode regularly sampled JPEG frames in one FFmpeg process."""
    Path(output_pattern).parent.mkdir(parents=True, exist_ok=True)
    args = [
        ffmpeg_path(),
        "-y",
        "-loglevel", "error",
        "-i", str(video_path),
        "-map", "0:v:0",
        "-an",
        "-r", f"{max(0.01, frames_per_second):.6f}",
        "-q:v", "2",
        "-start_number", "1",
        str(output_pattern),
    ]
    run_command(args)


def extract_audio_for_analysis(video_path: str | Path, output_path: str | Path) -> None:
    """Create a compact mono MP3 suitable for one-time story transcription."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    args = [
        ffmpeg_path(),
        "-y",
        "-loglevel", "error",
        "-i", str(video_path),
        "-map", "0:a:0",
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "libmp3lame",
        "-b:a", "32k",
        str(output_path),
    ]
    run_command(args)


def extract_audio_chunks_for_analysis(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    chunk_seconds: int = 1200,
) -> list[Path]:
    """Extract 32kbps MP3 chunks that stay below DashScope's Base64 limit."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("pre_director_audio_*.mp3"):
        stale.unlink(missing_ok=True)
    chunk_seconds = max(60, int(chunk_seconds))
    duration = probe_video(video_path).duration
    chunk_count = max(1, int(math.ceil(duration / chunk_seconds)))
    outputs: list[Path] = []
    for index in range(chunk_count):
        start = index * chunk_seconds
        output = directory / f"pre_director_audio_{index:04d}.mp3"
        args = [
            ffmpeg_path(),
            "-y",
            "-loglevel", "error",
            "-ss", f"{start:.3f}",
            "-i", str(video_path),
            "-t", f"{min(chunk_seconds, max(0.01, duration - start)):.3f}",
            "-map", "0:a:0",
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "libmp3lame",
            "-b:a", "32k",
            str(output),
        ]
        run_command(args)
        outputs.append(output)
    return outputs


def create_pre_director_video_chunks(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    duration: float,
    chunk_seconds: float = 45.0,
    max_base64_bytes: int = 9_500_000,
) -> list[dict[str, object]]:
    """Create compact MP4 chunks for Qwen-Omni video understanding.

    DashScope accepts larger videos by public URL, but local Base64 video input
    must stay below 10 MB after encoding. These chunks are intentionally small:
    low FPS, low width, and compact audio, with recursive splitting if needed.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("pre_director_video_*.mp4"):
        stale.unlink(missing_ok=True)

    chunks: list[dict[str, object]] = []
    chunk_seconds = max(3.0, float(chunk_seconds))

    def encode_chunk(start: float, end: float, depth: int = 0) -> None:
        if end <= start:
            return
        output = directory / f"pre_director_video_{len(chunks):04d}_{start:.2f}_{end:.2f}.mp4"
        args = [
            ffmpeg_path(),
            "-y",
            "-loglevel", "error",
            "-ss", f"{start:.3f}",
            "-i", str(video_path),
            "-t", f"{max(0.05, end - start):.3f}",
            "-map", "0:v:0",
            "-map", "0:a:0?",
            "-vf", "fps=3,scale=360:-2",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "34",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-ac", "1",
            "-ar", "16000",
            "-b:a", "24k",
            "-movflags", "+faststart",
            str(output),
        ]
        run_command(args)
        base64_size = math.ceil(output.stat().st_size / 3) * 4
        if base64_size > max_base64_bytes and end - start > 3.0 and depth < 6:
            output.unlink(missing_ok=True)
            mid = (start + end) / 2.0
            encode_chunk(start, mid, depth + 1)
            encode_chunk(mid, end, depth + 1)
            return
        if base64_size > max_base64_bytes:
            output.unlink(missing_ok=True)
            raise FfmpegError(f"pre_director_chunk_too_large: {start:.2f}-{end:.2f}s")
        chunks.append({
            "path": output,
            "start": start,
            "end": end,
            "base64_size": base64_size,
        })

    start = 0.0
    total = max(0.0, float(duration))
    while start < total:
        end = min(total, start + chunk_seconds)
        encode_chunk(start, end)
        start = end
    return chunks


def _cut_segment_once(
    video_path: str | Path,
    start: float,
    duration: float,
    output_path: str | Path,
    *,
    input_seek: bool = False,
) -> None:
    seek_args = (
        ["-ss", f"{start:.6f}", "-i", str(video_path)]
        if input_seek
        else ["-i", str(video_path), "-ss", f"{start:.6f}"]
    )
    args = [
        ffmpeg_path(),
        "-y",
        "-loglevel",
        "error",
        *seek_args,
        "-t",
        f"{duration:.6f}",
        "-map",
        "0:v:0?",
        "-map",
        "0:a:0?",
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
    ]
    # 如果没音频流就去掉音频 map
    run_command(args)


def _has_video_stream(video_path: str | Path) -> bool:
    try:
        return int(probe_video(video_path).width or 0) > 0
    except Exception:  # noqa: BLE001
        return False


def cut_segment(video_path: str | Path, start: float, end: float, output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.01, end - start)
    _cut_segment_once(video_path, start, duration, output_path)
    if not _has_video_stream(video_path) or _has_video_stream(output_path):
        return
    # 超短窗口落在帧与帧之间时一帧都切不出来，视频流会被静默丢掉只留音频。
    # 纯音频片段混进整包合成会让 ffmpeg 的 -map 0:v:0 直接失败，
    # 所以对齐到帧边界、用输入端定位重切，保证至少留下一帧。
    aligned_start, aligned_end, _start_frame, _end_frame = align_segment_to_frames(video_path, start, end)
    _cut_segment_once(
        video_path,
        aligned_start,
        max(0.01, aligned_end - aligned_start),
        output_path,
        input_seek=True,
    )
    if not _has_video_stream(output_path):
        raise FfmpegError(
            f"cut_segment_produced_no_video: {output_path} window=({start:.3f}, {end:.3f})"
        )


def align_segment_to_frames(video_path: str | Path, start: float, end: float) -> tuple[float, float, int, int]:
    meta = probe_video(video_path)
    fps = float(meta.fps or 0.0)
    if fps <= 0.01:
        fps = 30.0
    source_duration = max(0.0, float(meta.duration or 0.0))
    safe_start = max(0.0, min(float(start), source_duration))
    safe_end = max(safe_start + (1.0 / fps), min(float(end), source_duration or float(end)))
    start_frame = max(0, int(round(safe_start * fps)))
    end_frame = max(start_frame + 1, int(round(safe_end * fps)))
    if source_duration > 0:
        max_frame = max(1, int(math.ceil(source_duration * fps)))
        end_frame = min(end_frame, max_frame)
    aligned_start = start_frame / fps
    aligned_end = end_frame / fps
    return aligned_start, aligned_end, start_frame, end_frame


def cut_segment_precise(
    video_path: str | Path,
    start: float,
    end: float,
    output_path: str | Path,
    *,
    include_audio: bool = True,
) -> None:
    """Cut a frame-aligned half-open segment [start_frame, end_frame)."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    aligned_start, aligned_end, _start_frame, _end_frame = align_segment_to_frames(video_path, start, end)
    duration = max(0.01, aligned_end - aligned_start)
    args = [
        ffmpeg_path(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-ss",
        f"{aligned_start:.6f}",
        "-t",
        f"{duration:.6f}",
        "-map",
        "0:v:0?",
    ]
    if include_audio:
        args.extend(["-map", "0:a:0?"])
    args.extend([
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
    ])
    if include_audio:
        args.extend(["-c:a", "aac", "-b:a", "128k"])
    else:
        args.append("-an")
    args.extend(["-movflags", "+faststart", str(output)])
    run_command(args)


def _parse_fraction(value: str) -> float:
    if not value:
        return 0.0
    if "/" not in value:
        return float(value)
    top, bottom = value.split("/", 1)
    bottom_value = float(bottom)
    if bottom_value == 0:
        return 0.0
    return float(top) / bottom_value


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
