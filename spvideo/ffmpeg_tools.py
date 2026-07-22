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
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise FfmpegError(result.stderr.strip() or result.stdout.strip() or f"Command failed: {args}")
    return result


def _concat_file_line(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/").replace("'", r"'\''")
    return f"file '{value}'\n"


def concat_videos(
    segment_paths: list[str | Path],
    output_path: str | Path,
    *,
    reencode_fallback: bool = True,
) -> Path:
    """Concatenate videos in order with FFmpeg concat demuxer.

    The first pass uses stream copy for speed and exact frames. If copy fails
    because segments have mismatched container/codec metadata, the optional
    fallback re-encodes to a stable H.264 MP4.
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
            raise FfmpegError(f"concat_videos_output_missing: {output}")
        return output
    finally:
        list_path.unlink(missing_ok=True)


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


def cut_segment(video_path: str | Path, start: float, end: float, output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.01, end - start)
    args = [
        ffmpeg_path(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
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
