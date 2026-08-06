from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .models import TtsSegment


TIMESTAMP_RE = re.compile(
    r"(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})(?P<ms>[,.]\d{1,3})?"
)


def parse_srt(content: str, *, language: str = "en") -> list[TtsSegment]:
    content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not content:
        return []
    blocks = re.split(r"\n{2,}", content)
    segments: list[TtsSegment] = []
    for block in blocks:
        lines = [line.strip("\ufeff ") for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if "-->" in lines[0]:
            index_text = str(len(segments) + 1)
            timing = lines[0]
            text_lines = lines[1:]
        elif len(lines) >= 2 and "-->" in lines[1]:
            index_text = lines[0]
            timing = lines[1]
            text_lines = lines[2:]
        else:
            continue
        start_text, end_text = [part.strip() for part in timing.split("-->", 1)]
        start = parse_timestamp(start_text)
        end = parse_timestamp(end_text)
        text = "\n".join(text_lines).strip()
        if not text:
            continue
        safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", index_text).strip("_")
        segment_id = str(int(safe_id)) if safe_id.isdigit() else str(safe_id or len(segments) + 1)
        segments.append(
            TtsSegment(
                segment_id=segment_id,
                start=start,
                end=end,
                text=text,
                speaker="说话人1",
                language=language,
            )
        )
    return segments


def parse_timestamp(value: str) -> float:
    match = TIMESTAMP_RE.search(value.strip())
    if not match:
        raise ValueError(f"invalid_srt_timestamp: {value}")
    ms_text = (match.group("ms") or ".0").replace(",", ".")
    milliseconds = int(float(ms_text) * 1000)
    return (
        int(match.group("h")) * 3600
        + int(match.group("m")) * 60
        + int(match.group("s"))
        + milliseconds / 1000
    )


def format_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    sec, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{sec:02d},{ms:03d}"


def segments_to_srt(segments: list[dict[str, Any]] | list[TtsSegment]) -> str:
    blocks: list[str] = []
    for index, raw in enumerate(segments, start=1):
        item = raw.to_dict() if isinstance(raw, TtsSegment) else raw
        start = format_timestamp(float(item.get("start") or 0))
        end = format_timestamp(float(item.get("end") or 0))
        text = str(item.get("text") or item.get("translated_text") or item.get("tts_text") or "").strip()
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def save_srt(path: str | Path, segments: list[dict[str, Any]] | list[TtsSegment]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(segments_to_srt(segments), encoding="utf-8")
    return target
