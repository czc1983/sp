from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SEGMENT_TYPES = (
    "with_human",
    "without_human",
)


@dataclass
class VideoMeta:
    source_path: str
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str = ""
    audio_codec: str = ""
    audio_sample_rate: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrameFeatures:
    time: float
    path: str
    diff_prev: float = 0.0
    blue_ratio: float = 0.0
    skin_ratio: float = 0.0
    center_skin_ratio: float = 0.0
    white_ratio: float = 0.0
    dark_ratio: float = 0.0
    edge_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Segment:
    segment_id: str
    start: float
    end: float
    segment_type: str
    confidence: float
    output_path: str | None = None
    representative_frame: str | None = None
    detected: list[str] = field(default_factory=list)
    recommended_tech: str = ""
    needs_ai_driver: bool = False
    needs_manual_check: bool = False
    notes: list[str] = field(default_factory=list)
    # Gemini 扩展字段
    generation_route: str = ""
    scene_type: str = ""
    description: str = ""
    has_product: bool = False
    main_subject: str = ""
    # YOLO 检测扩展字段
    person_count: int = -1  # -1=未检测, 0=无人, 1=单人, 2+=多人
    transient_multi_person: bool = False
    # 边界来源追踪
    start_sources: list[str] = field(default_factory=list)
    end_sources: list[str] = field(default_factory=list)
    # 编辑操作:空=正常, deleted=已删除, merged=已合并(旧片段)
    edit_action: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration"] = self.duration
        return data


def safe_stem(value: str) -> str:
    keep = []
    for char in value:
        if char.isalnum() or char in ("-", "_"):
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "segment"


def ensure_dir(path: str | Path) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result
