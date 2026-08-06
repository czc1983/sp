from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class TtsSegment:
    segment_id: str
    start: float
    end: float
    text: str
    speaker: str = "说话人1"
    language: str = "en"
    role: str = "dialogue"
    emotion: str = ""
    emotion_intensity: str = ""  # low / medium / high，空串按 medium 处理

    @property
    def target_seconds(self) -> float:
        return max(0.0, float(self.end) - float(self.start))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["target_seconds"] = round(self.target_seconds, 3)
        return data


def project_relative(path: str | Path, root: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")
