from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .models import FrameFeatures


def analyze_frame(path: str | Path, time_seconds: float, previous_pixels: np.ndarray | None = None) -> tuple[FrameFeatures, np.ndarray]:
    image = Image.open(path).convert("RGB")
    image.thumbnail((160, 284))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]

    diff_prev = 0.0
    if previous_pixels is not None and previous_pixels.shape == pixels.shape:
        diff_prev = float(np.mean(np.abs(pixels - previous_pixels)))

    max_channel = np.max(pixels, axis=2)
    min_channel = np.min(pixels, axis=2)
    blue_mask = (blue > 0.35) & (blue > red * 1.25) & (blue > green * 1.05)
    white_mask = (red > 0.78) & (green > 0.78) & (blue > 0.78)
    dark_mask = max_channel < 0.14
    skin_mask = (
        (red > 0.35)
        & (green > 0.16)
        & (blue > 0.08)
        & (red > green * 1.06)
        & (red > blue * 1.18)
        & ((max_channel - min_channel) > 0.06)
    )

    height, width = skin_mask.shape
    center = skin_mask[int(height * 0.14) : int(height * 0.74), int(width * 0.22) : int(width * 0.78)]

    gray = (red * 0.299 + green * 0.587 + blue * 0.114)
    gx = np.abs(gray[:, 1:] - gray[:, :-1]).mean()
    gy = np.abs(gray[1:, :] - gray[:-1, :]).mean()

    features = FrameFeatures(
        time=time_seconds,
        path=str(Path(path)),
        diff_prev=diff_prev,
        blue_ratio=float(blue_mask.mean()),
        skin_ratio=float(skin_mask.mean()),
        center_skin_ratio=float(center.mean()) if center.size else 0.0,
        white_ratio=float(white_mask.mean()),
        dark_ratio=float(dark_mask.mean()),
        edge_score=float(gx + gy),
    )
    return features, pixels


def summarize_features(features: list[FrameFeatures]) -> dict[str, float]:
    if not features:
        return {
            "diff_prev": 0.0,
            "blue_ratio": 0.0,
            "skin_ratio": 0.0,
            "center_skin_ratio": 0.0,
            "white_ratio": 0.0,
            "dark_ratio": 0.0,
            "edge_score": 0.0,
        }
    keys = ("diff_prev", "blue_ratio", "skin_ratio", "center_skin_ratio", "white_ratio", "dark_ratio", "edge_score")
    return {key: float(np.mean([getattr(item, key) for item in features])) for key in keys}

