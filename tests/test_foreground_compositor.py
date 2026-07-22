from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from spvideo.foreground_compositor import (
    build_focus_video_from_color_mask_video,
    composite_generated_foreground,
)


class ForegroundCompositorTests(unittest.TestCase):
    def test_only_tracked_region_is_taken_from_generated_video(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = root / "base.mp4"
            generated = root / "generated.mp4"
            output = root / "output.mp4"
            track = root / "track"
            track.mkdir()
            self._write_video(base, 0)
            self._write_video(generated, 255)
            for index in range(4):
                mask = np.zeros((64, 64), dtype=np.uint8)
                mask[:, :32] = 255
                cv2.imwrite(str(track / f"mask_{index:04d}.png"), mask)
            (track / "track_summary.json").write_text(
                json.dumps({"track_fps": 4, "clip_start_time": 0}),
                encoding="utf-8",
            )

            result = composite_generated_foreground(
                base_video=base,
                generated_video=generated,
                track_dir=track,
                output_path=output,
                feather_pixels=1,
                dilate_pixels=0,
            )

            capture = cv2.VideoCapture(str(output))
            ok, frame = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertGreater(float(frame[:, :24].mean()), 220)
            self.assertLess(float(frame[:, 40:].mean()), 25)
            self.assertEqual(result["composited_frames"], 4)

    def test_color_mask_focus_extracts_one_palette_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = root / "base.mp4"
            mask = root / "mask.mp4"
            output = root / "output.mp4"
            self._write_video(base, 120)
            self._write_color_mask_video(mask)

            result = build_focus_video_from_color_mask_video(
                base_video=base,
                mask_video=mask,
                output_path=output,
                color="red",
                feather_pixels=1,
                dilate_pixels=0,
                background_dim=0.5,
            )

            capture = cv2.VideoCapture(str(output))
            ok, frame = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertGreater(float(frame[:, 42:58].mean()), 90)
            self.assertLess(float(frame[:, :20].mean()), 90)
            self.assertEqual(result["focused_frames"], 4)

    @staticmethod
    def _write_video(path: Path, value: int) -> None:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (64, 64))
        try:
            for _ in range(4):
                writer.write(np.full((64, 64, 3), value, dtype=np.uint8))
        finally:
            writer.release()

    @staticmethod
    def _write_color_mask_video(path: Path) -> None:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (64, 64))
        try:
            frame = np.full((64, 64, 3), 255, dtype=np.uint8)
            frame[:, :20] = (255, 0, 0)
            frame[:, 24:40] = (0, 255, 0)
            frame[:, 44:64] = (0, 0, 255)
            for _ in range(4):
                writer.write(frame)
        finally:
            writer.release()


if __name__ == "__main__":
    unittest.main()
