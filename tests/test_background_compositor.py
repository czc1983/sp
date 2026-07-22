from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from spvideo.asset_store import get_background_config, load_asset_store, upsert_background_config
from spvideo.background_compositor import compose_background


class BackgroundStoreTests(unittest.TestCase):
    def test_background_config_is_disabled_by_default_and_saved_per_segment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            asset = root / "background.png"
            cv2.imwrite(str(asset), np.zeros((8, 8, 3), dtype=np.uint8))

            self.assertEqual(load_asset_store(root)["backgrounds"], {})
            saved = upsert_background_config(
                root,
                segment_id="003",
                video_path=str(root / "clip.mp4"),
                mode="replace_static_image",
                asset_path=str(asset),
            )

            self.assertEqual(saved["status"], "ready")
            self.assertEqual(get_background_config(root, segment_id="003")["asset_path"], str(asset))


class BackgroundCompositorTests(unittest.TestCase):
    def test_foreground_is_preserved_and_background_is_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            foreground = root / "foreground.mp4"
            background = root / "background.png"
            output = root / "output.mp4"
            track_dir = root / "track"
            track_dir.mkdir()

            writer = cv2.VideoWriter(
                str(foreground),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (32, 24),
            )
            for _ in range(4):
                writer.write(np.full((24, 32, 3), (0, 0, 240), dtype=np.uint8))
            writer.release()
            cv2.imwrite(str(background), np.full((24, 32, 3), (240, 0, 0), dtype=np.uint8))

            summary = {"clip_start_time": 0.0, "track_fps": 10.0}
            (track_dir / "track_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            for index in range(4):
                mask = np.zeros((24, 32), dtype=np.uint8)
                mask[:, :16] = 255
                cv2.imwrite(str(track_dir / f"mask_{index:06d}.png"), mask)

            result = compose_background(
                foreground,
                background,
                [track_dir],
                output,
                feather_pixels=1,
                dilate_pixels=0,
                preserve_audio=False,
            )

            self.assertEqual(result["composited_frames"], 4)
            capture = cv2.VideoCapture(str(output))
            ok, frame = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertGreater(int(frame[12, 6, 2]), 180)
            self.assertGreater(int(frame[12, 26, 0]), 180)


if __name__ == "__main__":
    unittest.main()
