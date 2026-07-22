import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from spvideo.mask_preview import (
    render_mask_comparison_video,
    render_multi_mask_comparison_video,
)


class MaskPreviewTests(unittest.TestCase):
    def test_renders_side_by_side_mask_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            track = root / "track"
            output = root / "preview.mp4"
            track.mkdir()

            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48),
            )
            for _ in range(3):
                writer.write(np.full((48, 64, 3), 80, dtype=np.uint8))
            writer.release()

            (track / "track_summary.json").write_text(
                json.dumps({"track_fps": 10.0, "clip_start_time": 0.0}),
                encoding="utf-8",
            )
            for index in range(3):
                mask = np.zeros((48, 64), dtype=np.uint8)
                mask[8:40, 20:44] = 255
                cv2.imwrite(str(track / f"mask_{index:04d}.png"), mask)

            result = render_mask_comparison_video(source, track, output)

            self.assertTrue(output.exists())
            self.assertEqual(result["total_frames"], 3)
            self.assertEqual(result["masked_frames"], 3)
            self.assertEqual(result["width"], 128)
            self.assertAlmostEqual(result["mean_area_ratio"], 0.25, places=2)

    def test_renders_multiple_masks_and_reports_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            output = root / "multi_preview.mp4"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48),
            )
            for _ in range(3):
                writer.write(np.full((48, 64, 3), 80, dtype=np.uint8))
            writer.release()

            tracks = []
            for role_index, (left, right) in enumerate(((8, 36), (28, 56))):
                track = root / f"track_{role_index}"
                track.mkdir()
                (track / "track_summary.json").write_text(
                    json.dumps({"track_fps": 10.0, "clip_start_time": 0.0}),
                    encoding="utf-8",
                )
                for frame_index in range(3):
                    mask = np.zeros((48, 64), dtype=np.uint8)
                    mask[8:40, left:right] = 255
                    cv2.imwrite(str(track / f"mask_{frame_index:04d}.png"), mask)
                tracks.append({"name": f"role_{role_index}", "track_dir": str(track)})

            result = render_multi_mask_comparison_video(source, tracks, output)

            self.assertTrue(output.exists())
            self.assertEqual(result["total_frames"], 3)
            self.assertEqual(result["overlap_frames"], 3)
            self.assertEqual(len(result["roles"]), 2)
            self.assertTrue(all(item["masked_frames"] == 3 for item in result["roles"]))


if __name__ == "__main__":
    unittest.main()
