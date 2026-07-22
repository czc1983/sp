from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from web_ui.server import (
    _role_track_prompt_candidates,
    _try_reuse_same_role_ready_track,
)


class RoleTrackCandidateTests(unittest.TestCase):
    def test_director_prompt_stays_exact_without_synthetic_offsets(self) -> None:
        with patch("web_ui.server._clamp_video_time", side_effect=lambda _video, t: float(t)):
            candidates = _role_track_prompt_candidates(
                video_path="clip.mp4",
                role_name="养母",
                annotation={
                    "id": "ann_late",
                    "time": 3.3,
                    "point": [0.18, 0.41],
                },
                annotations=[
                    {
                        "id": "ann_early",
                        "type": "person",
                        "label_name": "养母",
                        "video_path": "clip.mp4",
                        "time": 1.45,
                        "point": [0.49, 0.50],
                    },
                    {
                        "id": "ann_late",
                        "type": "person",
                        "label_name": "养母",
                        "video_path": "clip.mp4",
                        "time": 3.3,
                        "point": [0.18, 0.41],
                    },
                ],
            )

        labels = [item["label"] for item in candidates]
        times = [item["time"] for item in candidates]
        self.assertFalse(any("ann_early" in label for label in labels))
        self.assertEqual(labels, ["导演身份点"])
        self.assertEqual(times, [3.3])

    def test_reuse_same_role_ready_track_mirrors_ready_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            track_dir = root / "ann_ready_sam3"
            track_dir.mkdir()
            (track_dir / "mask_0000.png").write_bytes(b"png")
            (track_dir / "track_summary.json").write_text(
                '{"prompt_source":"manual_annotation_direct","prompt_mode":"points"}',
                encoding="utf-8",
            )
            pair: dict = {"name": "养母", "annotation_id": "ann_failed"}
            logs: list[str] = []

            with patch("web_ui.server.update_annotation") as update_annotation:
                reused = _try_reuse_same_role_ready_track(
                    project_dir=str(root),
                    video_path="clip.mp4",
                    role_name="养母",
                    annotation={
                        "id": "ann_failed",
                        "point": [0.49, 0.50],
                    },
                    annotation_id="ann_failed",
                    annotations=[
                        {
                            "id": "ann_ready",
                            "type": "person",
                            "label_name": "养母",
                            "video_path": "clip.mp4",
                            "point": [0.49, 0.50],
                            "track_status": "ready",
                            "track_dir": str(track_dir),
                            "track_summary": str(track_dir / "track_summary.json"),
                            "tracked_frames": 50,
                            "track_frames": 50,
                        }
                    ],
                    pair=pair,
                    add_log=logs.append,
                )

            self.assertEqual(reused, (0.49, 0.5))
            self.assertEqual(pair["track_dir"], str(track_dir))
            update_annotation.assert_called_once()
            self.assertTrue(any("复用同角色已有轨迹" in message for message in logs))


if __name__ == "__main__":
    unittest.main()
