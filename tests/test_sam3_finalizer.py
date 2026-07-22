from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from spvideo.sam3_finalizer import (
    _needs_sam3_tracking,
    _persistent_loss_index,
    _segment_prompt_point,
    finalize_segments_with_sam3,
)


class Sam3FinalizerTests(unittest.TestCase):
    def test_persistent_loss_is_cut_at_first_lost_frame(self) -> None:
        tracked = [True, True, True, True, False, False, False, False, False, False]

        self.assertEqual(_persistent_loss_index(tracked), 4)

    def test_short_occlusion_that_recovers_is_not_cut(self) -> None:
        tracked = [True, True, True, False, False, True, True, True, True, True]

        self.assertIsNone(_persistent_loss_index(tracked))

    def test_stable_single_person_segment_does_not_start_expensive_tracking(self) -> None:
        self.assertFalse(_needs_sam3_tracking({"person_count": 1}))
        self.assertTrue(_needs_sam3_tracking({"person_count": 1, "end_sources": ["yolo"]}))

    @patch("spvideo.sam3_finalizer._video_size", return_value=(1000, 2000))
    def test_prompt_uses_video_dimensions_when_yolo_temp_frame_is_gone(self, _video_size) -> None:
        prompt = _segment_prompt_point(
            {
                "main_person_bbox": [100, 400, 500, 1400],
                "_all_frame_detections": [{"frame_path": "deleted.jpg"}],
            },
            "source.mp4",
        )

        self.assertEqual(prompt, [0.3, 0.45])

    @patch("spvideo.sam3_finalizer._segment_prompt_point", return_value=[0.5, 0.5])
    @patch("spvideo.sam3_finalizer._make_track_clip", return_value=("track.mp4", 12, 8.0))
    @patch("spvideo.sam3_finalizer._create_tracker")
    @patch(
        "spvideo.sam3_finalizer.find_sam3_runtime",
        return_value={"available": True, "module": "sam3", "checked": ["sam3"]},
    )
    def test_finalizer_adds_boundary_when_subject_is_lost(
        self,
        _runtime,
        create_tracker,
        _make_clip,
        _prompt,
    ) -> None:
        tracker = create_tracker.return_value
        tracker.track_by_point.return_value = {
            "boxes": [object(), object(), object(), object(), object(), object(), None, None, None, None, None, None],
            "scores": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        segment = {
            "start": 0.0,
            "end": 4.0,
            "person_count": 1,
            "start_sources": ["pyscene", "yolo"],
            "end_sources": ["pyscene"],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            payload = finalize_segments_with_sam3("source.mp4", [segment], temp_dir)

        self.assertTrue(payload["result"]["changed"])
        self.assertEqual(len(payload["segments"]), 2)
        self.assertEqual(payload["segments"][0]["end"], 0.75)
        self.assertIn("sam3_track_lost", payload["segments"][0]["end_sources"])
        tracker.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
