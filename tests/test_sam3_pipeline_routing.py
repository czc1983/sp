from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spvideo.models import VideoMeta
from spvideo.pipeline import run_segmentation_v2


class Sam3PipelineRoutingTests(unittest.TestCase):
    @patch("spvideo.pipeline.probe_video")
    @patch("spvideo.pipeline.sample_frames", return_value=[])
    @patch("spvideo.scene_detector.two_pass_segmentation")
    @patch("spvideo.sam3_finalizer.finalize_segments_with_sam3")
    def test_sam3_alone_still_runs_yolo_two_pass(
        self,
        finalize,
        two_pass,
        _frames,
        probe,
    ) -> None:
        probe.return_value = VideoMeta(
            source_path="source.mp4",
            duration=3.0,
            width=1080,
            height=1920,
            fps=30.0,
        )
        source_segment = {
            "start": 0.0,
            "end": 3.0,
            "person_count": 1,
            "is_pure_background": False,
            "main_person_bbox": [100, 200, 900, 1800],
            "start_sources": [],
            "end_sources": [],
        }
        two_pass.return_value = {
            "sub_segments": [source_segment],
            "stats": {
                "total_hard_cuts": 1,
                "total_sub_segments": 1,
                "pure_background_count": 0,
                "person_segment_count": 1,
            },
        }
        finalize.return_value = {
            "segments": [source_segment],
            "result": {"enabled": True, "skipped": False, "changed": False},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_segmentation_v2(
                "source.mp4",
                Path(temp_dir),
                sample_interval=0.1,
                export_video=False,
                use_scene_detect=False,
                use_omnishotcut=False,
                use_face_id=False,
                use_visual_model=False,
                use_sam3_finalize=True,
            )

        self.assertEqual(result["sam3_finalize"]["enabled"], True)
        self.assertEqual(_frames.call_args.args[3], 0.1)
        two_pass.assert_called_once()
        self.assertEqual(two_pass.call_args.kwargs["sample_interval"], 0.1)
        self.assertFalse(two_pass.call_args.kwargs["use_pyscene_detect"])


if __name__ == "__main__":
    unittest.main()
