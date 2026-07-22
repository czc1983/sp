from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from spvideo.models import VideoMeta
from spvideo.scene_detector import (
    _filter_nested_person_artifacts,
    _is_edge_partial_person,
    _shot_sample_times,
    subdivide_shot_by_person_presence,
    two_pass_segmentation,
)
from spvideo.visual_merge import merge_visually_similar_segments


class EdgePartialPersonTests(unittest.TestCase):
    def test_real_edge_partial_shape_passes_lower_confidence_threshold(self) -> None:
        self.assertTrue(
            _is_edge_partial_person(
                458,
                190,
                543,
                950,
                confidence=0.317,
                image_width=544,
                image_height=960,
            )
        )

    def test_very_low_confidence_edge_shape_is_rejected(self) -> None:
        self.assertFalse(
            _is_edge_partial_person(
                458,
                190,
                543,
                950,
                confidence=0.156,
                image_width=544,
                image_height=960,
            )
        )

    def test_nested_filter_keeps_valid_edge_partial_person(self) -> None:
        persons = [
            {
                "bbox": [20, 40, 500, 940],
                "confidence": 0.91,
                "area_ratio": 0.83,
                "edge_partial": False,
            },
            {
                "bbox": [458, 190, 543, 950],
                "confidence": 0.317,
                "area_ratio": 0.124,
                "edge_partial": True,
            },
        ]

        self.assertEqual(len(_filter_nested_person_artifacts(persons)), 2)


class FrameLevelSamplingTests(unittest.TestCase):
    def test_short_shot_is_scanned_at_source_frame_cadence(self) -> None:
        times = _shot_sample_times(
            0.0,
            1.31,
            sample_interval=1.0,
            fps=26.0,
        )

        self.assertIn(0.154, times)
        self.assertGreaterEqual(len(times), 30)

    def test_single_frame_multi_person_event_becomes_its_own_segment(self) -> None:
        detections = [
            self._detection(0.115, 1),
            self._detection(0.154, 2),
            self._detection(0.192, 1),
        ]

        segments = subdivide_shot_by_person_presence(
            0.0,
            1.0,
            detections,
            min_sub_duration=1.0,
            start_sources=["omnishotcut"],
            end_sources=["omnishotcut"],
        )

        self.assertEqual([segment["person_count"] for segment in segments], [1, 2, 1])
        self.assertTrue(segments[1]["transient_multi_person"])
        self.assertIn("yolo_transient_multi", segments[0]["end_sources"])
        self.assertIn("yolo_transient_multi", segments[1]["start_sources"])
        self.assertLess(segments[1]["end"] - segments[1]["start"], 0.05)

    @patch("spvideo.scene_detector.detect_persons_in_frame")
    @patch("spvideo.scene_detector._get_yolo_model")
    @patch("spvideo.omnishotcut_detector.detect_omnishotcut_shots")
    @patch("spvideo.ffmpeg_tools.extract_frame")
    @patch("spvideo.ffmpeg_tools.probe_video")
    def test_omni_only_mode_still_runs_yolo(
        self,
        probe_video,
        _extract_frame,
        detect_omni,
        _get_model,
        detect_persons,
    ) -> None:
        probe_video.return_value = VideoMeta(
            source_path="source.mp4",
            duration=0.2,
            width=544,
            height=960,
            fps=25.0,
        )
        detect_omni.return_value = {
            "shots": [{"index": 0, "start": 0.0, "end": 0.2}],
        }
        detect_persons.return_value = [{
            "bbox": [20, 20, 400, 940],
            "confidence": 0.9,
            "area_ratio": 0.7,
            "edge_partial": False,
        }]

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "source.mp4"
            video_path.touch()
            result = two_pass_segmentation(
                video_path,
                use_omnishotcut=True,
                use_pyscene_detect=False,
            )

        self.assertTrue(result["stats"]["yolo_enabled"])
        self.assertFalse(result["stats"]["pyscene_enabled"])
        self.assertEqual(result["sub_segments"][0]["person_count"], 1)

    @staticmethod
    def _detection(time: float, person_count: int) -> dict:
        persons = [
            {
                "bbox": [10 + index * 20, 10, 100 + index * 20, 300],
                "confidence": 0.9 if index == 0 else 0.317,
                "area_ratio": 0.4 if index == 0 else 0.124,
                "edge_partial": index > 0,
            }
            for index in range(person_count)
        ]
        return {
            "frame_path": f"frame_{time:.3f}.jpg",
            "person_count": person_count,
            "persons": persons,
            "time": time,
            "diff_prev": 0.0,
        }


class ProtectedVisualMergeTests(unittest.TestCase):
    @patch("spvideo.visual_merge._segment_transition_stats")
    @patch("spvideo.visual_merge._boundary_diff_score", return_value=0.01)
    def test_transient_yolo_boundary_is_not_merged(self, _score, transition_stats) -> None:
        transition_stats.return_value = {"transition_like": True}
        segments = [
            {
                "start": 0.0,
                "end": 0.135,
                "person_count": 1,
                "start_sources": ["omnishotcut"],
                "end_sources": ["yolo", "yolo_transient_multi"],
            },
            {
                "start": 0.135,
                "end": 0.173,
                "person_count": 2,
                "transient_multi_person": True,
                "start_sources": ["yolo", "yolo_transient_multi"],
                "end_sources": ["yolo", "yolo_transient_multi"],
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            result = merge_visually_similar_segments(
                "source.mp4",
                segments,
                temp_dir,
                fps=26.0,
            )

        self.assertEqual(len(result["segments"]), 2)
        self.assertEqual(result["result"]["decisions"][0]["reason"], "protected_boundary")


if __name__ == "__main__":
    unittest.main()
