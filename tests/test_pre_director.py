from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spvideo.models import FrameFeatures
from spvideo.pre_director import (
    _boundary_hints,
    _collect_characters,
    _merge_transcript_chunks,
    _select_director_frames,
    _audio_transcription_model,
    _enrich_scene_roles_from_transcript,
    analyze_pre_director,
    apply_pre_director_boundaries,
    collapse_transient_person_jitter,
    semantic_scene_for_time,
)


class PreDirectorBoundaryTests(unittest.TestCase):
    def test_high_confidence_semantic_boundary_splits_long_segment(self) -> None:
        segments = [{"start": 0.0, "end": 4.0, "start_sources": [], "end_sources": []}]
        result = apply_pre_director_boundaries(
            segments,
            [{"time": 2.0, "confidence": 0.9, "kind": "role_change"}],
            min_segment_duration=1.0,
        )

        self.assertEqual([(item["start"], item["end"]) for item in result], [(0.0, 2.0), (2.0, 4.0)])
        self.assertIn("pre_director", result[0]["end_sources"])
        self.assertIn("pre_director", result[1]["start_sources"])

    def test_semantic_hint_snaps_to_nearby_technical_boundary(self) -> None:
        segments = [
            {"start": 0.0, "end": 2.0, "start_sources": [], "end_sources": ["pyscene"]},
            {"start": 2.0, "end": 4.0, "start_sources": ["pyscene"], "end_sources": []},
        ]
        result = apply_pre_director_boundaries(
            segments,
            [{"time": 2.2, "confidence": 0.8, "kind": "action_change"}],
            min_segment_duration=1.0,
        )

        self.assertEqual(len(result), 2)
        self.assertIn("pre_director", result[0]["end_sources"])
        self.assertIn("pre_director", result[1]["start_sources"])

    def test_low_confidence_hint_is_ignored(self) -> None:
        result = apply_pre_director_boundaries(
            [{"start": 0.0, "end": 4.0}],
            [{"time": 2.0, "confidence": 0.4}],
            min_segment_duration=1.0,
        )
        self.assertEqual(len(result), 1)

    def test_semantic_hint_prefers_real_shot_boundary_over_transient_yolo(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "end_sources": ["omnishotcut", "pyscene"]},
            {"start": 1.0, "end": 1.18, "start_sources": ["omnishotcut", "pyscene"], "end_sources": ["yolo", "yolo_transient_multi"]},
            {"start": 1.18, "end": 2.0, "start_sources": ["yolo", "yolo_transient_multi"]},
        ]
        result = apply_pre_director_boundaries(
            segments,
            [{"time": 1.16, "confidence": 0.9, "kind": "role_change"}],
            min_segment_duration=1.0,
        )
        self.assertIn("pre_director", result[0]["end_sources"])
        self.assertNotIn("pre_director", result[1]["end_sources"])

    def test_transient_multi_person_jitter_collapses_near_semantic_boundary(self) -> None:
        segments = [
            {"start": 1.0, "end": 1.15, "person_count": 1, "shot_index": 1, "start_sources": ["pyscene"], "end_sources": ["yolo", "yolo_transient_multi"]},
            {"start": 1.15, "end": 1.18, "person_count": 2, "shot_index": 1, "transient_multi_person": True, "start_sources": ["yolo", "yolo_transient_multi"], "end_sources": ["yolo", "yolo_transient_multi"]},
            {"start": 1.18, "end": 2.3, "person_count": 1, "shot_index": 1, "start_sources": ["yolo", "yolo_transient_multi"], "end_sources": ["pyscene"]},
        ]
        result = collapse_transient_person_jitter(
            segments,
            [{"time": 1.05, "confidence": 0.9, "kind": "role_change"}],
        )
        self.assertEqual(len(result), 1)
        self.assertEqual((result[0]["start"], result[0]["end"], result[0]["person_count"]), (1.0, 2.3, 1))
        self.assertIn("collapsed_person_jitter", result[0])

    def test_real_multi_person_segment_stays_without_semantic_support(self) -> None:
        segments = [
            {"start": 1.0, "end": 1.15, "person_count": 1, "shot_index": 1, "end_sources": ["yolo", "yolo_transient_multi"]},
            {"start": 1.15, "end": 1.18, "person_count": 2, "shot_index": 1, "transient_multi_person": True, "start_sources": ["yolo", "yolo_transient_multi"], "end_sources": ["yolo", "yolo_transient_multi"]},
            {"start": 1.18, "end": 2.3, "person_count": 1, "shot_index": 1, "start_sources": ["yolo", "yolo_transient_multi"]},
        ]
        self.assertEqual(collapse_transient_person_jitter(segments, []), segments)

    def test_semantic_scene_is_resolved_by_midpoint(self) -> None:
        scene = semantic_scene_for_time(
            {"scenes": [{"start": 1.0, "end": 2.0, "description": "role changes"}]},
            1.5,
        )
        self.assertEqual(scene["description"], "role changes")

    def test_boundary_hint_recovers_role_change_from_misplaced_video_end(self) -> None:
        hints = _boundary_hints(
            [
                {"start": 0.0, "end": 1.2, "boundary_kind": "role_change", "boundary_reason": "人物切换", "boundary_confidence": 1.0},
                {"start": 1.2, "end": 2.3, "boundary_kind": "video_end", "boundary_reason": "视频结束", "boundary_confidence": 1.0},
            ],
            2.3,
        )
        self.assertEqual(hints[0]["kind"], "role_change")
        self.assertEqual(hints[0]["reason"], "人物切换")

    def test_character_timeline_keeps_audio_derived_role_candidate(self) -> None:
        characters = _collect_characters([{
            "start": 0.0,
            "end": 1.0,
            "characters": ["蓝衣女人"],
            "character_details": [{
                "visual_label": "蓝衣女人",
                "role_name": "婆婆",
                "description": "中年女性",
                "confidence": 0.86,
            }],
        }])
        self.assertEqual(characters[0]["role_name"], "婆婆")
        self.assertEqual(characters[0]["time_ranges"], [[0.0, 1.0]])

    def test_audio_chunks_are_merged_on_global_timeline(self) -> None:
        transcript = _merge_transcript_chunks([
            {"language": "zh", "summary": "前段", "utterances": [{"start": 1.0, "end": 2.0, "text": "A"}]},
            {"language": "zh", "summary": "后段", "utterances": [{"start": 3.0, "end": 4.0, "text": "B"}]},
        ], chunk_seconds=1200)
        self.assertEqual(transcript["utterances"][0]["start"], 1.0)
        self.assertEqual(transcript["utterances"][1]["start"], 1203.0)
        self.assertEqual(transcript["summary"], "前段；后段")

    def test_long_video_frames_are_reduced_for_director_analysis(self) -> None:
        frames = [type("Frame", (), {"time": index / 10})() for index in range(3000)]
        selected = _select_director_frames(frames, 300.0)
        self.assertEqual(len(selected), 240)
        self.assertEqual(selected[0], frames[0])
        self.assertEqual(selected[-1], frames[-1])

    def test_short_video_keeps_dense_director_frames(self) -> None:
        frames = [type("Frame", (), {"time": index / 10})() for index in range(24)]
        self.assertEqual(len(_select_director_frames(frames, 2.4)), 24)

    def test_plus_uses_flash_for_audio_transcription(self) -> None:
        self.assertEqual(_audio_transcription_model("qwen3.5-omni-plus"), "qwen3.5-omni-flash")

    def test_only_audio_explicit_role_names_survive(self) -> None:
        scenes = [{
            "start": 0.0,
            "end": 1.0,
            "character_details": [{"visual_label": "蓝衣女人", "role_name": "美女"}],
        }]
        without_name = _enrich_scene_roles_from_transcript(scenes, {"utterances": []})
        self.assertEqual(without_name[0]["character_details"][0]["role_name"], "")
        with_name = _enrich_scene_roles_from_transcript(scenes, {"utterances": [{
            "start": 0.1, "end": 0.9, "role_name": "婆婆", "text": "婆婆"
        }]})
        self.assertEqual(with_name[0]["character_details"][0]["role_name"], "婆婆")


    def test_mode2_plan_is_written_from_one_visual_call(self) -> None:
        calls = []

        class FakeClient:
            def __init__(self, **_kwargs):
                pass

            def analyze_mode2_plan(self, **kwargs):
                calls.append(kwargs)
                return {
                    "story_summary": "一名女子进入卧室。",
                    "scenes": [{
                        "start": 0.0,
                        "end": 2.0,
                        "description": "女子站在卧室中",
                        "characters": ["女子"],
                        "character_details": [],
                        "key_action": "进入",
                    }],
                    "asset_manifest": {"results": [{
                        "group_id": "R001",
                        "kind": "role",
                        "name": "女子",
                        "identity": "person_01",
                        "matched_role": "女主",
                        "physical_scene": "bedroom_01",
                        "evidence_times": [1.0],
                        "representative_frame_index": 2,
                        "visible_props": [],
                        "confidence": 0.92,
                        "reason": "清晰正脸",
                    }]},
                }

            def analyze_video_url(self, **_kwargs):
                raise AssertionError("Mode2 must not call the legacy video URL method")

            def analyze_full_video(self, *_args, **_kwargs):
                raise AssertionError("Mode2 must not call the legacy full-video method")

            def close(self):
                pass

        def fake_upload(*_args, **_kwargs):
            return {"url": "https://example.invalid/video.mp4", "size": 10}

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "pre_director.json"
            with patch("spvideo.gemini_analyzer.GeminiClient", FakeClient), patch(
                "spvideo.yuanqi_upload.upload_file_for_url", fake_upload
            ):
                result = analyze_pre_director(
                    [
                        FrameFeatures(time=0.0, path="frame_0.jpg"),
                        FrameFeatures(time=1.0, path="frame_1.jpg"),
                    ],
                    video_path="original.mp4",
                    duration=2.0,
                    output_path=output,
                    api_key="test-key",
                    include_asset_manifest=True,
                )
            persisted = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["video_url"], "https://example.invalid/video.mp4")
        self.assertEqual(result["asset_manifest"]["results"][0]["group_id"], "R001")
        self.assertEqual(result["story_summary"], "一名女子进入卧室。")
        self.assertEqual(persisted["analysis_frame_manifest"][1], {
            "index": 2,
            "time": 1.0,
            "path": "frame_1.jpg",
        })
        self.assertEqual(persisted["asset_manifest"]["results"][0]["matched_role"], "女主")

    def test_pre_director_default_keeps_legacy_visual_call_and_shape(self) -> None:
        calls = []

        class FakeClient:
            def __init__(self, **_kwargs):
                pass

            def analyze_full_video(self, frames, **_kwargs):
                calls.append(len(frames))
                return [{"start": 0.0, "end": 1.0, "description": "旧接口"}]

            def analyze_mode2_plan(self, **_kwargs):
                raise AssertionError("Default pre-director must stay on the legacy API")

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "pre_director.json"
            with patch("spvideo.gemini_analyzer.GeminiClient", FakeClient):
                result = analyze_pre_director(
                    [FrameFeatures(time=0.0, path="frame_0.jpg")],
                    duration=1.0,
                    output_path=output,
                    api_key="test-key",
                )

        self.assertEqual(calls, [1])
        self.assertNotIn("asset_manifest", result)
        self.assertEqual(result["scenes"][0]["description"], "旧接口")


if __name__ == "__main__":
    unittest.main()
