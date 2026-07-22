from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spvideo.auto_director import (
    analyze_auto_director_project,
    answer_auto_director_question,
    compose_auto_director_plan,
    resolve_auto_director_project_root,
    save_auto_director,
)


class AutoDirectorPlanTests(unittest.TestCase):
    def test_child_clip_directory_resolves_to_manifest_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            child = root / "02_segments" / "00_all_mp4_clips"
            child.mkdir(parents=True)
            (root / "manifest.json").write_text(
                json.dumps({"segments": []}),
                encoding="utf-8",
            )

            self.assertEqual(resolve_auto_director_project_root(child), root)

    def test_legacy_two_pass_project_can_be_analyzed_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            probe_dir = root / "01_probe"
            child = root / "02_segments" / "00_all_mp4_clips"
            probe_dir.mkdir(parents=True)
            child.mkdir(parents=True)
            (probe_dir / "two_pass_result.json").write_text(
                json.dumps({
                    "sub_segments": [{
                        "segment_id": "001",
                        "start": 1.0,
                        "end": 2.5,
                        "person_count": 1,
                    }],
                }),
                encoding="utf-8",
            )

            plan = analyze_auto_director_project(child, scan_faces=False)

            self.assertEqual(Path(plan["project_dir"]), root)
            self.assertTrue(any(
                question.get("segment_ids") == ["001"]
                for question in plan["questions"]
            ))
            self.assertTrue((root / "assets" / "auto_director.json").exists())

    def test_story_entities_and_object_clues_create_focused_questions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            segments = [
                {"segment_id": "001", "segment_type": "with_human", "person_count": 1},
                {"segment_id": "002", "segment_type": "with_human", "person_count": 1},
                {"segment_id": "003", "segment_type": "with_human", "person_count": 2},
            ]
            plan = compose_auto_director_plan(
                project_dir=root,
                manifest={"meta": {"source_path": "source.mp4"}, "segments": segments},
                assets={
                    "characters": {
                        "lead": {"name": "女主"},
                        "mother": {"name": "婆婆"},
                    },
                    "directors": {},
                },
                face_entities=[{
                    "id": "face_01",
                    "visual_label": "蓝衣女人",
                    "description": "在多个镜头中出现",
                    "segment_ids": ["001", "002"],
                    "preview_frames": [],
                    "suggested_role": "女主",
                    "suggestion_confidence": 0.81,
                }],
                story={
                    "status": "ready",
                    "summary": "蓝衣女人拿起手机后与另一名女人发生冲突。",
                    "characters": [],
                    "important_clues": [{
                        "id": "phone",
                        "kind": "object",
                        "description": "手机",
                        "segment_ids": ["002"],
                        "why_important": "可能承载剧情证据",
                        "needs_confirmation": True,
                    }],
                },
            )

            kinds = {item["kind"] for item in plan["questions"]}
            self.assertIn("person_identity", kinds)
            self.assertIn("multi_role_mapping", kinds)
            self.assertIn("object_policy", kinds)
            self.assertEqual(plan["story"]["summary"], "蓝衣女人拿起手机后与另一名女人发生冲突。")

            person_question = next(item for item in plan["questions"] if item["kind"] == "person_identity")
            save_auto_director(root, plan)
            answered = answer_auto_director_question(
                root,
                question_id=person_question["id"],
                answer="女主",
            )

            self.assertEqual(answered["entities"][0]["resolved_as"], "女主")
            self.assertEqual(answered["segment_decisions"]["001"]["suggested_roles"], ["女主"])
            self.assertEqual(answered["segment_decisions"]["002"]["suggested_roles"], ["女主"])
            self.assertEqual(answered["stats"]["answered_count"], 1)

    def test_unresolved_frame_groups_accept_multiple_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            segments = [
                {"segment_id": "001", "segment_type": "with_human", "person_count": 1},
                {"segment_id": "002", "segment_type": "with_human", "person_count": 1},
            ]
            plan = compose_auto_director_plan(
                project_dir=root,
                manifest={"segments": segments},
                assets={
                    "characters": {
                        "lead": {"name": "女主"},
                        "mother": {"name": "婆婆"},
                    },
                    "directors": {},
                },
                face_entities=[],
                story={},
            )
            question = next(
                item for item in plan["questions"]
                if item["kind"] == "segment_roles" and item["segment_ids"] == ["001"]
            )
            save_auto_director(root, plan)
            answered = answer_auto_director_question(
                root,
                question_id=question["id"],
                answer=["女主", "婆婆"],
            )
            self.assertEqual(
                answered["segment_decisions"]["001"]["suggested_roles"],
                ["女主", "婆婆"],
            )

    def test_no_human_or_unknown_segments_do_not_create_role_questions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = compose_auto_director_plan(
                project_dir=Path(temp_dir),
                manifest={"segments": [
                    {"segment_id": "001", "segment_type": "without_human", "person_count": 0},
                    {"segment_id": "002", "segment_type": "with_human", "person_count": -1},
                ]},
                assets={"characters": {}, "directors": {}},
                face_entities=[],
                story={},
            )

            self.assertFalse(any(
                question["kind"] == "segment_roles"
                for question in plan["questions"]
            ))

    def test_no_person_answer_becomes_segment_policy_not_role(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = compose_auto_director_plan(
                project_dir=root,
                manifest={"segments": [
                    {"segment_id": "001", "segment_type": "with_human", "person_count": 1},
                ]},
                assets={"characters": {}, "directors": {}},
                face_entities=[],
                story={},
            )
            question = next(
                item for item in plan["questions"]
                if item["kind"] == "segment_roles"
            )
            save_auto_director(root, plan)

            answered = answer_auto_director_question(
                root,
                question_id=question["id"],
                answer=["no_person"],
            )

            decision = answered["segment_decisions"]["001"]
            self.assertEqual(decision["policy"], "no_person")
            self.assertEqual(decision["suggested_roles"], [])

    def test_pre_director_story_is_reused_without_second_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            probe = root / "01_分析探针"
            probe.mkdir(parents=True)
            (root / "manifest.json").write_text(json.dumps({
                "meta": {"source_path": "source.mp4"},
                "segments": [
                    {"segment_id": "001", "start": 0.0, "end": 1.0, "segment_type": "with_human", "person_count": 1},
                    {"segment_id": "002", "start": 1.0, "end": 2.0, "segment_type": "with_human", "person_count": 1},
                ],
            }), encoding="utf-8")
            (probe / "pre_director.json").write_text(json.dumps({
                "status": "ready",
                "story_summary": "婆婆责问女主。",
                "audio_understanding": True,
                "audio_status": "ready",
                "transcript": {"utterances": [{"start": 0, "end": 1, "text": "你这个儿媳"}]},
                "characters": [{
                    "visual_label": "蓝衣女人",
                    "role_name": "婆婆",
                    "description": "中年女性，蓝色连衣裙",
                    "confidence": 0.88,
                    "time_ranges": [[0.0, 1.0]],
                }, {
                    "visual_label": "白衣年轻女人",
                    "role_name": "女主",
                    "description": "年轻女性，白衣",
                    "confidence": 0.84,
                    "time_ranges": [[1.0, 2.0]],
                }],
                "key_actions": [],
            }, ensure_ascii=False), encoding="utf-8")

            with patch("spvideo.auto_director._analyze_visual_story", side_effect=AssertionError("must not run")):
                plan = analyze_auto_director_project(
                    root,
                    use_story_model=True,
                    api_key="unused",
                    scan_faces=False,
                )

            self.assertEqual(plan["story"]["source"], "pre_director")
            self.assertTrue(plan["story"]["audio_understanding"])
            self.assertEqual(len(plan["entities"]), 2)
            self.assertEqual(plan["entities"][0]["segment_ids"], ["001"])
            self.assertEqual(plan["entities"][1]["segment_ids"], ["002"])
            self.assertEqual(plan["entities"][0]["suggested_role"], "婆婆")
            question = next(
                item for item in plan["questions"]
                if item["kind"] == "person_identity" and item["segment_ids"] == ["001"]
            )
            self.assertEqual(question["suggested_answer"], "婆婆")
            self.assertIn("婆婆", [item["value"] for item in question["options"]])

    def test_story_relationship_candidates_become_visible_role_options(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = compose_auto_director_plan(
                project_dir=root,
                manifest={"segments": [
                    {"segment_id": "001", "segment_type": "with_human", "person_count": 1},
                    {"segment_id": "002", "segment_type": "with_human", "person_count": 2},
                ]},
                assets={"characters": {}, "directors": {}},
                face_entities=[],
                story={
                    "status": "ready",
                    "characters": [{
                        "id": "p1",
                        "visual_label": "蓝衣中年女人",
                        "role_name": "",
                        "role_candidates": ["婆婆", "母亲"],
                        "relationships": ["可能是白衣年轻女人的婆婆"],
                        "description": "中年女性，语气强势",
                        "confidence": 0.72,
                        "segment_ids": ["001", "002"],
                    }],
                },
            )

            person_question = next(
                item for item in plan["questions"]
                if item["kind"] == "person_identity"
            )
            person_options = [item["value"] for item in person_question["options"]]
            self.assertIn("婆婆", person_options)
            self.assertIn("母亲", person_options)
            self.assertIn("蓝衣中年女人", person_options)
            self.assertIn("婆婆", person_question["candidate_roles"])

            multi_question = next(
                item for item in plan["questions"]
                if item["kind"] == "multi_role_mapping"
            )
            self.assertIn("婆婆", multi_question["candidate_roles"])

    def test_custom_person_identity_answer_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = compose_auto_director_plan(
                project_dir=root,
                manifest={"segments": [
                    {"segment_id": "001", "segment_type": "with_human", "person_count": 1},
                ]},
                assets={"characters": {}, "directors": {}},
                face_entities=[{
                    "id": "face_01",
                    "visual_label": "蓝衣女人",
                    "segment_ids": ["001"],
                    "preview_frames": [],
                }],
                story={},
            )
            question = next(item for item in plan["questions"] if item["kind"] == "person_identity")
            save_auto_director(root, plan)

            answered = answer_auto_director_question(
                root,
                question_id=question["id"],
                answer="张婆婆",
            )

            self.assertEqual(answered["entities"][0]["resolved_as"], "张婆婆")
            self.assertEqual(answered["segment_decisions"]["001"]["suggested_roles"], ["张婆婆"])


if __name__ == "__main__":
    unittest.main()
