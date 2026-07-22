from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spvideo.asset_store import (
    add_annotation,
    assign_director_role_annotation,
    get_director_plan,
    load_asset_store,
    upsert_character,
    upsert_director_plan,
)
from web_ui.server import _resolve_scail2_role_pairs


class DirectorPlanStoreTests(unittest.TestCase):
    def test_multi_role_plan_becomes_ready_after_identity_points_are_assigned(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            global_root = root / "global"
            global_characters = global_root / "characters"
            global_characters.mkdir(parents=True)
            video = root / "clip.mp4"
            first_ref = root / "new_lead.png"
            second_ref = root / "new_mother.png"
            for path in (video, first_ref, second_ref):
                path.touch()

            with patch(
                "spvideo.asset_store._GLOBAL_DIRS",
                {"root": global_root, "characters": global_characters},
            ):
                upsert_character(root, name="女主", ref_image=str(first_ref))
                upsert_character(root, name="婆婆", ref_image=str(second_ref))
                plan = upsert_director_plan(
                    root,
                    segment_id="002",
                    video_path=str(video),
                    roles=[{"name": "女主"}, {"name": "婆婆"}],
                    positive_prompt="a woman is slapped by another woman",
                )

                self.assertEqual(plan["status"], "incomplete")
                self.assertIn("缺少身份点", "；".join(plan["issues"]))

                lead = add_annotation(
                    root,
                    video_path=str(video),
                    time_seconds=0.0,
                    label_id=1,
                    label_name="女主",
                    kind="person",
                    point=[0.35, 0.45],
                    segment_id="002",
                )
                mother = add_annotation(
                    root,
                    video_path=str(video),
                    time_seconds=1.2,
                    label_id=2,
                    label_name="婆婆",
                    kind="person",
                    point=[0.91, 0.42],
                    segment_id="002",
                )
                assign_director_role_annotation(
                    root,
                    segment_id="002",
                    video_path=str(video),
                    role_name="女主",
                    annotation_id=lead["id"],
                )
                plan = assign_director_role_annotation(
                    root,
                    segment_id="002",
                    video_path=str(video),
                    role_name="婆婆",
                    annotation_id=mother["id"],
                )

                self.assertEqual(plan["status"], "ready")
                self.assertEqual([role["color"] for role in plan["roles"]], ["蓝色", "红色"])
                self.assertEqual(plan["roles"][1]["mark_time"], 1.2)
                self.assertEqual(get_director_plan(root, segment_id="002")["positive_prompt"], "a woman is slapped by another woman")
                self.assertIn("002", load_asset_store(root)["directors"])

    def test_clearing_director_role_annotation_keeps_role_unmarked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            global_root = root / "global"
            global_characters = global_root / "characters"
            global_characters.mkdir(parents=True)
            video = root / "clip.mp4"
            ref_image = root / "lead.png"
            for path in (video, ref_image):
                path.touch()

            with patch(
                "spvideo.asset_store._GLOBAL_DIRS",
                {"root": global_root, "characters": global_characters},
            ):
                upsert_character(root, name="女主", ref_image=str(ref_image))
                upsert_director_plan(
                    root,
                    segment_id="003",
                    video_path=str(video),
                    roles=[{"name": "女主"}],
                )
                ann = add_annotation(
                    root,
                    video_path=str(video),
                    time_seconds=0.5,
                    label_id=1,
                    label_name="女主",
                    kind="person",
                    point=[0.4, 0.5],
                    segment_id="003",
                )
                assign_director_role_annotation(
                    root,
                    segment_id="003",
                    video_path=str(video),
                    role_name="女主",
                    annotation_id=ann["id"],
                )

                cleared = upsert_director_plan(
                    root,
                    segment_id="003",
                    video_path=str(video),
                    roles=[{"name": "女主", "clear_annotation": True}],
                )

                self.assertEqual(cleared["roles"][0]["annotation_id"], "")
                self.assertIsNone(cleared["roles"][0]["point"])
                self.assertEqual(cleared["roles"][0]["track_status"], "missing")


class DirectorRoleMappingTests(unittest.TestCase):
    def test_director_annotations_sort_targets_by_screen_position(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "clip.mp4"
            lead_ref = root / "lead.png"
            mother_ref = root / "mother.png"
            for path in (video, lead_ref, mother_ref):
                path.touch()

            annotations = [
                {
                    "id": "lead_mark",
                    "type": "person",
                    "label_name": "女主",
                    "video_path": str(video),
                    "time": 0.0,
                    "point": [0.3, 0.5],
                },
                {
                    "id": "mother_mark",
                    "type": "person",
                    "label_name": "婆婆",
                    "video_path": str(video),
                    "time": 1.2,
                    "point": [0.9, 0.45],
                },
            ]
            director = {
                "roles": [
                    {"name": "婆婆", "annotation_id": "mother_mark"},
                    {"name": "女主", "annotation_id": "lead_mark"},
                ]
            }

            pairs, warning = _resolve_scail2_role_pairs(
                char_names=["婆婆", "女主"],
                characters={
                    "lead": {"name": "女主", "ref_image": str(lead_ref)},
                    "mother": {"name": "婆婆", "ref_image": str(mother_ref)},
                },
                originals=[],
                annotations=annotations,
                video_path=str(video),
                director_plan=director,
            )

        self.assertEqual([pair["name"] for pair in pairs], ["女主", "婆婆"])
        self.assertEqual(pairs[0]["source_time"], 0.0)
        self.assertEqual(warning, "")

    def test_failed_director_mark_keeps_identity_over_far_ready_same_role_track(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "clip.mp4"
            ref = root / "mother.png"
            track_dir = root / "tracks" / "far_ready"
            track_dir.mkdir(parents=True)
            (track_dir / "mask_0000.png").write_bytes(b"png")
            for path in (video, ref):
                path.touch()

            pairs, warning = _resolve_scail2_role_pairs(
                char_names=["mother"],
                characters={"mother": {"name": "mother", "ref_image": str(ref)}},
                originals=[],
                annotations=[
                    {
                        "id": "preferred_failed",
                        "type": "person",
                        "label_name": "mother",
                        "video_path": str(video),
                        "time": 1.2,
                        "point": [0.18, 0.41],
                        "track_status": "failed",
                    },
                    {
                        "id": "far_ready",
                        "type": "person",
                        "label_name": "mother",
                        "video_path": str(video),
                        "time": 0.1,
                        "point": [0.49, 0.50],
                        "track_status": "ready",
                        "track_dir": str(track_dir),
                        "tracked_frames": 50,
                    },
                ],
                video_path=str(video),
                director_plan={"roles": [{"name": "mother", "annotation_id": "preferred_failed"}]},
            )

        self.assertEqual(warning, "")
        self.assertEqual(pairs[0]["annotation_id"], "preferred_failed")
        self.assertAlmostEqual(pairs[0]["source_x"], 0.18)
        self.assertEqual(pairs[0]["track_dir"], "")

    def test_director_mark_can_reuse_nearby_ready_same_role_track(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "clip.mp4"
            ref = root / "mother.png"
            track_dir = root / "tracks" / "near_ready"
            track_dir.mkdir(parents=True)
            (track_dir / "mask_0000.png").write_bytes(b"png")
            for path in (video, ref):
                path.touch()

            pairs, warning = _resolve_scail2_role_pairs(
                char_names=["mother"],
                characters={"mother": {"name": "mother", "ref_image": str(ref)}},
                originals=[],
                annotations=[
                    {
                        "id": "preferred_failed",
                        "type": "person",
                        "label_name": "mother",
                        "video_path": str(video),
                        "time": 1.2,
                        "point": [0.18, 0.41],
                        "track_status": "failed",
                    },
                    {
                        "id": "near_ready",
                        "type": "person",
                        "label_name": "mother",
                        "video_path": str(video),
                        "time": 0.1,
                        "point": [0.26, 0.45],
                        "track_status": "ready",
                        "track_dir": str(track_dir),
                        "tracked_frames": 50,
                    },
                ],
                video_path=str(video),
                director_plan={"roles": [{"name": "mother", "annotation_id": "preferred_failed"}]},
            )

        self.assertEqual(warning, "")
        self.assertEqual(pairs[0]["annotation_id"], "near_ready")
        self.assertAlmostEqual(pairs[0]["source_x"], 0.26)
        self.assertEqual(pairs[0]["track_dir"], str(track_dir))


if __name__ == "__main__":
    unittest.main()
