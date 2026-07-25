from __future__ import annotations

from types import SimpleNamespace
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from web_ui.server import _mode2_scail2_route_check, _storyboard_mode2_shot_subclips


ROOT = Path(__file__).resolve().parents[1]


class Mode2OnlyWorkflowTests(unittest.TestCase):
    def test_legacy_mode1_dashboard_removed(self) -> None:
        self.assertFalse((ROOT / "web_ui" / "splitter_dashboard.html").exists())

    def test_server_root_serves_story_dashboard_and_blocks_legacy_routes(self) -> None:
        server = (ROOT / "web_ui" / "server.py").read_text(encoding="utf-8")
        self.assertIn('INDEX = ROOT / "story_generate_dashboard.html"', server)
        self.assertIn('"legacy_mode1_removed"', server)
        self.assertIn('"/api/transfer-segment"', server)
        self.assertIn('"/api/split"', server)

    def test_story_dashboard_has_no_mode_switch_or_legacy_buttons(self) -> None:
        html = (ROOT / "web_ui" / "story_generate_dashboard.html").read_text(encoding="utf-8")
        for text in (
            "切换模式",
            "回到模式 1",
            "白膜路线",
            "Seedance直出",
            "识别轨道",
            "生成当前子镜头蒙版",
            "合并已生成蒙版",
        ):
            self.assertNotIn(text, html)

    def test_scail2_route_accepts_single_clean_role_only(self) -> None:
        shot = {
            "segment_id": "S001",
            "person_count": 1,
            "role_asset_ids": ["R001"],
            "description": "single person close-up",
        }
        assets = [{"id": "R001", "kind": "role"}]
        ok, reason, meta = _mode2_scail2_route_check(shot, assets)
        self.assertTrue(ok, reason)
        self.assertEqual(meta["role_count"], 1)

    def test_scail2_route_rejects_multi_or_contact_shots(self) -> None:
        assets = [{"id": "R001", "kind": "role"}, {"id": "R002", "kind": "role"}]
        multi_shot = {
            "segment_id": "S002",
            "person_count": 2,
            "role_asset_ids": ["R001", "R002"],
            "description": "two people in frame",
        }
        ok, reason, meta = _mode2_scail2_route_check(multi_shot, assets)
        self.assertFalse(ok)
        self.assertEqual(meta["role_count"], 2)
        self.assertIn("Seedance", reason)

        contact_shot = {
            "segment_id": "S003",
            "person_count": 1,
            "role_asset_ids": ["R001"],
            "description": "single visible person but another person's hand touches the shoulder",
        }
        ok, reason, meta = _mode2_scail2_route_check(contact_shot, assets)
        self.assertFalse(ok)
        self.assertTrue(meta["contact_risk"])
        self.assertIn("Seedance", reason)

    def test_manual_empty_shot_cuts_do_not_rerun_hardcut(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "shot.mp4"
            video.write_bytes(b"placeholder")
            captured_ranges: list[list[tuple[float, float]]] = []

            def fake_create(_video_path, ranges, existing_subshots=None):
                captured_ranges.append(list(ranges))
                return [
                    {
                        "index": index,
                        "start": start,
                        "end": end,
                        "duration": round(end - start, 3),
                        "path": str(root / f"sub{index}.mp4"),
                    }
                    for index, (start, end) in enumerate(ranges, 1)
                ]

            with (
                patch("spvideo.ffmpeg_tools.probe_video", return_value=SimpleNamespace(duration=7.0)),
                patch("web_ui.server._mode2_reference_video_hard_cuts", side_effect=AssertionError("hardcut should not run")),
                patch("web_ui.server._mode2_create_reference_mask_subclips", side_effect=fake_create),
            ):
                result = _storyboard_mode2_shot_subclips({
                    "project_dir": str(root),
                    "video_path": str(video),
                    "segment_id": "S001",
                    "cut_times": [],
                    "manual_cut": True,
                })

            self.assertEqual(result["source"], "manual")
            self.assertEqual(result["cut_times"], [])
            self.assertEqual(captured_ranges, [[(0.0, 7.0)]])
            self.assertEqual(len(result["subclips"]), 1)

    def test_empty_cut_times_default_to_manual_for_legacy_undo(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "shot.mp4"
            video.write_bytes(b"placeholder")
            captured_ranges: list[list[tuple[float, float]]] = []

            def fake_create(_video_path, ranges, existing_subshots=None):
                captured_ranges.append(list(ranges))
                return [
                    {
                        "index": index,
                        "start": start,
                        "end": end,
                        "duration": round(end - start, 3),
                        "path": str(root / f"sub{index}.mp4"),
                    }
                    for index, (start, end) in enumerate(ranges, 1)
                ]

            with (
                patch("spvideo.ffmpeg_tools.probe_video", return_value=SimpleNamespace(duration=7.0)),
                patch("web_ui.server._mode2_reference_video_hard_cuts", side_effect=AssertionError("hardcut should not run")),
                patch("web_ui.server._mode2_create_reference_mask_subclips", side_effect=fake_create),
            ):
                result = _storyboard_mode2_shot_subclips({
                    "project_dir": str(root),
                    "video_path": str(video),
                    "segment_id": "S001",
                    "cut_times": [],
                })

            self.assertEqual(result["source"], "manual")
            self.assertEqual(result["cut_times"], [])
            self.assertEqual(captured_ranges, [[(0.0, 7.0)]])

    def test_auto_empty_shot_cuts_still_run_hardcut(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "shot.mp4"
            video.write_bytes(b"placeholder")
            captured_ranges: list[list[tuple[float, float]]] = []

            def fake_create(_video_path, ranges, existing_subshots=None):
                captured_ranges.append(list(ranges))
                return [
                    {
                        "index": index,
                        "start": start,
                        "end": end,
                        "duration": round(end - start, 3),
                        "path": str(root / f"sub{index}.mp4"),
                    }
                    for index, (start, end) in enumerate(ranges, 1)
                ]

            with (
                patch("spvideo.ffmpeg_tools.probe_video", return_value=SimpleNamespace(duration=7.0)),
                patch("web_ui.server._mode2_reference_video_hard_cuts", return_value=([(0.0, 1.0), (1.0, 7.0)], "")),
                patch("web_ui.server._mode2_create_reference_mask_subclips", side_effect=fake_create),
            ):
                result = _storyboard_mode2_shot_subclips({
                    "project_dir": str(root),
                    "video_path": str(video),
                    "segment_id": "S001",
                    "cut_times": [],
                    "auto_cut": True,
                })

            self.assertEqual(result["source"], "local_hardcut")
            self.assertEqual(result["cut_times"], [1.0])
            self.assertEqual(captured_ranges, [[(0.0, 1.0), (1.0, 7.0)]])

    def test_clear_shot_subclips_removes_saved_cut_data(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store_path = root / "assets" / "storyboard_assets.json"
            store_path.parent.mkdir(parents=True)
            store_path.write_text(json.dumps({
                "shots": [{
                    "segment_id": "S001",
                    "subshots": [{"index": 1, "start": 0.0, "end": 7.0, "path": "sub1.mp4"}],
                    "subshot_cut_times": [],
                    "subshot_status": "confirmed",
                    "subshot_updated_at": 1,
                }],
            }), encoding="utf-8")

            result = _storyboard_mode2_shot_subclips({
                "project_dir": str(root),
                "segment_id": "S001",
                "clear_subclips": True,
                "save": True,
            })

            self.assertTrue(result["cleared"])
            self.assertEqual(result["subclips"], [])
            saved = json.loads(store_path.read_text(encoding="utf-8"))
            shot = saved["shots"][0]
            self.assertNotIn("subshots", shot)
            self.assertNotIn("subshot_cut_times", shot)
            self.assertNotIn("subshot_status", shot)
            self.assertNotIn("subshot_updated_at", shot)


if __name__ == "__main__":
    unittest.main()
