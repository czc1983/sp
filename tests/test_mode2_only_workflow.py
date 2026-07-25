from __future__ import annotations

from types import SimpleNamespace
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from web_ui.server import (
    _delete_storyboard_mode2_asset,
    _mode2_scail2_route_check,
    _storyboard_mode2_shot_subclips,
)


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

    def test_story_dashboard_has_compose_step_before_video_generation(self) -> None:
        html = (ROOT / "web_ui" / "story_generate_dashboard.html").read_text(encoding="utf-8")
        self.assertIn('data-module="compose"', html)
        self.assertIn('data-module-view="compose"', html)
        self.assertIn('id="composeAutoPackBtn"', html)
        self.assertIn("function composeCreateSmartPlan", html)
        self.assertIn("function composePackContainingShot", html)
        self.assertIn("function composeDropPlacement", html)
        self.assertIn("function composeHandDropPlacement", html)
        self.assertIn("function composeMoveShotToSelectedPack", html)
        self.assertIn("function bindComposeDetailDrop", html)
        self.assertIn("function composePackStackHtml", html)
        self.assertIn("function confirmComposePlan", html)
        self.assertIn('openComposePage({ rebuild: true })', html)
        self.assertIn("5-10 秒最佳，超过 15 秒禁止", html)
        self.assertIn("卡包架", html)
        self.assertIn("compose-pack-shelf", html)
        self.assertIn("compose-pack-stack-card", html)
        self.assertIn("compose-shot-id", html)
        self.assertIn("extra-dense", html)
        self.assertIn('data-compose-board-mode="current"', html)
        self.assertIn("compose-strip-hand", html)
        self.assertIn("aspect-ratio: 2 / 3", html)
        self.assertIn("#41f3ff", html)
        self.assertIn("#ff4fd8", html)
        self.assertIn("drop-before", html)
        self.assertIn("drop-after", html)
        self.assertIn("afterId", html)
        self.assertIn("swapId", html)
        self.assertIn("last.rect.left + last.rect.width * 0.35", html)
        self.assertIn("selectedItems.concat", html)
        self.assertIn('event.dataTransfer.setData("text/plain"', html)
        self.assertIn("点牌放回镜头池", html)
        self.assertLess(html.index('data-module="storyboard"'), html.index('data-module="compose"'))
        self.assertLess(html.index('data-module="compose"'), html.index('data-module="generate"'))

    def test_asset_delete_uses_in_app_confirm_not_browser_confirm(self) -> None:
        html = (ROOT / "web_ui" / "story_generate_dashboard.html").read_text(encoding="utf-8")
        self.assertNotIn("window.confirm", html)
        self.assertIn("asset-delete-confirm-overlay", html)
        self.assertIn("删除接口没有响应", html)
        self.assertIn("markCurrentProjectAssetIgnored", html)
        self.assertIn('manual_asset_status: "ignored"', html)
        ignored_pos = html.index('if (asset.manual_asset_status === "ignored") return true;')
        role_pos = html.index('if (asset.kind === "role") return false;')
        self.assertLess(ignored_pos, role_pos)

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

    def test_save_shot_subclips_can_promote_to_real_timeline(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            source.write_bytes(b"placeholder")
            video = root / "shot.mp4"
            video.write_bytes(b"placeholder")
            store_path = root / "assets" / "storyboard_assets.json"
            store_path.parent.mkdir(parents=True)
            store_path.write_text(json.dumps({
                "video_path": str(source),
                "assets": [{
                    "id": "R001",
                    "kind": "role",
                    "used_shots": ["S001"],
                    "identity_anchors": [{"shot_id": "S001", "point": [0.5, 0.5]}],
                }],
                "shots": [{
                    "segment_id": "S001",
                    "start": 10.0,
                    "end": 17.0,
                    "duration": 7.0,
                    "person_count": 1,
                    "role_asset_ids": ["R001"],
                    "asset_ids": ["R001"],
                }],
            }), encoding="utf-8")

            def fake_create(_video_path, ranges, existing_subshots=None):
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
                patch("web_ui.server._mode2_create_reference_mask_subclips", side_effect=fake_create),
                patch("web_ui.server._mode2_ensure_shot_preview_clips", return_value=False),
            ):
                result = _storyboard_mode2_shot_subclips({
                    "project_dir": str(root),
                    "video_path": str(video),
                    "segment_id": "S001",
                    "cut_times": [1.0],
                    "manual_cut": True,
                    "save": True,
                    "promote_to_timeline": True,
                    "selected_subclip_index": 1,
                })

            self.assertTrue(result["promoted_to_timeline"])
            self.assertEqual(result["selected_index"], 1)
            self.assertEqual([shot["segment_id"] for shot in result["segments"]], ["S001", "S002"])
            self.assertEqual(
                [(shot["start"], shot["end"]) for shot in result["segments"]],
                [(10.0, 11.0), (11.0, 17.0)],
            )
            saved = json.loads(store_path.read_text(encoding="utf-8"))
            self.assertEqual([shot["segment_id"] for shot in saved["shots"]], ["S001", "S002"])
            role = saved["assets"][0]
            self.assertIn("S001", role["used_shots"])
            self.assertIn("S002", role["used_shots"])
            self.assertIn("S002", [item["shot_id"] for item in role["identity_anchors"]])

    def test_delete_storyboard_asset_keeps_image_files_and_unlinks_shots(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store_path = root / "assets" / "storyboard_assets.json"
            store_path.parent.mkdir(parents=True)
            image_path = root / "assets" / "manual_assets" / "R001" / "target.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"image")
            store_path.write_text(
                json.dumps({
                    "video_path": str(root / "source.mp4"),
                    "assets": [
                        {
                            "id": "R001",
                            "kind": "role",
                            "name": "老公",
                            "target_image": str(image_path),
                        },
                        {"id": "scene_1", "kind": "scene", "name": "房间"},
                    ],
                    "shots": [{
                        "segment_id": "S001",
                        "start": 0,
                        "end": 2,
                        "duration": 2,
                        "asset_ids": ["R001", "scene_1"],
                        "role_asset_ids": ["R001"],
                        "scene_asset_ids": ["scene_1"],
                    }],
                }, ensure_ascii=False),
                encoding="utf-8",
            )

            with patch("web_ui.server._audit_storyboard_mode2_assets", return_value={}):
                result = _delete_storyboard_mode2_asset({
                    "project_dir": str(root),
                    "asset_id": "R001",
                })

            self.assertFalse(result["files_deleted"])
            self.assertTrue(image_path.exists())
            saved = json.loads(store_path.read_text(encoding="utf-8"))
            self.assertEqual([asset["id"] for asset in saved["assets"]], ["scene_1"])
            shot = saved["shots"][0]
            self.assertEqual(shot["asset_ids"], ["scene_1"])
            self.assertEqual(shot["role_asset_ids"], [])
            self.assertEqual(shot["scene_asset_ids"], ["scene_1"])


if __name__ == "__main__":
    unittest.main()
