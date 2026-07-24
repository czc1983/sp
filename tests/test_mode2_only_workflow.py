from __future__ import annotations

import unittest
from pathlib import Path

from web_ui.server import _mode2_scail2_route_check


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


if __name__ == "__main__":
    unittest.main()
