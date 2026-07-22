from __future__ import annotations

import re
import unittest
from pathlib import Path


class LayerDefaultTests(unittest.TestCase):
    def test_first_five_layers_default_on_and_sam3_defaults_off(self) -> None:
        html = (Path(__file__).resolve().parents[1] / "web_ui" / "splitter_dashboard.html").read_text(encoding="utf-8")
        for key in (
            "use_pre_director",
            "use_omnishotcut",
            "use_scene_detect",
            "use_visual_merge",
            "use_face_id",
        ):
            self.assertRegex(html, rf'class="check on" data-key="{key}"')
        self.assertRegex(html, r'class="check" data-key="use_sam3_finalize"')
        self.assertNotRegex(html, r'class="check on" data-key="use_sam3_finalize"')

    def test_layer_numbers_are_continuous(self) -> None:
        html = (Path(__file__).resolve().parents[1] / "web_ui" / "splitter_dashboard.html").read_text(encoding="utf-8")
        titles = re.findall(r'<div class="check-title">([1-6])\.', html)
        self.assertEqual(titles[:6], ["1", "2", "3", "4", "5", "6"])


if __name__ == "__main__":
    unittest.main()
