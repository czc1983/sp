import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from web_ui.server import (
    _ensure_wan_role_tracks,
    _ensure_wan_text_object_tracks,
    _run_wan_multi_role_transfer,
    _transfer_backend_for_role_count,
    _track_uses_sam3_text_objects,
    _use_wan22_transfer,
    _wan22_allow_experimental_multi_focus,
    _wan22_allow_local_mask_fallback,
    _wan_focus_settings,
    _wan_multi_role_limit,
)


class WanSam3PromptingTests(unittest.TestCase):
    def test_single_defaults_to_wan_and_multi_defaults_to_scail2(self):
        with patch.dict(
            os.environ,
            {"SINGLE_ROLE_TRANSFER_BACKEND": "", "MULTI_ROLE_TRANSFER_BACKEND": ""},
            clear=False,
        ):
            self.assertTrue(_use_wan22_transfer(1))
            self.assertFalse(_use_wan22_transfer(3))

    def test_explicit_backend_overrides_role_count_defaults(self):
        self.assertTrue(_use_wan22_transfer(3, "wan22"))
        self.assertFalse(_use_wan22_transfer(1, "scail2"))
        self.assertEqual(_transfer_backend_for_role_count(3, "scail2"), "scail2")

    def test_wan_multi_role_default_is_unlimited(self):
        with patch.dict(os.environ, {"WAN22_MULTI_ROLE_LIMIT": ""}, clear=False):
            self.assertEqual(_wan_multi_role_limit(), 0)

    def test_wan_focus_defaults_are_clean_not_haloed(self):
        with patch.dict(
            os.environ,
            {
                "WAN22_FOCUS_FEATHER_PIXELS": "",
                "WAN22_FOCUS_ERODE_PIXELS": "",
                "WAN22_FOCUS_DILATE_PIXELS": "",
                "WAN22_FOCUS_BACKGROUND_DIM": "",
                "WAN22_FOCUS_BACKGROUND_BLUR_PIXELS": "",
            },
            clear=False,
        ):
            settings = _wan_focus_settings()

        self.assertEqual(settings["feather_pixels"], 5)
        self.assertEqual(settings["erode_pixels"], 3)
        self.assertEqual(settings["dilate_pixels"], 0)
        self.assertLessEqual(settings["background_dim"], 0.25)
        self.assertEqual(settings["background_blur_pixels"], 0)

    def test_wan_multi_role_does_not_fallback_to_local_masks_by_default(self):
        with patch.dict(os.environ, {"WAN22_ALLOW_LOCAL_MASK_FALLBACK": ""}, clear=False):
            self.assertFalse(_wan22_allow_local_mask_fallback())

    def test_wan_multi_role_experimental_focus_is_off_by_default(self):
        with patch.dict(os.environ, {"WAN22_ALLOW_EXPERIMENTAL_MULTI_FOCUS": ""}, clear=False):
            self.assertFalse(_wan22_allow_experimental_multi_focus())

    def test_wan_branch_returns_raw_role_outputs_without_local_composite(self):
        source = inspect.getsource(_run_wan_multi_role_transfer)

        self.assertIn('"backend": "wan22_raw"', source)
        self.assertIn('"role_output_paths"', source)
        self.assertIn('"primary_output_path"', source)
        self.assertIn("build_focus_video_from_color_mask_video", source)
        self.assertIn("remote_sam3_color_mask", source)
        self.assertIn("wan22_mask_output_paths", source)
        self.assertIn("_wan22_allow_local_mask_fallback", source)
        self.assertIn("_wan22_allow_experimental_multi_focus", source)
        self.assertNotIn("composite_generated_foreground", source)
        self.assertNotIn("composite_protected_region", source)

    def test_wan_tracking_uses_sam3_text_objects_without_detector_boxes(self):
        source = inspect.getsource(_ensure_wan_role_tracks)
        implementation = inspect.getsource(_ensure_wan_text_object_tracks)
        self.assertIn("_ensure_wan_text_object_tracks", source)
        self.assertIn("track_text_objects", implementation)
        self.assertNotIn("_snap_role_point_to_person", source)
        self.assertNotIn("_snap_role_point_to_person", implementation)
        self.assertNotIn("track_by_box", implementation)
        self.assertNotIn("track_by_point", implementation)

    def test_only_reuses_direct_manual_point_tracks(self):
        with tempfile.TemporaryDirectory() as tmp:
            track_dir = Path(tmp)
            summary_path = track_dir / "track_summary.json"
            summary_path.write_text(
                json.dumps({
                    "prompt_source": "sam3_text_person_manual_identity",
                    "prompt_mode": "text_objects",
                    "sam3_object_id": 2,
                }),
                encoding="utf-8",
            )
            self.assertTrue(_track_uses_sam3_text_objects(track_dir))

            summary_path.write_text(
                json.dumps({"prompt_source": "yolo_box", "prompt_mode": "points"}),
                encoding="utf-8",
            )
            self.assertFalse(_track_uses_sam3_text_objects(track_dir))


if __name__ == "__main__":
    unittest.main()
