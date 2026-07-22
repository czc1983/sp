from __future__ import annotations

import unittest
from unittest.mock import patch

import requests

from spvideo.comfy_inventory import (
    _plugin_from_extension,
    _plugin_from_module,
    check_scail2_server,
    evaluate_scail2_requirements,
)


class ComfyInventoryTests(unittest.TestCase):
    def test_custom_node_module_is_grouped_by_plugin_directory(self) -> None:
        self.assertEqual(
            _plugin_from_module("custom_nodes.ComfyUI-KJNodes.nodes.image_nodes"),
            "ComfyUI-KJNodes",
        )
        self.assertEqual(_plugin_from_module("comfy_extras.nodes_sam3"), "ComfyUI Core")

    def test_frontend_extension_is_grouped_by_extension_root(self) -> None:
        self.assertEqual(
            _plugin_from_extension("/extensions/ComfyUI-SCAIL2-Easy/scail2_easy.js"),
            "ComfyUI-SCAIL2-Easy",
        )
        self.assertEqual(_plugin_from_extension("/extensions/core/groupNode.js"), "ComfyUI Core")

    def test_requirement_check_reports_missing_node_and_model(self) -> None:
        result = evaluate_scail2_requirements([], {})
        self.assertFalse(result["ready"])
        self.assertIn("SCAIL2SimpleVideo", result["missing_nodes"])
        self.assertIn("SAM3_TrackToMask", result["missing_nodes"])
        self.assertIn("ImageCompositeMasked", result["missing_nodes"])
        self.assertIn("MaskToImage", result["missing_nodes"])
        self.assertIn(
            "diffusion_models/wan2.1_14B_SCAIL_2_fp8_scaled.safetensors",
            result["missing_models"],
        )

    def test_scail2_check_falls_back_to_cached_inventory(self) -> None:
        class BrokenResponse:
            status_code = 500

            def raise_for_status(self) -> None:
                raise requests.HTTPError("server unavailable")

        class BrokenSession:
            def get(self, *_args, **_kwargs):
                return BrokenResponse()

        cached = {
            "profile_checks": {
                "scail2": {
                    "ready": True,
                    "missing_nodes": [],
                    "missing_optional_nodes": [],
                    "missing_models": [],
                }
            }
        }
        with patch("spvideo.comfy_inventory.load_inventory", return_value=cached):
            result = check_scail2_server("http://example.invalid", session=BrokenSession())

        self.assertTrue(result["ready"])
        self.assertEqual(result["source"], "cache")
        self.assertIn("live_check_unavailable", result["warning"])


if __name__ == "__main__":
    unittest.main()
