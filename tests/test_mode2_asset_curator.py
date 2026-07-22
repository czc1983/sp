import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from spvideo.mode2_asset_curator import (
    MIN_PROP_CONFIDENCE,
    PROMPT_VERSION,
    curate_visual_groups,
)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def analyze_segment_keyframes(self, frame_paths, **kwargs):
        self.calls.append({"frame_paths": list(frame_paths), **kwargs})
        if not self.responses:
            return None
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _ready_result(group_id, *, kind="scene", frame_index=0, confidence=0.93):
    return {
        "group_id": group_id,
        "kind": kind,
        "name": "bedroom" if kind == "scene" else "lead",
        "identity": "person_01" if kind == "role" else "",
        "matched_role": "lead" if kind == "role" else "",
        "physical_scene": "bedroom_01" if kind == "scene" else "",
        "visible_props": [],
        "representative_frame_index": frame_index,
        "confidence": confidence,
        "reason": "clear visual evidence",
    }


class Mode2AssetCuratorTests(unittest.TestCase):
    def _frames(self, root: Path, count: int = 2):
        root.mkdir(parents=True, exist_ok=True)
        paths = []
        for index in range(count):
            path = root / f"frame_{index}.jpg"
            Image.new("RGB", (48, 64), (20 + index * 40, 30, 50)).save(path)
            paths.append(str(path))
        return paths

    def test_exact_cache_hit_skips_second_model_call(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            group = {"group_id": "VG001", "frame_paths": self._frames(root)}
            client = FakeClient([{"results": [_ready_result("VG001")]}])

            first = curate_visual_groups([group], cache_dir=root / "cache", client=client)
            second = curate_visual_groups([group], cache_dir=root / "cache", client=client)

            self.assertTrue(first[0]["usable"])
            self.assertTrue(second[0]["cache_hit"])
            self.assertEqual(len(client.calls), 1)
            self.assertTrue((root / "cache" / "manifest.json").is_file())
            manifest = json.loads((root / "cache" / "manifest.json").read_text("utf-8"))
            entry = next(iter(manifest["entries"].values()))
            self.assertTrue(Path(entry["result_path"]).is_file())
            self.assertTrue(Path(entry["raw_response_path"]).is_file())

    def test_successful_single_fallback_is_cached(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            group = {"group_id": "VG001", "frame_paths": self._frames(root)}
            low_confidence = _ready_result("VG001", confidence=0.3)
            client = FakeClient(
                [
                    {"results": [low_confidence]},
                    _ready_result("VG001", confidence=0.96),
                ]
            )

            first = curate_visual_groups([group], cache_dir=root / "cache", client=client)
            second = curate_visual_groups([group], cache_dir=root / "cache", client=client)

            self.assertEqual(first[0]["source"], "visual_model_single_fallback")
            self.assertTrue(first[0]["usable"])
            self.assertTrue(second[0]["cache_hit"])
            self.assertEqual(len(client.calls), 2)

    def test_json_fence_is_parsed(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            group = {"group_id": "VG001", "frame_paths": self._frames(root)}
            payload = json.dumps({"results": [_ready_result("VG001")]})
            client = FakeClient([f"```json\n{payload}\n```"])

            result = curate_visual_groups([group], cache_dir=root / "cache", client=client)

            self.assertEqual(result[0]["status"], "ready")
            self.assertEqual(result[0]["representative_frame_index"], 0)

    def test_out_of_bounds_representative_index_needs_review(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            group = {"group_id": "VG001", "frame_paths": self._frames(root)}
            client = FakeClient([{"results": [_ready_result("VG001", frame_index=9)]}])

            result = curate_visual_groups(
                [group],
                cache_dir=root / "cache",
                client=client,
                allow_single_group_fallback=False,
            )

            self.assertEqual(result[0]["status"], "needs_review")
            self.assertFalse(result[0]["usable"])
            self.assertIsNone(result[0]["representative_frame_index"])
            self.assertIn("invalid_representative_frame_index", result[0]["validation_errors"])

    def test_low_confidence_prop_is_not_usable_and_visible_props_are_filtered(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            group = {"group_id": "VG001", "frame_paths": self._frames(root)}
            raw = _ready_result(
                "VG001", kind="prop", confidence=MIN_PROP_CONFIDENCE - 0.01
            )
            raw["visible_props"] = [
                {"name": "chain", "frame_indices": [0], "confidence": 0.7}
            ]
            client = FakeClient([{"results": [raw]}])

            result = curate_visual_groups(
                [group],
                cache_dir=root / "cache",
                client=client,
                allow_single_group_fallback=False,
            )

            self.assertEqual(result[0]["status"], "needs_review")
            self.assertFalse(result[0]["usable"])
            self.assertEqual(result[0]["visible_props"], [])
            self.assertIn("low_prop_confidence", result[0]["validation_errors"])

    def test_mixed_group_is_preserved_and_not_retried(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            group = {"group_id": "VG001", "frame_paths": self._frames(root)}
            mixed = _ready_result("VG001", kind="mixed", confidence=0.98)
            client = FakeClient([{"results": [mixed]}])

            result = curate_visual_groups([group], cache_dir=root / "cache", client=client)

            self.assertEqual(result[0]["kind"], "mixed")
            self.assertEqual(result[0]["status"], "mixed")
            self.assertFalse(result[0]["usable"])
            self.assertEqual(len(client.calls), 1)

    def test_failed_batch_and_single_fallback_return_needs_review(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            group = {"group_id": "VG001", "frame_paths": self._frames(root)}
            client = FakeClient([None, None])

            result = curate_visual_groups([group], cache_dir=root / "cache", client=client)

            self.assertEqual(result[0]["status"], "needs_review")
            self.assertFalse(result[0]["usable"])
            self.assertEqual(result[0]["source"], "curation_failure")
            self.assertEqual(len(client.calls), 2)

    def test_pre_director_asset_manifest_is_normalized_with_zero_calls(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            group = {"group_id": "VG001", "frame_paths": self._frames(root)}
            client = FakeClient([])
            asset_manifest = {"results": [_ready_result("VG001", kind="role")]}

            result = curate_visual_groups(
                [group],
                cache_dir=root / "cache",
                client=client,
                known_roles=["lead"],
                asset_manifest=asset_manifest,
            )

            self.assertEqual(len(client.calls), 0)
            self.assertTrue(result[0]["usable"])
            self.assertEqual(result[0]["source"], "pre_director_asset_manifest")
            self.assertEqual(result[0]["matched_role"], "lead")

    def test_multiple_groups_use_one_batch_call(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            groups = [
                {"group_id": "VG001", "frame_paths": self._frames(root / "a")},
                {"group_id": "VG002", "frame_paths": self._frames(root / "b")},
            ]
            client = FakeClient(
                [
                    {
                        "results": [
                            _ready_result("VG001", kind="role"),
                            _ready_result("VG002", kind="scene"),
                        ]
                    }
                ]
            )

            result = curate_visual_groups(groups, cache_dir=root / "cache", client=client)

            self.assertEqual(len(result), 2)
            self.assertTrue(all(item["usable"] for item in result))
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(len(client.calls[0]["frame_paths"]), 2)
            self.assertIn(PROMPT_VERSION, client.calls[0]["prompt_override"])


if __name__ == "__main__":
    unittest.main()
