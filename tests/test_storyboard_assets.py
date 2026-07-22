import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from web_ui.server import (
    _attach_storyboard_asset_usage,
    _build_storyboard_assets_v2,
    _curate_storyboard_assets_v2,
    _flag_storyboard_asset_source_quality,
    _mode2_attach_existing_refined_outputs,
    _mode2_apply_asset_contract_fields,
    _mode2_dedupe_scene_assets,
    _mode2_refinement_provenance_payload,
    _mode2_split_scene_asset_in_data,
    _load_or_run_storyboard_mode2_understanding,
    _refresh_mode2_structured_fields,
    _split_storyboard_mode2_scenes,
    _storyboard_mode2_understanding_view,
)


class StoryboardAssetsTests(unittest.TestCase):
    @staticmethod
    def _write_dhash_pattern(path: Path, *, reverse: bool = False, bands: bool = False) -> None:
        from PIL import Image

        image = Image.new("L", (72, 64))
        for y in range(image.height):
            for x in range(image.width):
                if bands:
                    value = 255 if (y // 8) % 2 else 0
                else:
                    value = int(255 * x / max(1, image.width - 1))
                    if reverse:
                        value = 255 - value
                image.putpixel((x, y), value)
        image.convert("RGB").save(path)

    def test_assets_use_auto_director_roles_and_split_semantic_scenes(self):
        segments = [
            {
                "segment_id": "001",
                "start": 0.0,
                "end": 1.0,
                "person_count": 1,
                "description": "庭院中，母亲训斥女儿。",
            },
            {
                "segment_id": "002",
                "start": 1.0,
                "end": 2.0,
                "person_count": 2,
                "description": "庭院中，母亲训斥女儿。",
            },
            {
                "segment_id": "020",
                "start": 20.0,
                "end": 21.0,
                "person_count": 0,
                "description": "豪车车队在道路上行驶。",
            },
        ]
        plan = {
            "story": {
                "characters": [
                    {
                        "visual_label": "林爱萍",
                        "role_candidates": ["养母"],
                        "description": "中年女性，蓝色连衣裙。",
                        "segment_ids": ["001", "002"],
                    },
                    {
                        "visual_label": "沈辰",
                        "role_candidates": ["二哥"],
                        "description": "年轻男性。",
                        "segment_ids": ["020"],
                    },
                ]
            }
        }

        assets = _build_storyboard_assets_v2("source.mp4", segments, plan)

        role_names = [asset["name"] for asset in assets if asset["kind"] == "role"]
        scene_assets = [asset for asset in assets if asset["kind"] == "scene"]
        self.assertIn("林爱萍", role_names)
        self.assertIn("沈辰", role_names)
        self.assertEqual(len(scene_assets), 2)
        self.assertEqual(scene_assets[0]["source_segment_ids"], ["001", "002"])
        self.assertEqual(scene_assets[1]["source_segment_ids"], ["020"])

    def test_understanding_view_preserves_asset_and_frame_manifests(self):
        plan = {
            "status": "ready",
            "asset_manifest": {"results": [{"group_id": "G001", "kind": "scene"}]},
            "analysis_frame_manifest": [{"index": 1, "time": 1.25, "path": "f1.jpg"}],
            "understanding_analysis_fresh": True,
        }

        fresh = _storyboard_mode2_understanding_view(
            plan,
            project_dir="project",
            cache_path="pre_director.json",
            source_path="source.mp4",
        )
        cached = _storyboard_mode2_understanding_view(
            plan,
            project_dir="project",
            cache_path="pre_director.json",
            source_path="source.mp4",
            cache_hit=True,
        )

        self.assertEqual(fresh["asset_manifest"], plan["asset_manifest"])
        self.assertEqual(fresh["analysis_frame_manifest"], plan["analysis_frame_manifest"])
        self.assertTrue(fresh["understanding_analysis_fresh"])
        self.assertFalse(cached["understanding_analysis_fresh"])

    def test_enabled_understanding_requests_one_pass_asset_manifest(self):
        class FakeFrame:
            def to_dict(self):
                return {"time": 1.0, "path": "frame.jpg"}

        captured = {}

        def fake_analyze(_frames, **kwargs):
            captured.update(kwargs)
            return {
                "status": "ready",
                "story_summary": "summary",
                "characters": [],
                "scenes": [],
                "boundary_hints": [],
                "asset_manifest": {
                    "results": [{
                        "group_id": "scene_001",
                        "kind": "scene",
                        "evidence_times": [1.0],
                        "representative_frame_index": 1,
                        "confidence": 0.9,
                    }],
                },
                "analysis_frame_manifest": [{"index": 1, "time": 1.0, "path": "frame.jpg"}],
            }

        with TemporaryDirectory() as tmp, \
                patch("web_ui.server._storyboard_mode2_sample_frame_features", return_value=[FakeFrame()]), \
                patch("web_ui.server._storyboard_mode2_has_audio", return_value=False), \
                patch("spvideo.pre_director.analyze_pre_director", side_effect=fake_analyze):
            understanding, sampled = _load_or_run_storyboard_mode2_understanding(
                video_path="source.mp4",
                project_dir=tmp,
                duration=4.0,
                payload={
                    "use_pre_director": True,
                    "force_understand": True,
                    "pre_director_api_key": "test-key",
                },
                add_log=lambda _message: None,
            )

        self.assertTrue(captured["include_asset_manifest"])
        self.assertEqual(understanding["status"], "ready")
        self.assertEqual(understanding["asset_manifest"]["results"][0]["group_id"], "scene_001")
        self.assertEqual(sampled, [{"time": 1.0, "path": "frame.jpg"}])

    def test_curator_maps_manifest_to_asset_without_model_call(self):
        from PIL import Image

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame_a = root / "frame_a.jpg"
            frame_b = root / "frame_b.jpg"
            Image.new("RGB", (32, 32), (20, 30, 40)).save(frame_a)
            Image.new("RGB", (32, 32), (80, 90, 100)).save(frame_b)
            assets = [{
                "id": "scene_1",
                "kind": "scene",
                "name": "scene_1",
                "keyframes": [
                    {"time": 1.0, "path": str(frame_a)},
                    {"time": 2.0, "path": str(frame_b)},
                ],
            }]
            understanding = {
                "status": "ready",
                "summary": "bedroom scene",
                "characters": [],
                "analysis_frame_manifest": [
                    {"index": 1, "time": 1.0, "path": str(frame_a)},
                    {"index": 2, "time": 2.0, "path": str(frame_b)},
                ],
                "asset_manifest": {"results": [{
                    "group_id": "SCENE_GLOBAL_01",
                    "kind": "scene",
                    "name": "bedroom",
                    "physical_scene": "bedroom_01",
                    "evidence_times": [2.0],
                    "representative_frame_index": 2,
                    "visible_props": [],
                    "confidence": 0.94,
                    "reason": "clear room evidence",
                }]},
            }

            summary = _curate_storyboard_assets_v2(
                assets,
                understanding,
                project_dir=root,
                api_key="",
                allow_model_call=False,
            )

        self.assertEqual(summary["mapped_manifest_count"], 1)
        self.assertEqual(summary["model_call_scope_count"], 0)
        self.assertEqual(assets[0]["curator_status"], "ready")
        self.assertEqual(assets[0]["curator_representative_image"], str(frame_b))
        self.assertEqual(assets[0]["curator_confidence"], 0.94)
        self.assertEqual(assets[0]["name"], "bedroom")

    def test_disabled_understanding_without_manifest_never_calls_curator(self):
        from PIL import Image

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = root / "frame.jpg"
            Image.new("RGB", (32, 32), (20, 30, 40)).save(frame)
            assets = [{
                "id": "scene_1",
                "kind": "scene",
                "name": "scene_1",
                "keyframes": [{"time": 1.0, "path": str(frame)}],
            }]
            understanding = {
                "status": "disabled",
                "asset_manifest": {"results": []},
                "analysis_frame_manifest": [],
            }
            with patch("spvideo.mode2_asset_curator.curate_visual_groups") as mocked:
                summary = _curate_storyboard_assets_v2(
                    assets,
                    understanding,
                    project_dir=root,
                    api_key="paid-key-must-not-be-used",
                    allow_model_call=False,
                )

        mocked.assert_not_called()
        self.assertEqual(summary["status"], "needs_review")
        self.assertEqual(summary["model_call_scope_count"], 0)
        self.assertEqual(assets[0]["curator_status"], "needs_review")
        self.assertTrue(assets[0]["curator_needs_review"])

    def test_asset_usage_prefers_source_segment_ids(self):
        assets = [
            {"id": "role_1", "kind": "role", "source_segment_ids": ["001"]},
            {"id": "scene_1", "kind": "scene", "source_segment_ids": ["001"]},
            {"id": "scene_2", "kind": "scene", "source_segment_ids": ["020"]},
        ]
        shots = [
            {"segment_id": "S001", "source_segment_ids": ["001"], "person_count": 2},
            {"segment_id": "S002", "source_segment_ids": ["020"], "person_count": 0},
        ]

        _attach_storyboard_asset_usage(assets, shots)

        self.assertEqual(shots[0]["asset_ids"], ["role_1", "scene_1"])
        self.assertEqual(shots[1]["asset_ids"], ["scene_2"])
        self.assertEqual(assets[0]["used_shots"], ["S001"])
        self.assertEqual(assets[2]["used_shots"], ["S002"])

    def test_asset_usage_does_not_read_compiled_prompt(self):
        assets = [
            {"id": "role_3", "kind": "role", "name": "Fiance", "source_segment_ids": []},
        ]
        shots = [
            {
                "segment_id": "S001",
                "source_segment_ids": [],
                "person_count": 1,
                "description": "A woman walks alone in the room.",
                "prompt": "Fiance@image3=role Fiance; use role_3 as a reference image.",
            },
        ]

        _attach_storyboard_asset_usage(assets, shots)

        self.assertEqual(shots[0]["asset_ids"], [])
        self.assertEqual(assets[0]["used_shots"], [])

    def test_refresh_backfills_semantic_scene_and_prop_assets(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = {
                "video_path": "source.mp4",
                "assets": [
                    {"id": "role_1", "kind": "role", "name": "妻子", "source_segment_ids": ["U002"]},
                ],
                "shots": [
                    {"segment_id": "S001", "source_segment_ids": ["U001"], "person_count": 0},
                    {"segment_id": "S002", "source_segment_ids": ["U003"], "person_count": 1},
                ],
                "semantic_segments": [
                    {
                        "segment_id": "U001",
                        "start": 0.0,
                        "end": 2.5,
                        "description": "海边城市黄昏，海外某岛国文字。",
                        "person_count": 0,
                    },
                    {
                        "segment_id": "U003",
                        "start": 4.67,
                        "end": 7.33,
                        "description": "冷色封闭房间，白衣男子脚踝被铁链锁住，墙上挂满刑具。",
                        "key_action": "男子挣扎，铁链拖在地面。",
                        "person_count": 1,
                    },
                    {
                        "segment_id": "U005",
                        "start": 76.83,
                        "end": 117.14,
                        "description": "明亮卧室，黑衣男子躺在床上注视女子。",
                        "person_count": 2,
                    },
                ],
            }

            _refresh_mode2_structured_fields(root, data)

        kinds = [asset["kind"] for asset in data["assets"]]
        names = [asset["name"] for asset in data["assets"]]
        self.assertEqual(kinds.count("role"), 1)
        self.assertGreaterEqual(kinds.count("scene"), 3)
        self.assertIn("粗铁链", names)
        self.assertIn("床", names)
        self.assertIn("scene_1", data["shots"][0]["asset_ids"])
        self.assertIn("scene_2", data["shots"][1]["asset_ids"])
        prop = next(asset for asset in data["assets"] if asset["name"] == "粗铁链")
        self.assertEqual(prop["source_segment_ids"], ["U003"])

    def test_contract_display_prefers_clean_representatives(self):
        assets = [
            {
                "id": "role_1",
                "kind": "role",
                "source_image": "original_role_evidence.jpg",
                "source_images": ["original_role_frame.jpg"],
            },
            {
                "id": "scene_1",
                "kind": "scene",
                "refinement_kind": "clean_background",
                "refined_source_image": "clean_scene.png",
            },
            {
                "id": "scene_2",
                "kind": "scene",
                "source_quality_status": "scene_person_heavy",
                "source_usage_role": "shot_reference",
                "refinement_kind": "clean_background",
                "refined_source_image": "bad_people_scene.png",
                "source_quality_warning": "too many people",
            },
            {
                "id": "prop_1",
                "kind": "prop",
                "refinement_status": "ready",
                "refinement_kind": "prop_cutout",
                "refined_cutout_image": "bad_bed_crop.png",
                "refinement_quality": {"area_ratio": 0.59, "bbox_ratio": 0.76, "component_count": 1},
            },
            {
                "id": "prop_2",
                "kind": "prop",
                "source_quality_status": "prop_needs_visual_check",
                "source_usage_role": "object_evidence_candidate",
                "refinement_status": "ready",
                "refinement_kind": "prop_cutout",
                "refined_cutout_image": "unconfirmed_chain_crop.png",
                "refinement_quality": {"area_ratio": 0.01, "bbox_ratio": 0.12, "component_count": 1},
            },
            {
                "id": "prop_3",
                "kind": "prop",
                "manual_asset_status": "approved",
                "source_quality_status": "prop_needs_visual_check",
                "source_usage_role": "object_evidence_candidate",
                "refinement_status": "ready",
                "refinement_kind": "prop_cutout",
                "refined_cutout_image": "approved_chain_crop.png",
                "refinement_quality": {"area_ratio": 0.01, "bbox_ratio": 0.12, "component_count": 1},
            },
            {
                "id": "scene_3",
                "kind": "scene",
                "source_image": "semantic_room_sheet.jpg",
                "source_usage_role": "environment_reference",
            },
        ]

        _mode2_apply_asset_contract_fields(assets, [], {})

        self.assertEqual(assets[0]["representative_image"], "")
        self.assertEqual(assets[0]["representative_status"], "needs_target")
        self.assertEqual(assets[1]["representative_image"], "")
        self.assertEqual(assets[1]["representative_status"], "stale_refinement")
        self.assertFalse(assets[1]["representative_is_clean"])
        self.assertEqual(assets[1]["refinement_provenance_status"], "missing_provenance")
        self.assertTrue(assets[1]["refinement_quarantined"])
        self.assertEqual(assets[2]["representative_image"], "")
        self.assertEqual(assets[2]["representative_status"], "shot_reference_only")
        self.assertTrue(assets[2]["asset_hidden_by_default"])
        self.assertEqual(assets[2]["candidate_preview_image"], "")
        self.assertFalse(assets[2]["visual_confirmed"])
        self.assertEqual(assets[2]["source_trust_level"], "shot_reference_only")
        self.assertEqual(assets[3]["representative_image"], "")
        self.assertEqual(assets[3]["representative_status"], "needs_clean_source")
        self.assertEqual(assets[4]["representative_image"], "")
        self.assertEqual(assets[4]["representative_status"], "needs_clean_source")
        self.assertTrue(assets[4]["asset_hidden_by_default"])
        self.assertEqual(assets[4]["candidate_preview_image"], "")
        self.assertFalse(assets[4]["visual_confirmed"])
        self.assertEqual(assets[4]["source_visual_status"], "unconfirmed_semantic_prop")
        self.assertEqual(assets[4]["source_trust_level"], "semantic_candidate_only")
        self.assertEqual(assets[5]["representative_image"], "approved_chain_crop.png")
        self.assertTrue(assets[5]["representative_is_clean"])
        self.assertTrue(assets[5]["visual_confirmed"])
        self.assertEqual(assets[6]["representative_image"], "")
        self.assertEqual(assets[6]["representative_status"], "needs_clean_source")
        self.assertEqual(assets[6]["candidate_preview_image"], "semantic_room_sheet.jpg")
        self.assertFalse(assets[6]["visual_confirmed"])
        self.assertEqual(assets[6]["source_visual_status"], "unconfirmed_scene_candidate")
        self.assertEqual(assets[6]["source_trust_level"], "scene_candidate_only")

    def test_flag_marks_prop_sources_as_unconfirmed_evidence(self):
        assets = [
            {
                "id": "prop_1",
                "kind": "prop",
                "source_image": "chain_sheet.jpg",
                "source_kind": "prop_keyframe_bundle",
                "track_status": "ready",
                "refinement_status": "ready",
                "refinement_kind": "prop_cutout",
                "refined_cutout_image": "chain_crop.png",
                "refinement_quality": {"area_ratio": 0.01, "bbox_ratio": 0.12, "component_count": 1},
            },
        ]

        _flag_storyboard_asset_source_quality(assets)
        _mode2_apply_asset_contract_fields(assets, [], {})

        self.assertEqual(assets[0]["source_quality_status"], "prop_needs_visual_check")
        self.assertEqual(assets[0]["source_usage_role"], "object_evidence_candidate")
        self.assertEqual(assets[0]["track_status"], "needs_manual_review")
        self.assertEqual(assets[0]["representative_image"], "")
        self.assertEqual(assets[0]["candidate_preview_image"], "")
        self.assertFalse(assets[0]["visual_confirmed"])
        self.assertEqual(assets[0]["source_visual_status"], "unconfirmed_semantic_prop")
        self.assertEqual(assets[0]["source_trust_level"], "semantic_candidate_only")

    def test_mixed_scene_groups_block_target_and_refined_representatives(self):
        assets = [{
            "id": "scene_mixed",
            "kind": "scene",
            "manual_asset_status": "approved",
            "target_image": "target_european_room.png",
            "refinement_status": "ready",
            "refinement_kind": "clean_background",
            "refined_source_image": "refined_room.png",
            "keyframes": [
                {"path": "room_a.jpg", "visual_group_id": "room_a", "metrics": {}},
                {"path": "room_b.jpg", "visual_group_id": "room_b", "metrics": {}},
            ],
        }]

        _mode2_apply_asset_contract_fields(assets, [], {})

        asset = assets[0]
        self.assertTrue(asset["scene_mixed_visual_groups"])
        self.assertEqual(asset["scene_visual_group_ids"], ["room_a", "room_b"])
        self.assertEqual(asset["representative_image"], "")
        self.assertEqual(asset["representative_status"], "mixed_reference_bundle")
        self.assertFalse(asset["visual_confirmed"])
        self.assertEqual(asset["target_asset"]["status"], "blocked_mixed_source")
        self.assertFalse(asset["seedance_policy"]["send_target_image"])
        self.assertFalse(asset["seedance_ready"])

    def test_scene_refinement_requires_verified_source_provenance(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jpg"
            refined = root / "clean_bg.png"
            source.write_bytes(b"current-source")
            refined.write_bytes(b"refined-output")
            asset = {
                "id": "scene_5",
                "kind": "scene",
                "target_image": "target_scene.png",
                "source_segment_ids": ["U005"],
                "keyframes": [{"path": str(source), "score": 1.0, "metrics": {}}],
                "refinement_status": "ready",
                "refinement_kind": "clean_background",
                "refinement_method": "existing_grounding_dino_sam_lama",
                "refined_source_image": str(refined),
            }

            _mode2_apply_asset_contract_fields([asset], [], {})

        self.assertEqual(asset["refinement_provenance_status"], "missing_provenance")
        self.assertEqual(asset["refinement_status"], "quarantined")
        self.assertTrue(asset["refinement_quarantined"])
        self.assertEqual(asset["representative_image"], "")
        self.assertEqual(asset["representative_status"], "stale_refinement")
        self.assertFalse(asset["visual_confirmed"])
        self.assertEqual(asset["target_asset"]["status"], "blocked_untrusted_scene_source")
        self.assertFalse(asset["seedance_policy"]["send_target_image"])
        self.assertFalse(asset["seedance_ready"])

    def test_scene_refinement_blocks_stale_and_cross_asset_provenance(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jpg"
            refined = root / "clean_bg.png"
            source.write_bytes(b"source-v1")
            refined.write_bytes(b"refined-output")
            base_asset = {
                "id": "scene_5",
                "kind": "scene",
                "source_segment_ids": ["U005"],
                "keyframes": [{"path": str(source), "score": 1.0, "metrics": {}}],
            }
            provenance = _mode2_refinement_provenance_payload(
                base_asset,
                source_path=source,
                outputs={"refined_source_image": refined},
                job_id="job-old",
                prompt_id="prompt-old",
                method="grounding_dino_sam_lama",
                kind="clean_background",
            )

            source.write_bytes(b"source-v2")
            stale_asset = {
                **base_asset,
                "refinement_kind": "clean_background",
                "refined_source_image": str(refined),
                "refinement_provenance": provenance,
            }
            _mode2_apply_asset_contract_fields([stale_asset], [], {})
            self.assertEqual(stale_asset["refinement_provenance_status"], "source_hash_mismatch")
            self.assertEqual(stale_asset["representative_image"], "")

            source.write_bytes(b"source-v1")
            cross_asset = {
                **base_asset,
                "id": "scene_6",
                "refinement_kind": "clean_background",
                "refined_source_image": str(refined),
                "refinement_provenance": provenance,
            }
            _mode2_apply_asset_contract_fields([cross_asset], [], {})
            self.assertEqual(cross_asset["refinement_provenance_status"], "asset_mismatch")
            self.assertEqual(cross_asset["representative_image"], "")

    def test_existing_refined_output_needs_matching_sidecar_before_attach(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "scene_5_source.jpg"
            source.write_bytes(b"scene-five-source")
            refined_dir = root / "assets" / "refined" / "scene_5"
            refined_dir.mkdir(parents=True)
            refined = refined_dir / "old_scene_5_clean_bg.png"
            refined.write_bytes(b"old-cross-scene-output")
            asset = {
                "id": "scene_5",
                "kind": "scene",
                "source_segment_ids": ["U005"],
                "keyframes": [{"path": str(source), "score": 1.0, "metrics": {}}],
            }

            changed = _mode2_attach_existing_refined_outputs(root, asset)

            self.assertTrue(changed)
            self.assertNotIn("refined_source_image", asset)
            self.assertEqual(asset["refinement_provenance_status"], "missing_provenance")
            self.assertEqual(asset["refinement_status"], "quarantined")

            provenance = _mode2_refinement_provenance_payload(
                asset,
                source_path=source,
                outputs={"refined_source_image": refined},
                job_id="job-new",
                prompt_id="prompt-new",
                method="grounding_dino_sam_lama",
                kind="clean_background",
            )
            sidecar = refined_dir / "job-new_scene_5_scene_provenance.json"
            sidecar.write_text(json.dumps(provenance, ensure_ascii=False), encoding="utf-8")

            changed = _mode2_attach_existing_refined_outputs(root, asset)

            self.assertTrue(changed)
            self.assertEqual(asset["refined_source_image"], str(refined))
            self.assertEqual(asset["refinement_provenance_status"], "verified")
            self.assertEqual(asset["refinement_status"], "ready")

    def test_verified_scene_refinement_can_be_representative(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jpg"
            refined = root / "clean_bg.png"
            source.write_bytes(b"source")
            refined.write_bytes(b"clean-background")
            asset = {
                "id": "scene_verified",
                "kind": "scene",
                "source_segment_ids": ["U001"],
                "keyframes": [{"path": str(source), "score": 1.0, "metrics": {}}],
                "refinement_status": "ready",
                "refinement_kind": "clean_background",
                "refined_source_image": str(refined),
            }
            asset["refinement_provenance"] = _mode2_refinement_provenance_payload(
                asset,
                source_path=source,
                outputs={"refined_source_image": refined},
                job_id="job-ok",
                prompt_id="prompt-ok",
                method="grounding_dino_sam_lama",
                kind="clean_background",
            )

            _mode2_apply_asset_contract_fields([asset], [], {})

        self.assertEqual(asset["refinement_provenance_status"], "verified")
        self.assertEqual(asset["representative_image"], str(refined))
        self.assertEqual(asset["representative_status"], "ready")
        self.assertTrue(asset["representative_is_clean"])

    def test_scene_dedupe_rewrites_shots_then_mixed_canonical_splits_by_time(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            room_a = root / "room_a.png"
            room_a_copy = root / "room_a_copy.png"
            room_b = root / "room_b.png"
            room_b_copy = root / "room_b_copy.png"
            self._write_dhash_pattern(room_a)
            self._write_dhash_pattern(room_a_copy)
            self._write_dhash_pattern(room_b, reverse=True)
            self._write_dhash_pattern(room_b_copy, reverse=True)
            data = {
                "assets": [
                    {
                        "id": "scene_1",
                        "kind": "scene",
                        "name": "mixed room",
                        "scene_mixed_visual_groups": True,
                        "target_image": "old-target.png",
                        "refined_source_image": "old-refined.png",
                        "refinement_provenance": {"asset_id": "scene_1"},
                        "source_segment_ids": ["U001", "U002"],
                        "used_shots": ["S001", "S002"],
                        "keyframes": [
                            {"time": 1.0, "path": str(room_a), "score": 0.9, "metrics": {}},
                            {"time": 6.0, "path": str(room_b), "score": 0.8, "metrics": {}},
                        ],
                    },
                    {
                        "id": "scene_2",
                        "kind": "scene",
                        "name": "different name must not matter",
                        "scene_mixed_visual_groups": True,
                        "source_segment_ids": ["U001", "U002"],
                        "used_shots": ["S001", "S002"],
                        "keyframes": [
                            {"time": 1.0, "path": str(room_a_copy), "score": 0.7, "metrics": {}},
                            {"time": 6.0, "path": str(room_b_copy), "score": 0.6, "metrics": {}},
                        ],
                    },
                ],
                "shots": [
                    {"segment_id": "S001", "start": 0.0, "end": 4.0, "source_segment_ids": ["U001"], "asset_ids": ["scene_1", "scene_2"]},
                    {"segment_id": "S002", "start": 4.0, "end": 8.0, "source_segment_ids": ["U002"], "asset_ids": ["scene_2"]},
                ],
            }

            summary = _mode2_dedupe_scene_assets(data)

            self.assertTrue(summary["changed"])
            self.assertEqual(summary["aliases"], {"scene_2": "scene_1"})
            self.assertEqual([asset["id"] for asset in data["assets"]], ["scene_1"])
            self.assertEqual(data["shots"][0]["asset_ids"], ["scene_1"])
            self.assertEqual(data["shots"][1]["asset_ids"], ["scene_1"])

            split = _mode2_split_scene_asset_in_data(data, "scene_1")

        self.assertEqual(split["children_count"], 2)
        parent = next(asset for asset in data["assets"] if asset["id"] == "scene_1")
        children = [asset for asset in data["assets"] if asset.get("split_parent_asset_id") == "scene_1"]
        self.assertTrue(parent["superseded"])
        self.assertEqual(parent["manual_asset_status"], "ignored")
        self.assertEqual(len(children), 2)
        self.assertEqual(children[0]["split_assigned_shot_ids"], ["S001"])
        self.assertEqual(children[0]["source_segment_ids"], ["U001"])
        self.assertEqual(children[1]["split_assigned_shot_ids"], ["S002"])
        self.assertEqual(children[1]["source_segment_ids"], ["U002"])
        self.assertEqual(data["shots"][0]["asset_ids"], [children[0]["id"]])
        self.assertEqual(data["shots"][1]["asset_ids"], [children[1]["id"]])
        for child in children:
            self.assertEqual(child["target_image"], "")
            self.assertNotIn("refined_source_image", child)
            self.assertNotIn("refinement_provenance", child)

    def test_scene_split_failure_does_not_mutate_data(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame_a = root / "same_a.png"
            frame_b = root / "same_b.png"
            self._write_dhash_pattern(frame_a, bands=True)
            self._write_dhash_pattern(frame_b, bands=True)
            data = {
                "assets": [{
                    "id": "scene_1",
                    "kind": "scene",
                    "scene_mixed_visual_groups": True,
                    "keyframes": [
                        {"time": 1.0, "path": str(frame_a), "metrics": {}},
                        {"time": 2.0, "path": str(frame_b), "metrics": {}},
                    ],
                }],
                "shots": [{"segment_id": "S001", "start": 0.0, "end": 4.0, "asset_ids": ["scene_1"]}],
            }
            before = json.loads(json.dumps(data))

            with self.assertRaisesRegex(ValueError, "not_splittable"):
                _mode2_split_scene_asset_in_data(data, "scene_1")

        self.assertEqual(data, before)

    def test_batch_split_keeps_success_when_another_mixed_scene_is_skipped(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store_path = root / "assets" / "storyboard_assets.json"
            store_path.parent.mkdir(parents=True)
            room_a = root / "room_a.png"
            room_b = root / "room_b.png"
            same_a = root / "same_a.png"
            same_b = root / "same_b.png"
            self._write_dhash_pattern(room_a)
            self._write_dhash_pattern(room_b, reverse=True)
            self._write_dhash_pattern(same_a, bands=True)
            self._write_dhash_pattern(same_b, bands=True)
            store_path.write_text(json.dumps({
                "video_path": "",
                "assets": [
                    {
                        "id": "scene_1",
                        "kind": "scene",
                        "scene_mixed_visual_groups": True,
                        "source_segment_ids": ["U001", "U002"],
                        "keyframes": [
                            {"time": 1.0, "path": str(room_a), "metrics": {}},
                            {"time": 6.0, "path": str(room_b), "metrics": {}},
                        ],
                    },
                    {
                        "id": "scene_2",
                        "kind": "scene",
                        "scene_mixed_visual_groups": True,
                        "source_segment_ids": ["U003"],
                        "keyframes": [
                            {"time": 10.0, "path": str(same_a), "metrics": {}},
                            {"time": 11.0, "path": str(same_b), "metrics": {}},
                        ],
                    },
                ],
                "shots": [
                    {"segment_id": "S001", "start": 0.0, "end": 4.0, "source_segment_ids": ["U001"], "asset_ids": ["scene_1"]},
                    {"segment_id": "S002", "start": 4.0, "end": 8.0, "source_segment_ids": ["U002"], "asset_ids": ["scene_1"]},
                    {"segment_id": "S003", "start": 9.0, "end": 12.0, "source_segment_ids": ["U003"], "asset_ids": ["scene_2"]},
                ],
            }), encoding="utf-8")

            result = _split_storyboard_mode2_scenes({"project_dir": str(root), "all_mixed": True})

        self.assertEqual(result["split_count"], 1)
        self.assertEqual(result["children_count"], 2)
        self.assertEqual([item["status"] for item in result["results"]].count("split"), 1)
        self.assertEqual([item["status"] for item in result["results"]].count("skipped"), 1)
        self.assertIn("assets", result)
        self.assertIn("segments", result)


if __name__ == "__main__":
    unittest.main()
