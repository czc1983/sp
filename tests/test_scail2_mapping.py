from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from spvideo.scail2_client import Scail2Client
from spvideo.scail2_client import _content_addressed_name
from spvideo.models import VideoMeta
from web_ui.server import (
    _default_scail2_positive_prompt,
    _resolve_scail2_positive_prompt,
    _default_scail2_sam_text,
    _resolve_scail2_sam_text,
    _resolve_scail2_role_pairs,
)


class Scail2WorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = Scail2Client("http://example.invalid")

    def test_target_images_are_subjects_not_extra_references(self) -> None:
        workflow = self.client._patch_workflow(
            self.client._build_template(),
            "source.mp4",
            ["new_left.png", "new_right.png"],
            [],
            "the person",
            "two people talking",
            512,
            896,
        )

        pack = workflow["50"]["inputs"]
        self.assertEqual(pack["subject_count"], 2)
        self.assertEqual(pack["reference_count"], 0)
        self.assertEqual(pack["subject_1_image"], ["subject_resize_0", 0])
        self.assertEqual(pack["subject_2_image"], ["subject_resize_1", 0])
        self.assertNotIn("reference_1", pack)
        self.assertEqual(workflow["subject_load_0"]["inputs"]["image"], "new_left.png")
        self.assertEqual(workflow["subject_load_1"]["inputs"]["image"], "new_right.png")
        self.assertEqual(workflow["subject_resize_0"]["inputs"]["width"], 512)
        self.assertEqual(workflow["subject_resize_0"]["inputs"]["height"], 896)
        self.assertEqual(workflow["subject_resize_0"]["inputs"]["divisible_by"], 16)
        self.assertEqual(workflow["32"]["inputs"]["max_objects"], 2)
        self.assertEqual(workflow["32"]["inputs"]["detect_interval"], 2)
        self.assertEqual(workflow["51"]["inputs"]["max_objects"], 2)
        self.assertNotIn("role_repair_0_generate", workflow)

    def test_three_subjects_repair_interior_role_from_mapped_raw_track(self) -> None:
        workflow = self.client._patch_workflow(
            self.client._build_template(),
            "source.mp4",
            ["left.png", "middle.png", "right.png"],
            [],
            "people",
            "three people talking",
            512,
            896,
            driving_object_indices=[2, 1, 0],
            subject_appearance_hints=[" blue outfit", " yellow outfit", " green outfit"],
        )

        self.assertEqual(workflow["50"]["inputs"]["subject_count"], 3)
        self.assertEqual(workflow["32"]["inputs"]["max_objects"], 3)
        self.assertEqual(workflow["32"]["inputs"]["detect_interval"], 2)
        self.assertEqual(workflow["51"]["inputs"]["max_objects"], 3)
        prefix = "role_repair_1"
        repair = workflow[f"{prefix}_generate"]
        self.assertEqual(
            workflow[f"{prefix}_all_frames_mask"]["class_type"],
            "SAM3_TrackToMask",
        )
        self.assertEqual(
            workflow[f"{prefix}_all_frames_mask"]["inputs"]["track_data"],
            ["32", 0],
        )
        self.assertEqual(
            workflow[f"{prefix}_all_frames_mask"]["inputs"]["object_indices"],
            "1",
        )
        self.assertEqual(
            workflow[f"{prefix}_initial_mask"]["inputs"]["mask"],
            [f"{prefix}_all_frames_mask", 0],
        )
        self.assertEqual(
            workflow[f"{prefix}_driving_track"]["inputs"]["initial_mask"],
            [f"{prefix}_initial_mask", 0],
        )
        self.assertEqual(workflow[f"{prefix}_driving_track"]["inputs"]["detect_interval"], 1)
        self.assertEqual(
            workflow[f"{prefix}_reference_track"]["inputs"]["images"],
            ["subject_resize_1", 0],
        )
        self.assertEqual(repair["class_type"], "SCAIL2SimpleVideo")
        self.assertEqual(repair["inputs"]["positive"], [f"{prefix}_positive", 0])
        self.assertEqual(repair["inputs"]["sigmas"], [f"{prefix}_scheduler", 0])
        self.assertEqual(repair["inputs"]["reference_image"], ["subject_resize_1", 0])
        self.assertEqual(repair["inputs"]["chunk_frames"], 49)
        self.assertEqual(repair["inputs"]["context_frames"], 49)
        self.assertEqual(repair["inputs"]["context_overlap_frames"], 12)
        self.assertEqual(repair["inputs"]["driving_track_data"], [f"{prefix}_driving_track", 0])
        self.assertEqual(repair["inputs"]["reference_track_data"], [f"{prefix}_reference_track", 0])
        self.assertEqual(workflow[f"{prefix}_scheduler"]["inputs"]["steps"], 6)
        self.assertEqual(
            workflow[f"{prefix}_constrained_mask"]["inputs"]["destination"],
            [f"{prefix}_all_frames_mask", 0],
        )
        self.assertEqual(
            workflow[f"{prefix}_constrained_mask"]["inputs"]["source"],
            [f"{prefix}_tracked_mask", 0],
        )
        self.assertEqual(
            workflow[f"{prefix}_soft_mask"]["inputs"]["mask"],
            [f"{prefix}_constrained_mask", 0],
        )
        self.assertIn(
            "exact face, hair, apparent age, body shape, and clothing",
            workflow[f"{prefix}_positive"]["inputs"]["text"],
        )
        self.assertIn("yellow outfit", workflow[f"{prefix}_positive"]["inputs"]["text"])
        self.assertEqual(workflow[f"{prefix}_composite"]["inputs"]["destination"], ["40", 0])
        self.assertEqual(workflow["43"]["inputs"]["images"], [f"{prefix}_composite", 0])

    def test_interior_role_repair_keeps_face_closeups_out_of_full_body_reference(self) -> None:
        workflow = self.client._patch_workflow(
            self.client._build_template(),
            "source.mp4",
            ["left.png", "middle.png", "right.png"],
            [],
            "people",
            "three people talking",
            512,
            896,
            driving_object_indices=[2, 1, 0],
            subject_extra_ref_names=[[], ["middle_face_closeup.png"], []],
        )

        prefix = "role_repair_1"
        self.assertNotIn(f"{prefix}_reference_batch", workflow)
        self.assertEqual(
            workflow[f"{prefix}_generate"]["inputs"]["reference_image"],
            ["subject_resize_1", 0],
        )
        self.assertEqual(
            workflow[f"{prefix}_reference_track"]["inputs"]["images"],
            ["subject_resize_1", 0],
        )
        self.assertIn("single full-body reference", workflow[f"{prefix}_positive"]["inputs"]["text"])

    def test_multiple_interior_repairs_use_each_position_raw_mapping(self) -> None:
        workflow = self.client._patch_workflow(
            self.client._build_template(),
            "source.mp4",
            ["left.png", "inner_left.png", "inner_right.png", "right.png"],
            [],
            "person",
            "four people talking",
            512,
            896,
            driving_object_indices=[3, 2, 1, 0],
        )

        self.assertEqual(
            workflow["role_repair_1_all_frames_mask"]["inputs"]["object_indices"],
            "2",
        )
        self.assertEqual(
            workflow["role_repair_2_all_frames_mask"]["inputs"]["object_indices"],
            "1",
        )
        self.assertEqual(
            workflow["role_repair_2_composite"]["inputs"]["destination"],
            ["role_repair_1_composite", 0],
        )
        self.assertEqual(
            workflow["43"]["inputs"]["images"],
            ["role_repair_2_composite", 0],
        )

    def test_colored_mask_workflow_binds_reference_collage(self) -> None:
        workflow = self.client._patch_workflow(
            self.client._build_colored_mask_template(),
            "source.mp4",
            ["left.png", "middle.png", "right.png"],
            [],
            "person",
            "three people talking",
            512,
            896,
            video_window={"force_rate": 24, "frame_load_cap": 74},
            reference_collage_name="collage.png",
        )

        self.assertEqual(workflow["ref_load"]["inputs"]["image"], "collage.png")
        self.assertEqual(workflow["ref_resize"]["inputs"]["width"], 1536)
        self.assertEqual(workflow["ref_resize"]["inputs"]["height"], 896)
        self.assertEqual(workflow["21"]["inputs"]["image"], ["clip_ref_batch", 0])
        self.assertEqual(workflow["clip_ref_batch"]["class_type"], "ImageBatchMulti")
        self.assertEqual(workflow["clip_ref_batch"]["inputs"]["inputcount"], 3)
        self.assertEqual(workflow["clip_ref_batch"]["inputs"]["image_1"], ["subject_resize_0", 0])
        self.assertEqual(workflow["clip_ref_batch"]["inputs"]["image_2"], ["subject_resize_1", 0])
        self.assertEqual(workflow["clip_ref_batch"]["inputs"]["image_3"], ["subject_resize_2", 0])
        self.assertEqual(workflow["32"]["inputs"]["max_objects"], 3)
        self.assertEqual(workflow["33"]["inputs"]["max_objects"], 3)
        self.assertEqual(workflow["34"]["class_type"], "SCAIL2ColoredMask")
        self.assertEqual(workflow["34"]["inputs"]["object_indices"], "0,1,2")
        self.assertEqual(workflow["34"]["inputs"]["driving_track_data"], ["32", 0])
        self.assertEqual(workflow["34"]["inputs"]["ref_track_data"], ["33", 0])
        self.assertEqual(workflow["40"]["class_type"], "WanSCAILToVideo")
        self.assertEqual(workflow["40"]["inputs"]["pose_video_mask"], ["34", 0])
        self.assertEqual(workflow["40"]["inputs"]["reference_image_mask"], ["34", 1])
        self.assertEqual(workflow["40"]["inputs"]["previous_frame_count"], 5)
        self.assertEqual(workflow["2"]["inputs"]["frame_load_cap"], 74)
        self.assertEqual(workflow["40"]["inputs"]["length"], 74)

    def test_colored_mask_workflow_caps_to_single_chunk(self) -> None:
        workflow = self.client._patch_workflow(
            self.client._build_colored_mask_template(),
            "source.mp4",
            ["left.png", "middle.png", "right.png"],
            [],
            "person",
            "three people talking",
            512,
            896,
            video_window={"force_rate": 24, "frame_load_cap": 120},
            reference_collage_name="collage.png",
        )

        self.assertEqual(workflow["2"]["inputs"]["frame_load_cap"], 81)
        self.assertEqual(workflow["40"]["inputs"]["length"], 81)

    def test_colored_mask_clip_batch_includes_subject_extras(self) -> None:
        workflow = self.client._patch_workflow(
            self.client._build_colored_mask_template(),
            "source.mp4",
            ["left.png", "middle.png", "right.png"],
            [],
            "person",
            "three people talking",
            512,
            896,
            subject_extra_ref_names=[["left_face.png"], [], ["right_face.png"]],
            reference_collage_name="collage.png",
        )

        batch = workflow["clip_ref_batch"]["inputs"]
        self.assertEqual(batch["inputcount"], 5)
        self.assertEqual(batch["image_1"], ["subject_resize_0", 0])
        self.assertEqual(batch["image_2"], ["subject_extra_resize_0_0", 0])
        self.assertEqual(batch["image_3"], ["subject_resize_1", 0])
        self.assertEqual(batch["image_4"], ["subject_resize_2", 0])
        self.assertEqual(batch["image_5"], ["subject_extra_resize_2_0", 0])

    def test_wananimate_v2_workflow_builds_isolated_pose_and_clip_batches(self) -> None:
        workflow = self.client._patch_workflow(
            self.client._build_wananimate_scail2_template(),
            "source.mp4",
            ["left.png", "middle.png", "right.png"],
            [],
            "person",
            "three people talking",
            512,
            896,
            video_window={"force_rate": 24, "frame_load_cap": 120},
            subject_extra_ref_names=[["left_face.png"], [], ["right_face.png"]],
            reference_collage_name="collage.png",
        )

        self.assertEqual(workflow["ref_load"]["inputs"]["image"], "collage.png")
        self.assertEqual(workflow["ref_resize"]["inputs"]["width"], 1536)
        self.assertEqual(workflow["2"]["inputs"]["frame_load_cap"], 81)
        subject_batch = workflow["subject_ref_batch"]["inputs"]
        self.assertEqual(workflow["subject_ref_batch"]["class_type"], "ImageBatchMultiV2")
        self.assertEqual(subject_batch["inputcount"], 3)
        self.assertEqual(subject_batch["image_1"], ["subject_resize_0", 0])
        self.assertEqual(subject_batch["image_2"], ["subject_resize_1", 0])
        self.assertEqual(subject_batch["image_3"], ["subject_resize_2", 0])
        clip_batch = workflow["clip_ref_batch"]["inputs"]
        self.assertEqual(workflow["clip_ref_batch"]["class_type"], "ImageBatchMultiV2")
        self.assertEqual(clip_batch["image_1"], ["ref_resize", 0])
        self.assertEqual(clip_batch["image_2"], ["subject_ref_batch", 0])
        self.assertEqual(clip_batch["inputcount"], 2)
        self.assertNotIn("image_3", clip_batch)
        self.assertEqual(workflow["subject_extra_load_0_0"]["inputs"]["image"], "left_face.png")
        self.assertEqual(workflow["subject_extra_load_2_0"]["inputs"]["image"], "right_face.png")
        self.assertEqual(workflow["34"]["class_type"], "SCAIL2ColoredMaskV2")
        self.assertEqual(workflow["34"]["inputs"]["object_indices"], "0,1,2")
        self.assertNotIn("prefix_track_data", workflow["34"]["inputs"])
        self.assertEqual(workflow["34"]["inputs"]["prefix_mask_mode"], "Multi Image Single Color")
        self.assertIs(workflow["34"]["inputs"]["replacement_mode"], False)
        self.assertNotIn("36", workflow)
        self.assertEqual(workflow["embeds"]["class_type"], "WanAnimatePlus SCAIL_2 Embeds")
        self.assertNotIn("prefix_frames", workflow["embeds"]["inputs"])
        self.assertNotIn("prefix_mask", workflow["embeds"]["inputs"])
        self.assertEqual(workflow["embeds"]["inputs"]["pose_images"], ["black_pose_video", 0])
        self.assertEqual(workflow["embeds"]["inputs"]["pose_image_mask"], ["34", 0])
        self.assertEqual(workflow["embeds"]["inputs"]["reference_image_mask"], ["34", 1])
        self.assertIs(workflow["embeds"]["inputs"]["prefix_alpha_crop"], False)
        self.assertIs(workflow["embeds"]["inputs"]["preserve_main_ref_background"], False)
        self.assertIs(workflow["embeds"]["inputs"]["replacement_mode"], False)
        self.assertEqual(workflow["embeds"]["inputs"]["num_frames"], 81)
        self.assertEqual(workflow["embeds"]["inputs"]["pose_strength"], 1.0)
        self.assertEqual(workflow["embeds"]["inputs"]["ref_strength"], 1.0)
        self.assertEqual(workflow["source_people_mask"]["class_type"], "SAM3_TrackToMask")
        self.assertEqual(workflow["source_people_mask"]["inputs"]["track_data"], ["32", 0])
        self.assertEqual(workflow["black_pose_batch"]["inputs"]["width"], 512)
        self.assertEqual(workflow["black_pose_batch"]["inputs"]["height"], 896)
        self.assertEqual(workflow["black_pose_batch"]["inputs"]["batch_size"], 81)
        self.assertEqual(workflow["black_pose_video"]["inputs"]["destination"], ["black_pose_batch", 0])
        self.assertEqual(workflow["black_pose_video"]["inputs"]["source"], ["3", 0])
        self.assertEqual(workflow["black_pose_video"]["inputs"]["mask"], ["source_people_mask", 0])
        self.assertNotIn("two_phase", workflow)
        self.assertEqual(workflow["sampler"]["inputs"]["image_embeds"], ["embeds", 0])
        self.assertEqual(workflow["sampler"]["inputs"]["steps"], 8)
        self.assertEqual(workflow["43"]["inputs"]["images"], ["decode", 0])
        self.assertIn("added black curtain", workflow["text"]["inputs"]["negative_prompt"])
        self.assertIn("new choreography", workflow["text"]["inputs"]["negative_prompt"])
        self.assertIn("wrong age", workflow["text"]["inputs"]["negative_prompt"])
        self.assertIn("slimmed body", workflow["text"]["inputs"]["negative_prompt"])
        self.assertEqual(workflow["mask_pose_video"]["inputs"]["images"], ["34", 0])
        self.assertEqual(workflow["mask_pose_video"]["inputs"]["frame_rate"], 24)
        self.assertEqual(workflow["mask_reference_save"]["inputs"]["images"], ["34", 1])
        self.assertNotIn("mask_prefix_save", workflow)

    def test_wananimate_mask_workflow_uses_mapped_object_indices(self) -> None:
        workflow = self.client._patch_workflow(
            self.client._build_wananimate_scail2_template(),
            "source.mp4",
            ["left.png", "middle.png", "right.png"],
            [],
            "person",
            "mask inspection only",
            512,
            896,
            reference_collage_name="collage.png",
            driving_object_indices=[2, 1, 0],
        )

        self.assertEqual(workflow["34"]["inputs"]["object_indices"], "2,1,0")

    def test_sam3_initial_mask_seed_uses_detect_points_and_bbox(self) -> None:
        workflow = self.client._patch_workflow(
            self.client._build_wananimate_scail2_template(),
            "source.mp4",
            ["left.png", "middle.png", "right.png"],
            [],
            "a single human person, full body",
            "mask inspection only",
            512,
            896,
            reference_collage_name="collage.png",
        )

        self.client._add_sam3_initial_mask_seed(
            workflow,
            [(0.24, 0.45), (0.52, 0.43), (0.78, 0.47)],
            width=512,
            height=896,
        )

        self.assertEqual(workflow["sam_seed_frame"]["class_type"], "ImageFromBatch")
        self.assertEqual(workflow["sam_seed_detect_0"]["class_type"], "SAM3_Detect")
        self.assertEqual(workflow["sam_seed_detect_0"]["inputs"]["image"], ["sam_seed_frame", 0])
        self.assertEqual(workflow["sam_seed_detect_0"]["inputs"]["bboxes"], ["sam_seed_bbox_0", 0])
        self.assertEqual(workflow["sam_seed_positive_0"]["class_type"], "PrimitiveString")
        self.assertIn("\"x\"", workflow["sam_seed_positive_0"]["inputs"]["value"])
        self.assertEqual(workflow["sam_seed_initial_mask_batch"]["class_type"], "MaskBatchMulti")
        self.assertEqual(workflow["sam_seed_initial_mask_batch"]["inputs"]["inputcount"], 3)
        self.assertEqual(workflow["32"]["inputs"]["initial_mask"], ["sam_seed_initial_mask_batch", 0])

    def test_sam3_initial_mask_seed_accepts_freehand_shape(self) -> None:
        workflow = self.client._patch_workflow(
            self.client._build_wananimate_scail2_template(),
            "source.mp4",
            ["left.png", "right.png"],
            [],
            "a single human person, full body",
            "mask inspection only",
            512,
            896,
            reference_collage_name="collage.png",
        )

        seed_items = [
            {"shape": {"type": "freehand", "points": [(0.2, 0.2), (0.42, 0.22), (0.4, 0.7), (0.18, 0.68)]}},
            {"point": (0.72, 0.45)},
        ]
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            self.client,
            "upload_file",
            return_value="uploaded_seed.png",
        ) as upload:
            self.client._add_sam3_mixed_initial_mask_seed(
                workflow,
                seed_items,
                width=512,
                height=896,
                output_dir=Path(tmpdir),
            )

        upload.assert_called_once()
        self.assertEqual(workflow["sam_seed_shape_load_0"]["class_type"], "LoadImage")
        self.assertEqual(workflow["sam_seed_initial_mask_batch"]["inputs"]["mask_1"], ["sam_seed_shape_load_0", 1])
        self.assertEqual(workflow["sam_seed_initial_mask_batch"]["inputs"]["mask_2"], ["sam_seed_detect_1", 0])
        self.assertEqual(workflow["32"]["inputs"]["initial_mask"], ["sam_seed_initial_mask_batch", 0])

    def test_sam3_freehand_shapes_use_independent_tracks(self) -> None:
        workflow = self.client._patch_workflow(
            self.client._build_wananimate_scail2_template(),
            "source.mp4",
            ["left.png", "right.png"],
            [],
            "a single human person, full body",
            "mask inspection only",
            512,
            896,
            reference_collage_name="collage.png",
        )

        seed_items = [
            {"shape": {"type": "freehand", "points": [(0.2, 0.2), (0.42, 0.22), (0.4, 0.7), (0.18, 0.68)]}},
            {"shape": {"type": "freehand", "points": [(0.55, 0.18), (0.78, 0.2), (0.76, 0.72), (0.54, 0.7)]}},
        ]
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            self.client,
            "upload_file",
            side_effect=["seed_left.png", "seed_right.png"],
        ):
            save_ids = self.client._add_sam3_independent_identity_tracks(
                workflow,
                seed_items,
                width=512,
                height=896,
                output_dir=Path(tmpdir),
            )

        self.assertEqual(save_ids, ["preview_track_0_save", "preview_track_1_save"])
        self.assertEqual(workflow["sam_seed_track_0"]["class_type"], "SAM3_VideoTrack")
        self.assertEqual(workflow["sam_seed_track_0"]["inputs"]["max_objects"], 1)
        self.assertEqual(workflow["sam_seed_track_0"]["inputs"]["initial_mask"], ["sam_seed_shape_load_0", 1])
        self.assertEqual(workflow["preview_track_0_mask"]["inputs"]["track_data"], ["sam_seed_track_0", 0])
        self.assertEqual(workflow["preview_track_1_mask"]["inputs"]["track_data"], ["sam_seed_track_1", 0])

    def test_sparse_sam3_track_masks_hold_last_nonempty_frame(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "mask_00001.png"
            second = root / "mask_00002.png"
            third = root / "mask_00003.png"
            valid = Image.new("L", (16, 16), 0)
            for x in range(4, 12):
                for y in range(4, 12):
                    valid.putpixel((x, y), 255)
            valid.save(first)
            Image.new("L", (16, 16), 0).save(second)
            Image.new("L", (16, 16), 0).save(third)

            repaired = self.client._hold_last_nonempty_masks([[first, second, third]])

            self.assertEqual(repaired, 2)
            self.assertGreater(self.client._mask_nonzero_ratio(second), 0.1)
            self.assertGreater(self.client._mask_nonzero_ratio(third), 0.1)

    def test_wananimate_mask_only_prunes_generation_nodes(self) -> None:
        workflow = self.client._patch_workflow(
            self.client._build_wananimate_scail2_template(),
            "source.mp4",
            ["left.png", "middle.png", "right.png"],
            [],
            "person",
            "mask inspection only",
            512,
            896,
            reference_collage_name="collage.png",
        )

        pruned = self.client._prune_workflow_for_outputs(
            workflow,
            ["mask_pose_video", "mask_reference_save"],
        )

        self.assertIn("mask_pose_video", pruned)
        self.assertIn("mask_reference_save", pruned)
        self.assertIn("34", pruned)
        self.assertIn("32", pruned)
        self.assertIn("33", pruned)
        self.assertNotIn("36", pruned)
        self.assertNotIn("subject_ref_batch", pruned)
        self.assertNotIn("43", pruned)
        self.assertNotIn("sample", pruned)
        self.assertNotIn("decode", pruned)
        self.assertNotIn("wan_model", pruned)
        self.assertNotIn("embeds", pruned)

    def test_output_asset_selection_ignores_non_media_values(self) -> None:
        node_output = {
            "text": ["not-media"],
            "images": [
                {"filename": "mask_1.png", "subfolder": "SCAIL2", "type": "output"},
                {"filename": "mask_2.png", "subfolder": "SCAIL2", "type": "output"},
            ],
        }

        assets = self.client._output_assets(node_output)

        self.assertEqual([item["filename"] for item in assets], ["mask_1.png", "mask_2.png"])
        self.assertEqual(self.client._first_output_asset(node_output)["filename"], "mask_1.png")

    def test_final_output_selection_requires_video_media(self) -> None:
        node_output = {
            "images": [{"filename": "debug_mask.png", "subfolder": "SCAIL2", "type": "output"}],
            "gifs": [{"filename": "final.mp4", "subfolder": "SCAIL2", "type": "output"}],
        }

        self.assertEqual(self.client._first_video_asset(node_output)["filename"], "final.mp4")
        self.assertIsNone(self.client._first_video_asset({"images": node_output["images"]}))

    def test_history_poll_tolerates_transient_gateway_error(self) -> None:
        class GatewayResponse:
            status_code = 502

        class GatewaySession:
            def get(self, *_args, **_kwargs):
                return GatewayResponse()

        self.client._session = GatewaySession()
        self.assertEqual(self.client._fetch_history("prompt-id"), {})

    def test_formal_transfer_workflow_drops_debug_mask_outputs(self) -> None:
        workflow = self.client._without_debug_mask_outputs(
            self.client._build_wananimate_scail2_template()
        )

        self.assertIn("43", workflow)
        self.assertNotIn("mask_pose_video", workflow)
        self.assertNotIn("mask_reference_save", workflow)
        self.assertNotIn("mask_prefix_save", workflow)

    def test_multi_person_scail2_defaults_track_person_instances(self) -> None:
        self.assertEqual(_default_scail2_sam_text(3), "a single human person, full body")
        prompt = _default_scail2_positive_prompt([
            {"name": "left"},
            {"name": "middle"},
            {"name": "right"},
        ])
        self.assertIn("3 people", prompt)
        self.assertIn("matching reference subject", prompt)
        self.assertIn("if the source video is a dance scene", prompt)
        self.assertIn("preserve each reference subject's apparent age", prompt)
        self.assertIn("do not add a theater stage", prompt)
        self.assertEqual(_resolve_scail2_sam_text("people", 3), "a single human person, full body")
        self.assertEqual(_resolve_scail2_sam_text("", 3), "a single human person, full body")

    def test_custom_scail2_prompt_keeps_hard_scene_constraints(self) -> None:
        prompt = _resolve_scail2_positive_prompt(
            "a cinematic live-action scene",
            [{"name": "left"}, {"name": "middle"}, {"name": "right"}],
        )

        self.assertIn("a cinematic live-action scene", prompt)
        self.assertIn("keep the original plot moment", prompt)
        self.assertIn("preserve each reference subject's apparent age", prompt)
        self.assertIn("do not add a theater stage", prompt)

    def test_scail2_prompt_adds_parent_body_age_constraints(self) -> None:
        prompt = _default_scail2_positive_prompt([
            {"name": "养母"},
            {"name": "千金"},
            {"name": "养父"},
        ])

        self.assertIn("养父: mature father", prompt)
        self.assertIn("broader solid body", prompt)
        self.assertIn("not a young skinny man", prompt)
        self.assertIn("养母: mature mother", prompt)

    def test_extra_reference_has_its_own_slot(self) -> None:
        workflow = self.client._patch_workflow(
            self.client._build_template(),
            "source.mp4",
            ["new_person.png"],
            ["new_person_back.png"],
            "the person",
            "a person talking",
            512,
            896,
        )

        pack = workflow["50"]["inputs"]
        self.assertEqual(pack["subject_count"], 1)
        self.assertEqual(pack["reference_count"], 1)
        self.assertEqual(pack["subject_1_image"], ["subject_resize_0", 0])
        self.assertEqual(pack["reference_1"], ["extra_ref_resize_0", 0])

    def test_subject_extra_references_are_uploaded_but_not_bound_to_subject_slots(self) -> None:
        workflow = self.client._patch_workflow(
            self.client._build_template(),
            "source.mp4",
            ["left.png", "middle.png"],
            [],
            "person",
            "two people talking",
            512,
            896,
            subject_extra_ref_names=[[], ["middle_back.png", "middle_face.png"]],
        )

        pack = workflow["50"]["inputs"]
        self.assertEqual(pack["subject_count"], 2)
        self.assertEqual(pack["reference_count"], 0)
        self.assertEqual(pack["subject_1_image"], ["subject_resize_0", 0])
        self.assertEqual(pack["subject_2_image"], ["subject_resize_1", 0])
        self.assertNotIn("subject_2_image_1", pack)
        self.assertNotIn("subject_2_image_2", pack)
        self.assertNotIn("subject_1_image_1", pack)
        self.assertEqual(
            workflow["subject_extra_load_1_0"]["inputs"]["image"],
            "middle_back.png",
        )
        self.assertEqual(
            workflow["subject_extra_load_1_1"]["inputs"]["image"],
            "middle_face.png",
        )

    def test_video_window_and_size_are_patched(self) -> None:
        workflow = self.client._patch_workflow(
            self.client._build_template(),
            "source.mp4",
            ["new_person.png"],
            [],
            "the person",
            "a person talking",
            896,
            512,
            video_window={
                "force_rate": 30,
                "frame_load_cap": 180,
                "skip_first_frames": 3,
                "select_every_nth": 2,
            },
        )

        loader = workflow["2"]["inputs"]
        self.assertEqual(loader["force_rate"], 30)
        self.assertEqual(loader["frame_load_cap"], 180)
        self.assertEqual(loader["skip_first_frames"], 3)
        self.assertEqual(loader["select_every_nth"], 2)
        self.assertEqual(workflow["3"]["inputs"]["custom_width"], 896)
        self.assertEqual(workflow["3"]["inputs"]["custom_height"], 512)
        self.assertEqual(workflow["3"]["inputs"]["resolution"], "custom")
        self.assertEqual(workflow["43"]["inputs"]["frame_rate"], 30)

    def test_sampler_presets_patch_known_nodes(self) -> None:
        workflow = self.client._patch_workflow(
            self.client._build_template(),
            "source.mp4",
            ["new_person.png"],
            [],
            "the person",
            "a person talking",
            512,
            896,
            sampler_preset="quality",
        )

        self.assertEqual(workflow["14"]["inputs"]["steps"], 12)
        self.assertEqual(workflow["40"]["inputs"]["chunk_frames"], 81)
        self.assertEqual(workflow["40"]["inputs"]["overlap_frames"], 9)
        self.assertEqual(workflow["40"]["inputs"]["context_overlap_frames"], 24)

    def test_metadata_window_caps_at_available_frames(self) -> None:
        meta = VideoMeta("source.mp4", 2.0, 1920, 1080, 30.0)
        window = self.client._resolve_video_window(meta, {"frame_load_cap": 241})
        self.assertEqual(window["force_rate"], 30)
        self.assertEqual(window["frame_load_cap"], 60)
        self.assertEqual(self.client._normalized_size(1920, 1080, 512, 896), (896, 512))

    def test_upload_names_include_content_hash_to_bust_comfy_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "char_same.png"
            path.write_bytes(b"first")
            first_name = _content_addressed_name(path)
            path.write_bytes(b"second")
            second_name = _content_addressed_name(path)

        self.assertNotEqual(first_name, second_name)
        self.assertTrue(first_name.startswith("char_same_"))
        self.assertTrue(second_name.endswith(".png"))

    def test_reference_collage_is_written_for_multi_subjects(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            refs = []
            for index, color in enumerate(((255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255))):
                path = root / f"ref_{index}.png"
                Image.new("RGBA", (120, 240), color).save(path)
                refs.append(str(path))

            collage = self.client._create_reference_collage(
                refs,
                output_dir=root,
                width=512,
                height=896,
                role_names=["left", "middle", "right"],
            )
            with Image.open(collage) as image:
                self.assertEqual(image.size, (1536, 896))

        self.assertTrue(collage.name.startswith("scail2_reference_collage_left_middle_right_"))

    def test_reference_clothing_hint_uses_dominant_outfit_color(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed")

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "yellow_subject.png"
            Image.new("RGB", (100, 200), (245, 196, 26)).save(image_path)
            hint = self.client._reference_clothing_hint(str(image_path))

        self.assertIn("yellow outfit", hint)


class Scail2RoleMappingTests(unittest.TestCase):
    def test_current_video_positions_sort_targets_left_to_right(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "clip.mp4"
            left_ref = root / "left.png"
            right_ref = root / "right.png"
            for path in (video, left_ref, right_ref):
                path.touch()

            pairs, warning = _resolve_scail2_role_pairs(
                char_names=["right_role", "left_role"],
                characters={
                    "right": {"name": "right_role", "ref_image": str(right_ref)},
                    "left": {"name": "left_role", "ref_image": str(left_ref)},
                },
                originals=[],
                annotations=[
                    {
                        "type": "person",
                        "label_name": "right_role",
                        "video_path": str(video),
                        "time": 0.1,
                        "point": [0.8, 0.5],
                    },
                    {
                        "type": "person",
                        "label_name": "left_role",
                        "video_path": str(video),
                        "time": 0.1,
                        "point": [0.2, 0.5],
                    },
                ],
                video_path=str(video),
            )

        self.assertEqual([pair["name"] for pair in pairs], ["left_role", "right_role"])
        self.assertEqual(warning, "")

    def test_missing_positions_block_unsafe_multi_role_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "clip.mp4"
            first_ref = root / "first.png"
            second_ref = root / "second.png"
            for path in (video, first_ref, second_ref):
                path.touch()

            with self.assertRaisesRegex(ValueError, "多人转绘已阻止"):
                _resolve_scail2_role_pairs(
                    char_names=["first", "second"],
                    characters={
                        "first": {"name": "first", "ref_image": str(first_ref)},
                        "second": {"name": "second", "ref_image": str(second_ref)},
                    },
                    originals=[],
                    annotations=[],
                    video_path=str(video),
                )

    def test_role_appearing_after_first_second_blocks_multi_role_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "clip.mp4"
            first_ref = root / "first.png"
            second_ref = root / "second.png"
            for path in (video, first_ref, second_ref):
                path.touch()

            with self.assertRaisesRegex(ValueError, "第一帧"):
                _resolve_scail2_role_pairs(
                    char_names=["first", "second"],
                    characters={
                        "first": {"name": "first", "ref_image": str(first_ref)},
                        "second": {"name": "second", "ref_image": str(second_ref)},
                    },
                    originals=[],
                    annotations=[
                        {
                            "type": "person",
                            "label_name": "first",
                            "video_path": str(video),
                            "time": 0.0,
                            "point": [0.2, 0.5],
                        },
                        {
                            "type": "person",
                            "label_name": "second",
                            "video_path": str(video),
                            "time": 1.1,
                            "point": [0.8, 0.5],
                        },
                    ],
                    video_path=str(video),
                )


class Scail2UploadTests(unittest.TestCase):
    def test_upload_file_reports_retryable_http_details(self) -> None:
        class FakeResponse:
            def __init__(self) -> None:
                self.status_code = 502
                self.headers = {"content-type": "text/plain"}
                self.text = "bad gateway"

        class FakeSession:
            def __init__(self) -> None:
                self.calls = 0
                self.response = FakeResponse()

            def post(self, *args, **kwargs):  # noqa: ANN002, ANN003
                self.calls += 1
                return self.response

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "clip.mp4"
            path.write_bytes(b"fake-mp4-bytes")

            client = Scail2Client("http://example.invalid")
            fake_session = FakeSession()
            client._session = fake_session

            with mock.patch("spvideo.scail2_client.time.sleep", return_value=None):
                with self.assertRaises(RuntimeError) as ctx:
                    client.upload_file(str(path))

        message = str(ctx.exception)
        self.assertIn("HTTP 502", message)
        self.assertIn("bad gateway", message)
        self.assertEqual(fake_session.calls, 4)


if __name__ == "__main__":
    unittest.main()
