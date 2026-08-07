from __future__ import annotations

from types import SimpleNamespace
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from spvideo.comfy_client import ComfyClient
from web_ui.server import (
    _cancel_h3_job_if_cleared,
    _delete_generation_package_mask_clip,
    _delete_storyboard_mode2_asset,
    _load_storyboard_job_snapshot,
    _mode2_ensure_shot_preview_clips,
    _mode2_renumber_timeline_shots,
    _mode2_split_timeline_shot,
    _required_h3_identity,
    _restore_generation_package_h3_jobs,
    _restore_generation_package_state,
    _storyboard_mode2_shot_subclips,
    _storyboard_build_class_pure_shots,
    _storyboard_enrich_reference_segments_with_understanding,
    _validated_h3_resume_output,
)


ROOT = Path(__file__).resolve().parents[1]


class Mode2OnlyWorkflowTests(unittest.TestCase):
    def test_storyboard_shots_keep_scene_single_multi_as_separate_units(self) -> None:
        reference_segments = [
            {"segment_id": "V001", "start": 0.0, "end": 1.2, "person_count": 0, "is_pure_background": True},
            {"segment_id": "V002", "start": 1.2, "end": 2.7, "person_count": 1},
            {"segment_id": "V003", "start": 2.7, "end": 4.4, "person_count": 2},
            {"segment_id": "V004", "start": 4.4, "end": 5.8, "person_count": 1},
        ]

        shots = _storyboard_build_class_pure_shots(
            reference_segments,
            video_path="source.mp4",
            understanding_status="fallback",
            boundary_hints=[],
        )

        self.assertEqual([shot["shot_class"] for shot in shots], ["scene", "single", "multi", "single"])
        self.assertEqual([shot["shot_class_label"] for shot in shots], ["场景", "单人", "多人", "单人"])
        self.assertEqual([(shot["start"], shot["end"]) for shot in shots], [
            (0.0, 1.2),
            (1.2, 2.7),
            (2.7, 4.4),
            (4.4, 5.8),
        ])

    def test_semantic_characters_do_not_turn_visual_scene_into_person_shot(self) -> None:
        enriched = _storyboard_enrich_reference_segments_with_understanding(
            [{
                "segment_id": "V001",
                "start": 0.0,
                "end": 2.0,
                "person_count": 0,
                "is_pure_background": True,
            }],
            {"scenes": [{"start": 0.0, "end": 2.0, "characters": ["角色A"]}]},
            duration=2.0,
        )

        self.assertEqual(enriched[0]["person_count"], 0)
        self.assertEqual(enriched[0]["shot_class"], "scene")
        self.assertEqual(enriched[0]["shot_class_label"], "场景")

    def test_legacy_mode1_dashboard_removed(self) -> None:
        self.assertFalse((ROOT / "web_ui" / "splitter_dashboard.html").exists())

    def test_server_root_serves_story_dashboard_and_blocks_legacy_routes(self) -> None:
        server = (ROOT / "web_ui" / "server.py").read_text(encoding="utf-8")
        self.assertIn('INDEX = ROOT / "story_generate_dashboard.html"', server)
        self.assertIn('"legacy_mode1_removed"', server)
        self.assertIn('"/api/transfer-segment"', server)
        self.assertIn('"/api/split"', server)
        legacy_block = server.split("legacy_mode1_paths = {", 1)[1].split("}", 1)[0]
        self.assertIn('"/api/storyboard-reference-mask"', legacy_block)
        self.assertIn('"/api/storyboard-reference-white-mask"', legacy_block)
        self.assertIn('"/api/storyboard-reference-expression-mask"', legacy_block)
        self.assertIn('"/api/storyboard-server-transfer"', legacy_block)
        self.assertNotIn('if parsed.path == "/api/storyboard-server-transfer"', server)
        self.assertNotIn("def _run_transfer_job", server)
        self.assertIn('if parsed.path == "/api/generation-package/prepare"', server)
        self.assertIn('if parsed.path == "/api/generation-package/restore"', server)
        self.assertIn('if parsed.path == "/api/generation-package/split-output"', server)
        self.assertIn('if parsed.path == "/api/generation-package/delete-mask-clip"', server)
        self.assertIn("def _prepare_generation_package_video", server)
        self.assertIn("def _restore_generation_package_state", server)
        self.assertIn("def _delete_generation_package_mask_clip", server)
        self.assertIn("generation_state.json", server)
        self.assertIn("maskRuns", server)
        self.assertIn("def _split_generation_package_output", server)

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
        self.assertIn('id="composeAddPackBtn"', html)
        self.assertIn('id="composePackSummary"', html)
        self.assertIn("STORY_COMPOSE_PLAN_KEY_PREFIX", html)
        self.assertIn("function composeCreateSmartPlan", html)
        self.assertIn("function composeAddPack", html)
        self.assertIn("function composeUnpackedShotIds", html)
        self.assertIn("composeDragSourcePackId", html)
        self.assertIn("function saveComposePlanDraft", html)
        self.assertIn("function loadComposePlanDraft", html)
        self.assertIn("function composePackContainingShot", html)
        self.assertIn("function composeDropPlacement", html)
        self.assertIn("function composeHandDropPlacement", html)
        self.assertIn("function composeMoveShotToSelectedPack", html)
        self.assertIn("function bindComposeDetailDrop", html)
        self.assertIn("function resetGenerationRuntimeForConfirmedPlan", html)
        self.assertIn("resetGenerationRuntimeForConfirmedPlan(composeConfirmedPlan.packs)", html)
        self.assertIn("function composePackStackHtml", html)
        self.assertIn("function composePackShotListText", html)
        self.assertIn("function composePackCardTitle", html)
        self.assertIn("compose-pack-shot-list", html)
        self.assertIn("function confirmComposePlan", html)
        self.assertIn('openComposePage({ rebuild: true })', html)
        self.assertIn("5-15秒", html)
        self.assertIn("超过 15 秒", html)
        self.assertIn("卡包架", html)
        self.assertIn("compose-pack-shelf", html)
        self.assertIn("compose-pack-stack-card", html)
        self.assertIn("compose-shot-id", html)
        self.assertIn("白膜包", html)
        self.assertIn('data-compose-board-mode="mask"', html)
        self.assertIn("compose-shot-grid.mask-mode", html)
        self.assertIn("STORY_WHITE_MASK_SHELF_KEY_PREFIX", html)
        self.assertIn("function renderComposeMaskStore", html)
        self.assertIn("function renderComposeMaskPacks", html)
        self.assertIn("function appendWhiteMaskShelfRun", html)
        self.assertIn("function composeWhiteMaskShelfItems", html)
        self.assertIn("function composeWhiteMaskClipCards", html)
        self.assertIn("function composeMaskVideoThumbHtml", html)
        self.assertIn("function composeMaskCardHtml", html)
        self.assertIn("function composeMaskHandCardsHtml", html)
        self.assertIn("function composeMaskPackStackHtml", html)
        self.assertIn("function bindComposeMaskVideoPlayback", html)
        self.assertIn("function pauseOtherComposeMaskVideos", html)
        self.assertIn("function openComposeMaskVideoPreview", html)
        self.assertIn("function closeComposeMaskVideoPreview", html)
        self.assertIn("function bindComposeMaskPreviewChrome", html)
        self.assertIn('document.addEventListener("pointerdown"', html)
        self.assertIn('target.closest("[data-compose-mask-video-thumb]")', html)
        self.assertIn("popover.contains(target)", html)
        self.assertIn("function openWhiteMaskClipDeleteDialog", html)
        self.assertIn("function removeWhiteMaskClipFromShelf", html)
        self.assertIn("function removeWhiteMaskClipFromRuntime", html)
        self.assertIn("function addWhiteMaskClipToManualHand", html)
        self.assertIn("function returnWhiteMaskClipToTable", html)
        self.assertIn("function composeManualWhiteMaskHandId", html)
        self.assertIn("function composeIsManualWhiteMaskItem", html)
        self.assertIn("data-delete-white-mask-clip", html)
        self.assertIn("data-grab-white-mask-clip", html)
        self.assertIn("data-return-white-mask-clip", html)
        self.assertIn("data-return-white-mask-card", html)
        self.assertIn("returnAsClose", html)
        self.assertIn("compose-mask-return-btn", html)
        self.assertIn("white-mask-delete-confirm-overlay", html)
        self.assertIn("/api/generation-package/delete-mask-clip", html)
        self.assertIn("从当前项目物理删除这个白膜 mp4", html)
        self.assertIn("手动白膜包", html)
        self.assertIn("抓到白膜手牌", html)
        self.assertIn("放回白膜池", html)
        self.assertIn("function composeSelectedWhiteMaskShelfItem", html)
        self.assertIn("composeSelectedMaskShelfId", html)
        self.assertIn("renderComposeMaskStore()", html)
        self.assertIn("renderComposeMaskPacks()", html)
        self.assertIn('if (composeBoardMode !== "mask") return;', html)
        self.assertIn('if (composeBoardMode === "mask")', html)
        self.assertIn('id="composePackShelfTitle"', html)
        self.assertIn("白膜卡包架", html)
        self.assertIn("白膜手牌", html)
        self.assertIn("data-compose-mask-card", html)
        self.assertIn("data-compose-mask-video-thumb", html)
        self.assertIn("data-compose-mask-video", html)
        self.assertIn("data-video-path", html)
        self.assertIn("compose-mask-play", html)
        self.assertIn('id="composeMaskVideoPreview"', html)
        self.assertIn('id="composeMaskVideoPreviewPlayer"', html)
        self.assertIn("compose-mask-preview-popover", html)
        self.assertIn("data-mask-id", html)
        self.assertIn("compose-mask-card", html)
        self.assertIn("mask-pack", html)
        self.assertIn("appendWhiteMaskShelfRun(pack.id, run)", html)
        self.assertIn("暂无白膜包。生成白膜后会保存在这里。", html)
        self.assertNotIn('id="composeMaskStoreList"', html)
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
        self.assertIn("function storyboardFrameAlignedCutTime", html)
        self.assertIn("function storyboardLeftRelativeTime", html)
        self.assertIn("function localStoryboardShotCutResult", html)
        self.assertIn("function applyLocalShotCutWorkbench", html)
        self.assertIn("applyLocalShotCutWorkbench(cutTimes", html)
        self.assertIn("function storyboardApiUrl", html)
        self.assertIn('window.location?.protocol === "file:"', html)
        self.assertIn('window.location.replace("http://127.0.0.1:7861/story_generate_dashboard.html")', html)
        self.assertIn('fetch(storyboardApiUrl("/api/pick-path")', html)
        self.assertIn('fetch(storyboardApiUrl("/api/storyboard-draft")', html)
        self.assertIn('fetch(storyboardApiUrl("/api/jobs/" + encodeURIComponent(currentStoryJobId)))', html)
        self.assertIn('await pickPath("file", "#videoPathInput")', html)
        self.assertIn('fetch(storyboardApiUrl("/api/storyboard-shot-subclips")', html)
        self.assertIn("aligned.frameStep", html)
        self.assertIn("virtual_subclips: !save", html)
        self.assertIn("segment_start", html)
        self.assertIn("video_offset", html)
        self.assertIn("media_start", html)
        self.assertIn("parentMediaStart", html)
        self.assertIn("function bindStoryboardSegmentPlayback", html)
        self.assertIn("configureStoryboardSegmentPlayback(sourceVideo, previewStart, previewEnd)", html)
        self.assertIn("setStoryboardVideoSource(generatedVideo, path, previewStart, previewEnd)", html)
        self.assertIn("shotVideoEndTime(seg)", html)
        self.assertIn("storyboardDurationLabel(seg)", html)
        self.assertNotIn("Math.abs(value - rounded) < 0.08", html)
        self.assertIn("last.rect.left + last.rect.width * 0.35", html)
        self.assertIn("selectedItems.concat", html)
        self.assertIn('event.dataTransfer.setData("text/plain"', html)
        self.assertIn("点牌放回镜头池", html)
        self.assertIn("未入包镜头不参与本次生成", html)
        self.assertIn("视频页将严格使用当前卡包，不再自动重排。", html)
        normalize_fn = html.split("function normalizeComposePlanDraft", 1)[1].split("function saveComposePlanDraft", 1)[0]
        self.assertIn("seenInPack", normalize_fn)
        self.assertNotIn("const seen = new Set()", normalize_fn)
        ensure_fn = html.split("function ensureComposePlan", 1)[1].split("function composeThumbHtml", 1)[0]
        self.assertIn("seenInPack", ensure_fn)
        self.assertNotIn("const seen = new Set()", ensure_fn)
        self.assertNotIn("missing.length || stale.length", ensure_fn)
        move_fn = html.split("function composeMoveShotToPack", 1)[1].split("function composeMoveShotToSelectedPack", 1)[0]
        self.assertIn("alreadyInTarget && (beforeId || afterId)", move_fn)
        self.assertNotIn("(composePlan.packs || []).forEach((entry) =>", move_fn)
        return_fn = html.split("function composeReturnShotToBoard", 1)[1].split("function composeClearPlan", 1)[0]
        self.assertIn("packId = composePlan.selectedPackId", return_fn)
        self.assertIn("entry.id === targetPackId", return_fn)
        self.assertLess(html.index('data-module="storyboard"'), html.index('data-module="compose"'))
        self.assertLess(html.index('data-module="compose"'), html.index('data-module="generate"'))

    def test_story_dashboard_uses_compact_poker_image_cards(self) -> None:
        html = (ROOT / "web_ui" / "story_generate_dashboard.html").read_text(encoding="utf-8")
        self.assertIn("function compactMediaLabel", html)
        self.assertIn("aspect-ratio: 2 / 3 !important", html)
        self.assertIn("#assetBoard .asset-card-body .asset-media-primary", html)
        self.assertIn("display: none !important;", html)
        self.assertIn("#assetBoard .asset-card-body .asset-media-secondary", html)
        self.assertIn("repeat(auto-fill, minmax(96px, 108px))", html)
        self.assertIn("max-width: 108px !important", html)
        self.assertIn("object-fit: contain !important", html)
        self.assertIn("width: min(150px, 100%) !important", html)
        self.assertIn(".manual-asset-modal.asset-image-editor-modal .asset-editor-preview > span", html)
        self.assertIn("manual-asset-create-modal", html)
        self.assertIn("grid-template-columns: minmax(0, 1fr) 240px !important", html)
        self.assertIn(".manual-asset-create-modal .manual-asset-upload-preview::after", html)
        self.assertIn("#seedanceImageSlots .slot-preview", html)
        self.assertIn(".subshot-ref-thumb", html)

    def test_video_generation_page_uses_package_workbench(self) -> None:
        html = (ROOT / "web_ui" / "story_generate_dashboard.html").read_text(encoding="utf-8")
        self.assertIn('id="generatePackageWorkbench"', html)
        self.assertIn('id="generationPackageList"', html)
        self.assertIn('id="generationCompareOriginal"', html)
        self.assertIn('id="generationCompareMask"', html)
        self.assertIn('id="generationCompareResult"', html)
        self.assertIn('id="generateWhiteMaskBtn"', html)
        self.assertIn('id="generateResultBtn"', html)
        self.assertIn('id="generateFullPipelineBtn"', html)
        self.assertIn("function renderGenerationPackageWorkbench", html)
        self.assertIn("function startGenerationPackageMask", html)
        self.assertIn("function startGenerationPackageResult", html)
        self.assertIn("function generationPackApiWhiteMaskPrompt", html)
        self.assertIn("function generationActivePromptLabel", html)
        self.assertIn("function generationActivePromptText", html)
        self.assertIn("function generationPromptEditorHtml", html)
        self.assertIn("function generationEffectivePromptText", html)
        self.assertIn("function setGenerationPromptOverride", html)
        self.assertIn("data-generation-prompt-editor", html)
        self.assertIn("data-generation-prompt-input", html)
        self.assertIn("已人工修改，框内内容将发送", html)
        self.assertIn("恢复自动草稿", html)
        self.assertIn("generation-prompt-editor", html)
        self.assertIn('grid-template-areas:', html)
        self.assertIn('"references outputs"', html)
        self.assertIn('"prompt prompt"', html)
        self.assertIn("grid-area: references", html)
        self.assertIn("grid-area: prompt", html)
        self.assertIn("grid-area: outputs", html)
        self.assertIn("grid-template-rows: 104px minmax(165px, 1fr)", html)
        self.assertNotIn("generation-material-info", html)
        self.assertIn("STORY_GENERATION_RUNTIME_KEY_PREFIX", html)
        self.assertIn("function saveGenerationRuntimeDraft", html)
        self.assertIn("function loadGenerationRuntimeDraft", html)
        self.assertIn("function ensureGenerationRuntimeForPacks", html)
        self.assertIn("saveGenerationRuntimeDraft()", html)
        self.assertIn("ensureGenerationRuntimeForPacks(packs)", html)
        self.assertIn("function generationRuntimeHasActiveH3Job", html)
        self.assertIn("function generationRuntimeNeedsH3JobRecovery", html)
        self.assertIn("function resumeGenerationH3Polling", html)
        self.assertIn("resumeGenerationH3Polling(pack)", html)
        self.assertIn('const STORY_RATIO_OPTIONS = ["source", "9:16", "16:9"]', html)
        self.assertIn('if (ratio === "source") return { width: 0, height: 0 };', html)
        self.assertIn("whiteMaskPath", html)
        self.assertIn("maskSegments", html)
        self.assertIn("maskRuns", html)
        self.assertIn("resultRuns", html)
        self.assertIn("function generationRunFromSplitData", html)
        self.assertIn("function appendGenerationRun", html)
        self.assertIn("白膜结果包", html)
        self.assertIn("成片结果包", html)
        self.assertIn("if (run) appendWhiteMaskShelfRun(pack.id, run)", html)
        self.assertIn("generation-output-package", html)
        self.assertIn("resultPath", html)
        self.assertIn("function ensureGenerationPackageSourcePreview", html)
        self.assertIn("正在合成整包预览", html)
        self.assertIn("整包预览合成失败", html)
        self.assertIn("sourcePreparing", html)
        self.assertIn("sourcePreviewFailed", html)
        self.assertIn("generationActivePromptText(pack)", html)
        self.assertIn("generationActivePromptLabel(pack)", html)
        self.assertIn("视频生成提示词", html)
        self.assertIn("框内内容将发送", html)
        self.assertIn("function generationEditorPromptText", html)
        self.assertIn("这是严格的视频人物替换生成任务", html)
        self.assertIn("参考视频已经是一条整合后的连续视频", html)
        pack_prompt_fn = html.split("function generationPackPrompt", 1)[1].split("function generationPackApiWhiteMaskPrompt", 1)[0]
        self.assertNotIn("buildPromptDraft", pack_prompt_fn)
        self.assertNotIn("promptLines", pack_prompt_fn)
        self.assertIn("把视频里的人物全部转换成彩色树脂关节人偶素体", html)
        self.assertIn("声音与原视频完全一致", html)
        white_mask_fn = html.split("function generationPackApiWhiteMaskPrompt", 1)[1].split("function generationPackShotPayload", 1)[0]
        self.assertNotIn("人物参考图绑定", white_mask_fn)
        self.assertNotIn("参考图只负责人物身份和外形轮廓", white_mask_fn)
        self.assertNotIn("禁止继承原视频中的原人物形象", white_mask_fn)
        self.assertNotIn("替换参考视频1中的女方人物", white_mask_fn)
        self.assertNotIn("替换参考视频1中的男方人物", white_mask_fn)
        self.assertNotIn("视频人物白膜重绘任务，不是最终彩色视频生成", html)
        self.assertNotIn("输出目标：把参考视频1(@video1)里的原人物替换为参考图人物", html)
        self.assertNotIn("先把参考图里的人物替换进 @video1 的对应占位", html)
        self.assertNotIn("女方/男方绑定：女方占位始终使用女方参考图", html)
        self.assertNotIn("唯一身份参考", html)
        self.assertNotIn("手部、手臂、局部接触镜头", html)
        self.assertNotIn("核心控制项是动作、接触点、手势、相对位置", html)
        self.assertNotIn("手部不作为独立身份参考", html)
        self.assertNotIn("归属依据是相邻身体、手臂连接、衣袖/服装外轮廓、站位和动作连续性", html)
        self.assertNotIn("不是把参考视频1里的原演员直接涂白", html)
        self.assertNotIn("强制负向", html)
        self.assertNotIn("不要直接涂白", html)
        self.assertNotIn("手部特写", html)
        self.assertIn("function generationPackShotPayload", html)
        self.assertIn("function prepareGenerationPackageSource", html)
        self.assertIn("function splitGenerationPackageOutput", html)
        self.assertIn("function generationSplitClipCardsHtml", html)
        self.assertIn("function generationManualReferenceImages", html)
        self.assertIn("function generationReferenceAssetsForPicker", html)
        self.assertIn("function generationReferencePickerHtml", html)
        self.assertIn("function ensureGenerationPackageManualReferences", html)
        self.assertIn("function resetGenerationManualReferences", html)
        self.assertIn("generationWorkbench.referencePlanKey", html)
        self.assertIn("会上传的参考图", html)
        self.assertIn("可选资产图", html)
        self.assertIn("已清空旧参考图", html)
        self.assertIn("data-generation-add-reference", html)
        self.assertIn("data-generation-remove-reference", html)
        self.assertIn("已选参考图", html)
        self.assertNotIn("function generationPackReferenceImages", html)
        self.assertNotIn("prefer_current_shot_roles: true", html)
        pack_role_fn = html.split("function generationPackRoleAssets", 1)[1].split("function generationManualReferenceImages", 1)[0]
        self.assertNotIn("storyLegacyReferenceSegments", pack_role_fn)
        self.assertNotIn("legacy_reference", pack_role_fn)
        self.assertIn("autoGenerateResult", html)
        self.assertIn("whiteMaskPath", html)
        source_info_fn = html.split("function generationPackSourceInfo", 1)[1].split("function generationPackSourceVideo", 1)[0]
        self.assertIn("runtime.packageSourcePath", source_info_fn)
        self.assertIn("shots.length > 1", source_info_fn)
        self.assertIn("整包", source_info_fn)
        render_fn = html.split("function renderGenerationPackageWorkbench", 1)[1].split("function selectGenerationPackage", 1)[0]
        self.assertIn("sourceInfo.emptyText", render_fn)
        server_restore_fn = html.split("async function restoreGenerationPackageRuntime", 1)[1].split("function ensureGenerationPackageRuntimeRestored", 1)[0]
        self.assertIn("data.h3MaskStateAuthoritative === true", server_restore_fn)
        self.assertIn("data.h3MaskCleared === true", server_restore_fn)
        self.assertIn('clearGenerationH3Poll(id, "mask")', server_restore_fn)
        restore_fn = html.split("function ensureGenerationPackageRuntimeRestored", 1)[1].split("function applyGenerationResultClipsToTimeline", 1)[0]
        self.assertIn("generationRuntimeNeedsH3JobRecovery(runtime)", restore_fn)
        self.assertIn("restoreAttempts < 20", restore_fn)
        self.assertIn("setTimeout", restore_fn)
        self.assertNotIn("runtime.whiteMaskPath || runtime.resultPath", restore_fn)
        mask_fn = html.split("async function startGenerationPackageMask", 1)[1].split("async function startGenerationPackageResult", 1)[0]
        self.assertIn("return startGenerationPackageH3Mask(pack, options)", mask_fn)
        self.assertNotIn("Seedance", mask_fn)
        self.assertNotIn("Scail2", mask_fn)
        result_fn = html.split("async function startGenerationPackageResult", 1)[1].split("async function startGenerationPackageFull", 1)[0]
        self.assertIn("return startGenerationPackageH3Result(pack)", result_fn)
        self.assertNotIn("Seedance", result_fn)
        self.assertNotIn("Scail2", result_fn)
        self.assertNotIn("/api/storyboard-server-transfer", html)
        self.assertNotIn("function startMode2ServerTransfer", html)
        self.assertNotIn("function pollMode2ServerTransferJob", html)
        self.assertNotIn("function startSeedancePackageTaskPolling", html)
        self.assertNotIn("function pollSeedancePackageTask", html)
        self.assertNotIn("SEEDANCE_A_RELAY_CONFIG_KEY", html)
        self.assertNotIn("NUKO_CHANNEL1_CONFIG_KEY", html)
        self.assertNotIn("function syncGenerationPackageToSeedanceDraft", html)
        self.assertNotIn("function loadSeedanceDraft", html)
        self.assertNotIn("function buildSeedancePayload", html)
        self.assertIn('submitGenerationH3Job("/api/mode2/h3-white-mask"', html)
        self.assertIn('submitGenerationH3Job("/api/mode2/h3-charswap"', html)
        h3_mask_fn = html.split("async function startGenerationPackageH3Mask", 1)[1].split("async function startGenerationPackageH3Result", 1)[0]
        self.assertIn("package_id: pack.id", h3_mask_fn)
        self.assertIn("shot_id: entry.shotId", h3_mask_fn)
        self.assertIn("auto_generate_result: jobs.autoGenerateResult === true", h3_mask_fn)
        h3_result_fn = html.split("async function startGenerationPackageH3Result", 1)[1].split("function attachGenerationH3Clips", 1)[0]
        self.assertIn("package_id: pack.id", h3_result_fn)
        self.assertIn("shot_id: entry.shotId", h3_result_fn)
        h3_poll_fn = html.split("function pollGenerationH3Jobs", 1)[1].split("function resumeGenerationH3Polling", 1)[0]
        self.assertIn("generationH3PollInFlight[key] === pollVersion", h3_poll_fn)
        self.assertIn("generationH3PollVersions[key] !== pollVersion", h3_poll_fn)
        self.assertLess(html.index('id="generationCompareOriginal"'), html.index('id="generationCompareMask"'))
        self.assertLess(html.index('id="generationCompareMask"'), html.index('id="generationCompareResult"'))

    def test_generation_package_restore_keeps_all_mask_runs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            shot_a = root / "shot_a.mp4"
            shot_b = root / "shot_b.mp4"
            mask_a = root / "mask_a.mp4"
            mask_b = root / "mask_b.mp4"
            for path in (shot_a, shot_b, mask_a, mask_b):
                path.write_bytes(b"video")
            segments = [
                {"shot_id": "S001", "video_path": str(shot_a), "duration": 1.0, "start": 0.0, "end": 1.0},
                {"shot_id": "S002", "video_path": str(shot_b), "duration": 1.0, "start": 1.0, "end": 2.0},
            ]
            mask_dir = root / "04_AI输出成片" / "generation_packages" / "P001" / "mask"
            mask_dir.mkdir(parents=True)

            def write_mask_run(token: str, source: Path, created_at: float) -> None:
                clips = []
                for index, segment in enumerate(segments, start=1):
                    clip_path = mask_dir / f"mask_{index:02d}_{segment['shot_id']}_{token}.mp4"
                    clip_path.write_bytes(b"clip")
                    clips.append({
                        "shot_id": segment["shot_id"],
                        "path": str(clip_path),
                        "start": segment["start"],
                        "end": segment["end"],
                        "duration": segment["duration"],
                        "original_start": segment["start"],
                        "original_end": segment["end"],
                        "original_duration": segment["duration"],
                        "index": index - 1,
                    })
                manifest = {
                    "package_id": "P001",
                    "run_id": token,
                    "source_path": str(source),
                    "output_role": "mask",
                    "clips": clips,
                    "segments": segments,
                    "source_duration": 2.0,
                    "timeline_duration": 2.0,
                    "duration_scale": 1.0,
                    "shot_count": len(clips),
                    "created_at": created_at,
                }
                (mask_dir / f"mask_P001_{token}.json").write_text(json.dumps(manifest), encoding="utf-8")

            write_mask_run("old", mask_a, 10.0)
            write_mask_run("new", mask_b, 20.0)

            restored = _restore_generation_package_state({
                "project_dir": str(root),
                "package_id": "P001",
                "shots": [
                    {"shot_id": "S001", "video_path": str(shot_a), "duration": 1.0},
                    {"shot_id": "S002", "video_path": str(shot_b), "duration": 1.0},
                ],
            })

            self.assertTrue(restored["restored"])
            self.assertEqual(restored["status"], "mask_done")
            self.assertEqual(restored["whiteMaskPath"], str(mask_b))
            self.assertEqual(len(restored["maskRuns"]), 2)
            self.assertEqual([run["id"] for run in restored["maskRuns"]], ["new", "old"])
            self.assertEqual([clip["shot_id"] for clip in restored["maskRuns"][0]["clips"]], ["S001", "S002"])

    def test_generation_package_restore_keeps_running_mask_task(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            shot_a = root / "shot_a.mp4"
            shot_b = root / "shot_b.mp4"
            for path in (shot_a, shot_b):
                path.write_bytes(b"video")
            state_dir = root / "04_AI输出成片" / "generation_packages" / "P001"
            state_dir.mkdir(parents=True)
            (state_dir / "generation_state.json").write_text(json.dumps({
                "status": "mask_running",
                "maskTaskId": "task-mask-001",
                "maskPollContext": {
                    "packageId": "P001",
                    "outputRole": "mask",
                    "sourcePath": str(root / "P001_source.mp4"),
                    "outputLabel": "白膜视频",
                },
            }), encoding="utf-8")

            restored = _restore_generation_package_state({
                "project_dir": str(root),
                "package_id": "P001",
                "shots": [
                    {"shot_id": "S001", "video_path": str(shot_a), "duration": 1.0},
                    {"shot_id": "S002", "video_path": str(shot_b), "duration": 1.0},
                ],
            })

            self.assertTrue(restored["restored"])
            self.assertEqual(restored["status"], "mask_running")
            self.assertEqual(restored["maskTaskId"], "task-mask-001")
            self.assertEqual(restored["maskPollContext"]["outputRole"], "mask")

    def test_generation_package_restore_keeps_pending_h3_without_job_id(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            shot = root / "shot.mp4"
            shot.write_bytes(b"video")
            state_dir = root / "04_AI输出成片" / "generation_packages" / "P001"
            state_dir.mkdir(parents=True)
            (state_dir / "generation_state.json").write_text(json.dumps({
                "status": "mask_running",
                "h3Jobs": {
                    "role": "mask",
                    "startedAt": 1000,
                    "entries": [{
                        "jobId": "",
                        "shotId": "P001（整包）",
                        "status": "running",
                    }],
                },
            }), encoding="utf-8")

            with TemporaryDirectory() as jobs_tmp, patch(
                "web_ui.server.STORYBOARD_MODE2_JOB_ROOT",
                Path(jobs_tmp),
            ):
                restored = _restore_generation_package_state({
                    "project_dir": str(root),
                    "package_id": "P001",
                    "shots": [{"shot_id": "S001", "video_path": str(shot), "duration": 1.0}],
                })

            self.assertTrue(restored["restored"])
            self.assertEqual(restored["status"], "mask_running")
            self.assertEqual(restored["h3Jobs"]["entries"][0]["jobId"], "")

    def test_h3_snapshot_restore_recovers_charswap_jobs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs_root = root / "jobs"
            jobs_root.mkdir()
            mask_a = root / "mask_a.mp4"
            mask_b = root / "mask_b.mp4"
            mask_a.write_bytes(b"video")
            mask_b.write_bytes(b"video")
            for job_id, mask_path, created_at in (
                ("result-a", mask_a, 20.0),
                ("result-b", mask_b, 21.0),
            ):
                (jobs_root / f"{job_id}.json").write_text(json.dumps({
                    "id": job_id,
                    "type": "mode2_h3_charswap",
                    "status": "running",
                    "created_at": created_at,
                    "project_dir": str(root),
                    "mask_video_path": str(mask_path),
                    "logs": ["> submitted"],
                }), encoding="utf-8")

            restored = {
                "whiteMaskPath": str(root / "mask_package.mp4"),
                "maskSegments": [
                    {"shot_id": "S001", "path": str(mask_a), "duration": 1.0},
                    {"shot_id": "S002", "path": str(mask_b), "duration": 1.5},
                ],
            }
            segments = [
                {"shot_id": "S001", "duration": 1.0},
                {"shot_id": "S002", "duration": 1.5},
            ]
            with patch("web_ui.server.STORYBOARD_MODE2_JOB_ROOT", jobs_root):
                recovered = _restore_generation_package_h3_jobs(
                    root, "P001", segments, restored, {},
                )

            self.assertEqual(recovered["status"], "result_running")
            self.assertEqual(recovered["h3Jobs"]["role"], "result")
            self.assertEqual(
                [(item["shotId"], item["jobId"]) for item in recovered["h3Jobs"]["entries"]],
                [("S001", "result-a"), ("S002", "result-b")],
            )

    def test_h3_snapshot_restore_recovers_identified_charswap_without_mask_segments(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs_root = root / "jobs"
            jobs_root.mkdir()
            mask = root / "mask.mp4"
            mask.write_bytes(b"video")
            (jobs_root / "result-a.json").write_text(json.dumps({
                "id": "result-a",
                "type": "mode2_h3_charswap",
                "status": "running",
                "created_at": 20.0,
                "project_dir": str(root),
                "package_id": "P001",
                "shot_id": "S001",
                "mask_video_path": str(mask),
            }), encoding="utf-8")

            with patch("web_ui.server.STORYBOARD_MODE2_JOB_ROOT", jobs_root):
                recovered = _restore_generation_package_h3_jobs(
                    root,
                    "P001",
                    [{"shot_id": "S001", "duration": 1.25}],
                    {},
                    {},
                )

            self.assertEqual(recovered["status"], "result_running")
            self.assertEqual(recovered["h3Jobs"]["entries"], [{
                "jobId": "result-a",
                "shotId": "S001",
                "sourcePath": str(mask),
                "duration": 1.25,
                "status": "running",
                "outputPath": "",
                "error": "",
                "logsTail": [],
            }])

    def test_h3_snapshot_restore_isolates_same_mask_path_by_package(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs_root = root / "jobs"
            jobs_root.mkdir()
            mask = root / "shared_mask.mp4"
            mask.write_bytes(b"video")
            for package_id, job_id, created_at in (
                ("P001", "job-p1", 20.0),
                ("P002", "job-p2", 21.0),
            ):
                (jobs_root / f"{job_id}.json").write_text(json.dumps({
                    "id": job_id,
                    "type": "mode2_h3_charswap",
                    "status": "running",
                    "created_at": created_at,
                    "project_dir": str(root),
                    "package_id": package_id,
                    "shot_id": "S001",
                    "mask_video_path": str(mask),
                }), encoding="utf-8")

            with patch("web_ui.server.STORYBOARD_MODE2_JOB_ROOT", jobs_root):
                recovered = _restore_generation_package_h3_jobs(
                    root,
                    "P001",
                    [{"shot_id": "S001", "duration": 1.0}],
                    {},
                    {},
                )

            self.assertEqual(recovered["h3Jobs"]["entries"][0]["jobId"], "job-p1")

    def test_h3_snapshot_restore_keeps_white_mask_auto_result_flag(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs_root = root / "jobs"
            jobs_root.mkdir()
            source = root / "package.mp4"
            source.write_bytes(b"video")
            (jobs_root / "mask-job.json").write_text(json.dumps({
                "id": "mask-job",
                "type": "mode2_h3_white_mask",
                "status": "running",
                "created_at": 20.0,
                "project_dir": str(root),
                "package_id": "P001",
                "shot_id": "P001（整包）",
                "auto_generate_result": True,
                "clip_path": str(source),
            }), encoding="utf-8")

            with patch("web_ui.server.STORYBOARD_MODE2_JOB_ROOT", jobs_root):
                recovered = _restore_generation_package_h3_jobs(
                    root,
                    "P001",
                    [{"shot_id": "S001", "duration": 1.0}],
                    {"packageSourcePath": str(source)},
                    {},
                )

            self.assertTrue(recovered["h3Jobs"]["autoGenerateResult"])
            self.assertTrue(recovered["pendingAutoResult"])
            self.assertEqual(recovered["h3Jobs"]["entries"][0]["shotId"], "P001（整包）")

    def test_h3_request_identity_is_required_and_trimmed(self) -> None:
        self.assertEqual(_required_h3_identity(" P001 ", "package_id"), "P001")
        with self.assertRaisesRegex(ValueError, "h3_shot_id_required"):
            _required_h3_identity("", "shot_id")
        with self.assertRaisesRegex(ValueError, "h3_package_id_invalid"):
            _required_h3_identity("P001\ninvalid", "package_id")

    def test_generation_package_restore_prefers_newer_server_h3_snapshot(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs_root = root / "jobs"
            jobs_root.mkdir()
            shot = root / "shot.mp4"
            source = root / "package.mp4"
            shot.write_bytes(b"video")
            source.write_bytes(b"video")
            segments = [{"shot_id": "S001", "video_path": str(shot), "duration": 1.0}]
            state_dir = root / "04_AI输出成片" / "generation_packages" / "P001"
            state_dir.mkdir(parents=True)
            (state_dir / "generation_state.json").write_text(json.dumps({
                "status": "mask_running",
                "packageSourcePath": str(source),
                "packageSegments": segments,
                "h3Jobs": {
                    "role": "mask",
                    "startedAt": 1000,
                    "entries": [{"jobId": "old-job", "status": "running"}],
                },
            }), encoding="utf-8")
            (jobs_root / "new-job.json").write_text(json.dumps({
                "id": "new-job",
                "type": "mode2_h3_white_mask",
                "status": "running",
                "created_at": 2.0,
                "project_dir": str(root),
                "clip_path": str(source),
            }), encoding="utf-8")

            with (
                patch("web_ui.server.STORYBOARD_MODE2_JOB_ROOT", jobs_root),
                patch("web_ui.server._generation_package_source_duration_valid", return_value=True),
            ):
                restored = _restore_generation_package_state({
                    "project_dir": str(root),
                    "package_id": "P001",
                    "shots": segments,
                })

            self.assertEqual(restored["status"], "mask_running")
            self.assertEqual(restored["h3Jobs"]["entries"][0]["jobId"], "new-job")

    def test_h3_snapshot_restore_ignores_jobs_before_clear_tombstone(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs_root = root / "jobs"
            jobs_root.mkdir()
            source = root / "package.mp4"
            source.write_bytes(b"video")
            (jobs_root / "old-mask.json").write_text(json.dumps({
                "id": "old-mask",
                "type": "mode2_h3_white_mask",
                "status": "done",
                "created_at": 10.0,
                "project_dir": str(root),
                "clip_path": str(source),
            }), encoding="utf-8")

            with patch("web_ui.server.STORYBOARD_MODE2_JOB_ROOT", jobs_root):
                recovered = _restore_generation_package_h3_jobs(
                    root,
                    "P001",
                    [{"shot_id": "S001", "duration": 1.0}],
                    {"packageSourcePath": str(source)},
                    {"h3ClearedAt": 11.0},
                )

            self.assertEqual(recovered, {})

    def test_generation_package_restore_returns_authoritative_h3_clear_state(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            shot = root / "shot.mp4"
            shot.write_bytes(b"video")
            state_dir = root / "04_AI输出成片" / "generation_packages" / "P001"
            state_dir.mkdir(parents=True)
            (state_dir / "generation_state.json").write_text(json.dumps({
                "status": "idle",
                "h3ClearedAt": 25.0,
                "whiteMaskPath": "",
                "maskSegments": [],
                "maskRuns": [],
            }), encoding="utf-8")

            with TemporaryDirectory() as jobs_tmp, patch(
                "web_ui.server.STORYBOARD_MODE2_JOB_ROOT",
                Path(jobs_tmp),
            ):
                restored = _restore_generation_package_state({
                    "project_dir": str(root),
                    "package_id": "P001",
                    "shots": [{"shot_id": "S001", "video_path": str(shot), "duration": 1.0}],
                })

            self.assertEqual(restored["h3ClearedAt"], 25.0)
            self.assertTrue(restored["restored"])
            self.assertTrue(restored["h3MaskStateAuthoritative"])
            self.assertTrue(restored["h3MaskCleared"])
            self.assertEqual(restored["whiteMaskPath"], "")
            self.assertEqual(restored["maskSegments"], [])
            self.assertEqual(restored["maskRuns"], [])

    def test_h3_resumable_snapshot_stays_running_after_backend_restart(self) -> None:
        with TemporaryDirectory() as tmp:
            jobs_root = Path(tmp)
            snapshot = {
                "id": "mask-job",
                "type": "mode2_h3_white_mask",
                "status": "running",
                "created_at": 20.0,
                "project_dir": str(jobs_root / "project"),
                "package_id": "P001",
                "comfy_prompt_id": "prompt-1",
                "comfy_url": "https://pod.example.com",
                "out_dir": str(jobs_root / "project" / "04_AI输出成片" / "h3_jobs" / "mask-job"),
                "out_name": "whitemask.mp4",
            }
            (jobs_root / "mask-job.json").write_text(json.dumps(snapshot), encoding="utf-8")

            with (
                patch("web_ui.server.STORYBOARD_MODE2_JOB_ROOT", jobs_root),
                patch("web_ui.server._h3_snapshot_was_cleared", return_value=False),
            ):
                restored = _load_storyboard_job_snapshot("mask-job")

            self.assertEqual(restored["status"], "running")
            self.assertTrue(restored["restored_from_snapshot"])

    def test_h3_in_memory_job_is_cancelled_after_clear_tombstone(self) -> None:
        job = {
            "id": "mask-job",
            "type": "mode2_h3_white_mask",
            "status": "done",
            "result": {"output_path": "stale.mp4"},
            "logs": [],
        }
        jobs = {"mask-job": dict(job)}
        with (
            patch("web_ui.server.JOBS", jobs),
            patch("web_ui.server._h3_snapshot_was_cleared", return_value=True),
            patch("web_ui.server._write_storyboard_job_snapshot") as write_snapshot,
        ):
            cancelled = _cancel_h3_job_if_cleared(job)

        self.assertEqual(cancelled["status"], "cancelled")
        self.assertIsNone(cancelled["result"])
        self.assertTrue(jobs["mask-job"]["_cancel"])
        self.assertEqual(write_snapshot.call_args.args[0]["status"], "cancelled")

    def test_h3_resume_output_target_must_match_job_directory_and_fixed_name(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected_dir = root / "04_AI输出成片" / "h3_jobs" / "mask-job"
            job = {
                "id": "mask-job",
                "type": "mode2_h3_white_mask",
                "project_dir": str(root),
                "out_dir": str(expected_dir),
                "out_name": "whitemask.mp4",
            }
            self.assertEqual(_validated_h3_resume_output(job), (expected_dir.resolve(), "whitemask.mp4"))
            with self.assertRaisesRegex(ValueError, "h3_resume_output_target_invalid"):
                _validated_h3_resume_output({**job, "out_name": "other.mp4"})
            with self.assertRaisesRegex(ValueError, "h3_resume_output_target_invalid"):
                _validated_h3_resume_output({**job, "out_dir": str(root / "other")})

    def test_comfy_submission_callback_runs_once_before_waiting(self) -> None:
        client = ComfyClient("https://pod.example.com")
        events: list[tuple[str, str, str]] = []
        history = {"prompt-1": {"status": {"status_str": "success", "completed": True}}}

        def wait_for_completion(prompt_id, client_id, _workflow, _log):
            events.append(("wait", prompt_id, client_id))
            return history

        with (
            patch.object(client, "submit_workflow", return_value=("prompt-1", "client-1")),
            patch.object(client, "wait_for_completion", side_effect=wait_for_completion),
        ):
            client.run_workflow(
                {"1": {"class_type": "Example", "inputs": {}}},
                on_submitted=lambda prompt_id, client_id: events.append(
                    ("submitted", prompt_id, client_id)
                ),
            )

        self.assertEqual(events, [
            ("submitted", "prompt-1", "client-1"),
            ("wait", "prompt-1", "client-1"),
        ])

    def test_generation_package_delete_mask_clip_removes_file_manifest_and_state(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask_dir = root / "04_AI输出成片" / "generation_packages" / "P001" / "mask"
            mask_dir.mkdir(parents=True)
            source = root / "04_AI输出成片" / "channel2_api_white_mask_P001.mp4"
            source.parent.mkdir(parents=True, exist_ok=True)
            clip_a = mask_dir / "mask_01_S001_run.mp4"
            clip_b = mask_dir / "mask_02_S002_run.mp4"
            for path in (source, clip_a, clip_b):
                path.write_bytes(b"video")
            clips = [
                {"shot_id": "S001", "path": str(clip_a), "duration": 1.0},
                {"shot_id": "S002", "path": str(clip_b), "duration": 1.0},
            ]
            manifest = mask_dir / "mask_P001_run.json"
            manifest.write_text(json.dumps({
                "package_id": "P001",
                "run_id": "run",
                "source_path": str(source),
                "output_role": "mask",
                "clips": clips,
                "segments": [],
                "shot_count": 2,
                "created_at": 20.0,
            }), encoding="utf-8")
            state_path = root / "04_AI输出成片" / "generation_packages" / "P001" / "generation_state.json"
            state_path.write_text(json.dumps({
                "package_id": "P001",
                "status": "mask_done",
                "whiteMaskPath": str(source),
                "maskSegments": clips,
                "maskSplitManifest": str(manifest),
                "maskRuns": [{
                    "id": "run",
                    "role": "mask",
                    "source_path": str(source),
                    "manifest_path": str(manifest),
                    "clips": clips,
                    "shot_count": 2,
                    "created_at": 20.0,
                }],
            }), encoding="utf-8")

            result = _delete_generation_package_mask_clip({
                "project_dir": str(root),
                "package_id": "P001",
                "manifest_path": str(manifest),
                "clip_path": str(clip_a),
                "shot_id": "S001",
            })

            self.assertTrue(result["success"])
            self.assertTrue(result["files_deleted"])
            self.assertFalse(clip_a.exists())
            self.assertTrue(clip_b.exists())
            manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual([clip["shot_id"] for clip in manifest_data["clips"]], ["S002"])
            state_data = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state_data["status"], "mask_done")
            self.assertEqual([clip["shot_id"] for clip in state_data["maskRuns"][0]["clips"]], ["S002"])

    def test_asset_delete_uses_in_app_confirm_not_browser_confirm(self) -> None:
        html = (ROOT / "web_ui" / "story_generate_dashboard.html").read_text(encoding="utf-8")
        self.assertNotIn("window.confirm", html)
        self.assertIn("asset-delete-confirm-overlay", html)
        self.assertIn("删除接口没有响应", html)
        self.assertIn("从当前项目物理删除这个资产", html)
        self.assertIn("项目内素材文件已删除", html)
        self.assertNotIn("markCurrentProjectAssetIgnored", html)
        self.assertNotIn("图片文件不会删除", html)
        ignored_pos = html.index('if (asset.manual_asset_status === "ignored") return true;')
        role_pos = html.index('if (asset.kind === "role") return false;')
        self.assertLess(ignored_pos, role_pos)

    def test_manual_empty_shot_cuts_do_not_rerun_hardcut(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "shot.mp4"
            video.write_bytes(b"placeholder")
            captured_ranges: list[list[tuple[float, float]]] = []

            def fake_create(_video_path, ranges, *, media_offset=0.0, existing_subshots=None):
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

            def fake_create(_video_path, ranges, *, media_offset=0.0, existing_subshots=None):
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

            def fake_create(_video_path, ranges, *, media_offset=0.0, existing_subshots=None):
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

    def test_manual_shot_cut_snaps_to_last_valid_frame(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "shot.mp4"
            video.write_bytes(b"placeholder")
            captured_ranges: list[list[tuple[float, float]]] = []

            def fake_create(_video_path, ranges, *, media_offset=0.0, existing_subshots=None):
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
                patch("spvideo.ffmpeg_tools.probe_video", return_value=SimpleNamespace(duration=1.0, fps=30.0)),
                patch("web_ui.server._mode2_create_reference_mask_subclips", side_effect=fake_create),
            ):
                result = _storyboard_mode2_shot_subclips({
                    "project_dir": str(root),
                    "video_path": str(video),
                    "segment_id": "S001",
                    "cut_times": [0.99],
                    "manual_cut": True,
                })

            self.assertEqual(result["cut_times"], [round(29 / 30, 6)])
            self.assertEqual(captured_ranges, [[(0.0, round(29 / 30, 6)), (round(29 / 30, 6), 1.0)]])

    def test_manual_virtual_shot_cuts_do_not_write_preview_videos(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "shot.mp4"
            video.write_bytes(b"placeholder")

            with (
                patch("spvideo.ffmpeg_tools.probe_video", return_value=SimpleNamespace(duration=1.0, fps=30.0)),
                patch("web_ui.server._mode2_create_reference_mask_subclips", side_effect=AssertionError("ffmpeg preview cuts should not run")),
            ):
                result = _storyboard_mode2_shot_subclips({
                    "project_dir": str(root),
                    "video_path": str(video),
                    "segment_id": "S001",
                    "cut_times": [0.99],
                    "manual_cut": True,
                    "virtual_subclips": True,
                })

            self.assertEqual(result["cut_times"], [round(29 / 30, 6)])
            self.assertEqual([item["path"] for item in result["subclips"]], [str(video), str(video)])
            self.assertTrue(all(item["virtual_subclip"] for item in result["subclips"]))

    def test_virtual_shot_cut_uses_segment_window_inside_full_source(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "source.mp4"
            video.write_bytes(b"placeholder")

            with (
                patch("spvideo.ffmpeg_tools.probe_video", return_value=SimpleNamespace(duration=120.0, fps=30.0)),
                patch("web_ui.server._mode2_create_reference_mask_subclips", side_effect=AssertionError("ffmpeg preview cuts should not run")),
            ):
                result = _storyboard_mode2_shot_subclips({
                    "project_dir": str(root),
                    "video_path": str(video),
                    "segment_id": "S015",
                    "segment_start": 65.0,
                    "segment_end": 66.0,
                    "segment_duration": 1.0,
                    "video_offset": 65.0,
                    "cut_times": [0.99],
                    "manual_cut": True,
                    "virtual_subclips": True,
                })

            self.assertEqual(result["cut_times"], [round(29 / 30, 6)])
            self.assertEqual(
                [(item["start"], item["end"], item["media_start"], item["media_end"]) for item in result["subclips"]],
                [(0.0, round(29 / 30, 6), 65.0, round(65 + 29 / 30, 6)), (round(29 / 30, 6), 1.0, round(65 + 29 / 30, 6), 66.0)],
            )

    def test_saved_manual_cut_payload_requests_physical_subclips(self) -> None:
        html = (ROOT / "web_ui" / "story_generate_dashboard.html").read_text(encoding="utf-8")
        payload_fn = html.split("function currentShotCutPayload", 1)[1].split(
            "async function openShotCutWorkbench",
            1,
        )[0]

        self.assertNotIn("virtual_subclips: true", payload_fn)
        self.assertIn("virtual_subclips: !save", payload_fn)

    def test_saved_real_subclips_cut_full_source_with_media_offset(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            source.write_bytes(b"placeholder")
            store_path = root / "assets" / "storyboard_assets.json"
            store_path.parent.mkdir(parents=True)
            store_path.write_text(json.dumps({
                "video_path": str(source),
                "shots": [{
                    "segment_id": "S015",
                    "start": 65.0,
                    "end": 67.0,
                    "duration": 2.0,
                }],
            }), encoding="utf-8")
            cut_calls: list[tuple[Path, float, float, Path]] = []

            def fake_cut(video_path, start, end, output_path):
                cut_calls.append((Path(video_path), float(start), float(end), Path(output_path)))

            with (
                patch("spvideo.ffmpeg_tools.probe_video", return_value=SimpleNamespace(duration=120.0, fps=30.0)),
                patch("spvideo.ffmpeg_tools.cut_segment_precise", side_effect=fake_cut),
                patch("web_ui.server._mode2_ensure_shot_preview_clips", return_value=False),
            ):
                result = _storyboard_mode2_shot_subclips({
                    "project_dir": str(root),
                    "video_path": str(source),
                    "segment_id": "S015",
                    "segment_start": 65.0,
                    "segment_end": 67.0,
                    "segment_duration": 2.0,
                    "video_offset": 65.0,
                    "cut_times": [0.5],
                    "manual_cut": True,
                    "save": True,
                })

            self.assertEqual(
                [(video_path, start, end) for video_path, start, end, _target in cut_calls],
                [(source, 65.0, 65.5), (source, 65.5, 67.0)],
            )
            self.assertEqual(
                [(item["media_start"], item["media_end"]) for item in result["subclips"]],
                [(65.0, 65.5), (65.5, 67.0)],
            )
            self.assertTrue(all(not item.get("virtual_subclip") for item in result["subclips"]))
            self.assertEqual(len({target for _video, _start, _end, target in cut_calls}), 2)

    def test_promoted_physical_subclips_keep_independent_preview_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            source.write_bytes(b"placeholder")
            store_path = root / "assets" / "storyboard_assets.json"
            store_path.parent.mkdir(parents=True)
            store_path.write_text(json.dumps({
                "video_path": str(source),
                "assets": [],
                "shots": [{
                    "segment_id": "S001",
                    "start": 10.0,
                    "end": 12.0,
                    "duration": 2.0,
                }],
            }), encoding="utf-8")
            cut_targets: list[Path] = []

            def fake_cut(_video_path, _start, _end, output_path):
                cut_targets.append(Path(output_path))

            with (
                patch("spvideo.ffmpeg_tools.probe_video", return_value=SimpleNamespace(duration=20.0, fps=30.0)),
                patch("spvideo.ffmpeg_tools.cut_segment_precise", side_effect=fake_cut),
                patch("web_ui.server._mode2_ensure_shot_preview_clips", return_value=False),
            ):
                result = _storyboard_mode2_shot_subclips({
                    "project_dir": str(root),
                    "video_path": str(source),
                    "segment_id": "S001",
                    "segment_start": 10.0,
                    "segment_end": 12.0,
                    "segment_duration": 2.0,
                    "video_offset": 10.0,
                    "cut_times": [0.5],
                    "manual_cut": True,
                    "save": True,
                    "promote_to_timeline": True,
                })

            self.assertTrue(result["promoted_to_timeline"])
            self.assertEqual(len(result["segments"]), 2)
            self.assertEqual(len(cut_targets), 2)
            self.assertNotEqual(cut_targets[0], cut_targets[1])
            for child, target in zip(result["segments"], cut_targets):
                target_text = str(target)
                self.assertEqual(child.get("preview_clip_path"), target_text)
                self.assertEqual(child.get("clip_output_path"), target_text)
                self.assertEqual(child.get("output_path"), target_text)
                self.assertNotEqual(child.get("output_path"), str(source))
                self.assertFalse(child.get("virtual_subclip"))

            saved = json.loads(store_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [shot["output_path"] for shot in saved["shots"]],
                [str(target) for target in cut_targets],
            )

    def test_renumbering_ordinary_shot_preserves_preview_paths_but_clears_generated_outputs(self) -> None:
        preview_path = str(ROOT / "clips" / "mode2_shots" / "S009_00010000_00012000.mp4")
        shots = [{
            "segment_id": "S009",
            "start": 10.0,
            "end": 12.0,
            "duration": 2.0,
            "preview_clip_path": preview_path,
            "clip_output_path": preview_path,
            "output_path": preview_path,
            "generated_path": "generated.mp4",
            "seedance_output_path": "seedance.mp4",
            "seedance_task_id": "task-1",
        }]

        renumbered, shot_id_map = _mode2_renumber_timeline_shots(shots)

        self.assertEqual(shot_id_map, {"S009": ["S001"]})
        self.assertEqual(renumbered[0]["segment_id"], "S001")
        for key in ("preview_clip_path", "clip_output_path", "output_path"):
            self.assertEqual(renumbered[0].get(key), preview_path)
        for key in ("generated_path", "seedance_output_path", "seedance_task_id"):
            self.assertNotIn(key, renumbered[0])

    def test_ensure_shot_preview_clips_preserves_existing_physical_subclip(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            physical_subclip = root / "subshots" / "source" / "source_sub01_00010000_00012000_p1.mp4"
            physical_subclip.parent.mkdir(parents=True)
            physical_subclip.write_bytes(b"physical-subclip")
            shot = {
                "segment_id": "S001",
                "start": 10.0,
                "end": 12.0,
                "duration": 2.0,
                "full_source_video_path": str(source),
                "source_video_path": str(source),
                "preview_clip_path": str(physical_subclip),
                "clip_output_path": str(physical_subclip),
                "output_path": str(physical_subclip),
                "manual_timeline_edit": {"action": "promote_subclips_to_timeline"},
            }
            data = {"video_path": str(source), "shots": [shot]}

            with patch("spvideo.ffmpeg_tools.cut_segment") as cut_mock:
                changed = _mode2_ensure_shot_preview_clips(root, data)

            cut_mock.assert_not_called()
            self.assertFalse(changed)
            for key in ("preview_clip_path", "clip_output_path", "output_path"):
                self.assertEqual(shot.get(key), str(physical_subclip))

    def test_ensure_shot_preview_clips_reuses_old_numbered_matching_range(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            clips_dir = root / "clips" / "mode2_shots"
            clips_dir.mkdir(parents=True)
            old_numbered_clip = clips_dir / "S011_00010000_00012000.mp4"
            old_numbered_clip.write_bytes(b"existing-preview")
            shot = {
                "segment_id": "S012",
                "start": 10.0,
                "end": 12.0,
                "duration": 2.0,
            }
            data = {"video_path": str(source), "shots": [shot]}

            with patch("spvideo.ffmpeg_tools.cut_segment") as cut_mock:
                changed = _mode2_ensure_shot_preview_clips(root, data)

            cut_mock.assert_not_called()
            self.assertTrue(changed)
            for key in ("preview_clip_path", "clip_output_path", "output_path"):
                self.assertEqual(shot.get(key), str(old_numbered_clip))
            self.assertEqual(data.get("clips_dir"), str(clips_dir))

    def test_ensure_shot_preview_clips_cuts_when_no_reusable_preview_exists(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            source.write_bytes(b"source")
            shot = {
                "segment_id": "S003",
                "start": 10.0,
                "end": 12.0,
                "duration": 2.0,
            }
            data = {"video_path": str(source), "shots": [shot]}

            def fake_cut(_source, _start, _end, output_path):
                target = Path(output_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"new-preview")

            with patch("spvideo.ffmpeg_tools.cut_segment", side_effect=fake_cut) as cut_mock:
                changed = _mode2_ensure_shot_preview_clips(root, data)

            cut_mock.assert_called_once()
            source_arg, start_arg, end_arg, target_arg = cut_mock.call_args.args
            expected = root / "clips" / "mode2_shots" / "S003_00010000_00012000.mp4"
            self.assertEqual(Path(source_arg), source)
            self.assertEqual((start_arg, end_arg), (10.0, 12.0))
            self.assertEqual(Path(target_arg), expected)
            self.assertTrue(changed)
            for key in ("preview_clip_path", "clip_output_path", "output_path"):
                self.assertEqual(shot.get(key), str(expected))

    def test_timeline_split_snaps_to_frame_instead_of_rejecting_near_edge(self) -> None:
        shots = [{
            "segment_id": "S001",
            "start": 0.0,
            "end": 1.0,
            "duration": 1.0,
            "fps": 30.0,
        }]

        next_shots, summary, selected_id, selected_index = _mode2_split_timeline_shot(shots, 0, 0.99)

        self.assertEqual(selected_id, "S001")
        self.assertEqual(selected_index, 0)
        self.assertEqual(summary["split_time"], 0.967)
        self.assertEqual([(shot["start"], shot["end"]) for shot in next_shots], [(0.0, 0.967), (0.967, 1.0)])

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

            def fake_create(_video_path, ranges, *, media_offset=0.0, existing_subshots=None):
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

            self.assertTrue(result["files_deleted"])
            self.assertFalse(image_path.exists())
            saved = json.loads(store_path.read_text(encoding="utf-8"))
            self.assertEqual([asset["id"] for asset in saved["assets"]], ["scene_1"])
            shot = saved["shots"][0]
            self.assertEqual(shot["asset_ids"], ["scene_1"])
            self.assertEqual(shot["role_asset_ids"], [])
            self.assertEqual(shot["scene_asset_ids"], ["scene_1"])


if __name__ == "__main__":
    unittest.main()
