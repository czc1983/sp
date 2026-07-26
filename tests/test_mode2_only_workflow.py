from __future__ import annotations

from types import SimpleNamespace
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from web_ui.server import (
    SEEDANCE_A_TASKS,
    SEEDANCE_A_TASKS_LOCK,
    _delete_storyboard_mode2_asset,
    _mode2_scail2_route_check,
    _restore_generation_package_state,
    _submit_seedance_a_task,
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
        legacy_block = server.split("legacy_mode1_paths = {", 1)[1].split("}", 1)[0]
        self.assertIn('"/api/storyboard-reference-mask"', legacy_block)
        self.assertIn('"/api/storyboard-reference-white-mask"', legacy_block)
        self.assertIn('"/api/storyboard-reference-expression-mask"', legacy_block)
        self.assertIn('if parsed.path == "/api/storyboard-reference-white-mask"', server)
        self.assertIn('if parsed.path == "/api/generation-package/prepare"', server)
        self.assertIn('if parsed.path == "/api/generation-package/restore"', server)
        self.assertIn('if parsed.path == "/api/generation-package/split-output"', server)
        self.assertIn("def _prepare_generation_package_video", server)
        self.assertIn("def _restore_generation_package_state", server)
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
        self.assertIn("compose-pack-shot-list", html)
        self.assertIn("段：", html)
        self.assertIn("function confirmComposePlan", html)
        self.assertIn('openComposePage({ rebuild: true })', html)
        self.assertIn("5-10 秒最佳", html)
        self.assertIn("超过 15 秒禁止", html)
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
        self.assertIn("appendWhiteMaskShelfRun(packageId, maskRun)", html)
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
        self.assertIn("function syncGenerationPackageToSeedanceDraft", html)
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
        self.assertIn("whiteMaskPath", html)
        self.assertIn("maskSegments", html)
        self.assertIn("maskRuns", html)
        self.assertIn("resultRuns", html)
        self.assertIn("function generationRunFromSplitData", html)
        self.assertIn("function appendGenerationRun", html)
        self.assertIn("白膜结果包", html)
        self.assertIn("成片结果包", html)
        self.assertIn("已保存${isMaskOutput ? \"白膜结果包\" : \"成片结果包\"}", html)
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
        self.assertIn("把参考图替换原视频中的人物，并把原视频转绘成白膜视频，人物站位、方向、动作表情保持不变。", html)
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
        self.assertIn("当前资产缩略图", html)
        self.assertIn("data-generation-add-reference", html)
        self.assertIn("data-generation-remove-reference", html)
        self.assertIn("已选参考图", html)
        self.assertIn("本项目已关闭自动放图", html)
        self.assertNotIn("function generationPackReferenceImages", html)
        self.assertNotIn("prefer_current_shot_roles: true", html)
        pack_role_fn = html.split("function generationPackRoleAssets", 1)[1].split("function generationManualReferenceImages", 1)[0]
        self.assertNotIn("storyLegacyReferenceSegments", pack_role_fn)
        self.assertNotIn("legacy_reference", pack_role_fn)
        sync_fn = html.split("function syncGenerationPackageToSeedanceDraft", 1)[1].split("function generationPlayerHtml", 1)[0]
        self.assertIn("generationManualReferenceImages()", sync_fn)
        self.assertIn("seedanceDraft.images = normalizeMaterialList(seedanceDraft.images, 1)", sync_fn)
        self.assertNotIn("generationPackReferenceImages", sync_fn)
        self.assertNotIn("seedanceDraft.images = normalizeMaterialList(refs, 1)", sync_fn)
        self.assertIn('outputRole: "mask"', html)
        self.assertIn('compareOutputLabel: "白膜视频"', html)
        self.assertIn("autoGenerateResult", html)
        self.assertIn("whiteMaskPath", html)
        self.assertIn("先生成白膜视频", html)
        self.assertIn("多人接口需要先用原视频生成白膜，再用白膜生成结果。", html)
        source_info_fn = html.split("function generationPackSourceInfo", 1)[1].split("function generationPackSourceVideo", 1)[0]
        self.assertIn("runtime.packageSourcePath || runtime.sourcePath", source_info_fn)
        self.assertIn("shots.length > 1", source_info_fn)
        self.assertIn("整包", source_info_fn)
        render_fn = html.split("function renderGenerationPackageWorkbench", 1)[1].split("function selectGenerationPackage", 1)[0]
        self.assertIn("sourceInfo.emptyText", render_fn)
        self.assertIn("ensureGenerationPackageSourcePreview(pack)", render_fn)
        mask_fn = html.split("async function startGenerationPackageMask", 1)[1].split("async function startGenerationPackageResult", 1)[0]
        self.assertIn("submitSeedanceATask", mask_fn)
        self.assertIn('seedanceDraft.prompt = generationEditorPromptText(pack, "mask")', mask_fn)
        self.assertNotIn("seedanceSubmitting", mask_fn)
        self.assertNotIn("startSeedanceReferenceWhiteMask", mask_fn)
        result_fn = html.split("async function startGenerationPackageResult", 1)[1].split("async function startGenerationPackageFull", 1)[0]
        self.assertNotIn("seedanceSubmitting", result_fn)
        self.assertIn("seedancePackagePollTimers", html)
        self.assertIn("function startSeedancePackageTaskPolling", html)
        self.assertIn("function pollSeedancePackageTask", html)
        self.assertLess(html.index('id="generationCompareOriginal"'), html.index('id="generationCompareMask"'))
        self.assertLess(html.index('id="generationCompareMask"'), html.index('id="generationCompareResult"'))

    def test_channel2_submit_uploads_references_and_sends_public_urls(self) -> None:
        class FakeResponse:
            ok = True
            status_code = 200
            text = '{"data":{"task_id":"task_public_refs"}}'

            def json(self):
                return {"data": {"task_id": "task_public_refs"}}

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_a = root / "wife.png"
            image_b = root / "husband.png"
            video = root / "P001_source.mp4"
            image_a.write_bytes(b"image-a")
            image_b.write_bytes(b"image-b")
            video.write_bytes(b"video")
            calls: list[dict] = []

            def fake_upload(ref, _api_key):
                return "https://public.example/" + Path(str(ref)).name

            def fake_post(url, **kwargs):
                calls.append({"url": url, "json": kwargs.get("json")})
                return FakeResponse()

            try:
                with (
                    patch("web_ui.server._seedance_a_upload_ref", side_effect=fake_upload),
                    patch("web_ui.server.requests.post", side_effect=fake_post),
                ):
                    result = _submit_seedance_a_task({
                        "taskName": "api_white_mask_P001",
                        "project_dir": str(root),
                        "source_video_path": str(video),
                        "package_id": "P001",
                        "relay_api_key": "sk-test",
                        "relay_base_url": "https://upstream.example",
                        "payload": {
                            "model": "sd-2.0-r3",
                            "prompt": "测试白膜",
                            "duration": 7,
                            "ratio": "9:16",
                            "aspect_ratio": "9:16",
                            "resolution": "720x1280",
                            "images": [str(image_a), str(image_b)],
                            "referenceVideos": [str(video)],
                        },
                    })
            finally:
                with SEEDANCE_A_TASKS_LOCK:
                    SEEDANCE_A_TASKS.pop("task_public_refs", None)

            self.assertEqual(result["task_id"], "task_public_refs")
            self.assertEqual(calls[0]["url"], "https://upstream.example/v1/videos")
            upstream = calls[0]["json"]
            upstream_text = json.dumps(upstream, ensure_ascii=False)
            self.assertNotIn(str(image_a), upstream_text)
            self.assertNotIn(str(image_b), upstream_text)
            self.assertNotIn(str(video), upstream_text)
            self.assertEqual(upstream["images"], [
                "https://public.example/wife.png",
                "https://public.example/husband.png",
            ])
            self.assertEqual(upstream["image_urls"], upstream["images"])
            self.assertEqual(upstream["model"], "sd-2.0-r3")
            self.assertEqual(upstream["duration"], 7)
            self.assertEqual(upstream["aspect_ratio"], "9:16")
            self.assertEqual(upstream["size"], "720x1280")
            self.assertEqual(upstream["video_urls"], ["https://public.example/P001_source.mp4"])
            self.assertNotIn("metadata", upstream)
            self.assertNotIn("videos", upstream)
            self.assertNotIn("referenceVideos", upstream)
            self.assertNotIn("reference_videos", upstream)
            self.assertNotIn("video_reference", upstream)
            self.assertNotIn("video_url", upstream)
            self.assertNotIn("image_url", upstream)
            self.assertNotIn("ratio", upstream)
            self.assertNotIn("async", upstream)
            self.assertTrue(Path(result["debug_path"]).exists())

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
