"""Gemini 视觉理解：分析片段内容、反推生成路线。

用法::

    from spvideo.gemini_analyzer import create_client

    client = create_client(api_key="sk-xxx")
    result = client.analyze_segment_keyframes(keyframe_paths, start=0.0, end=10.0)
    # → {"has_human": true, "scene_type": "person_presenting_product", ...}
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from .models import FrameFeatures, Segment

logger = logging.getLogger(__name__)

# ── 默认配置 ──────────────────────────────────────────────────────────
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.5-omni-plus"
DIFF_THRESHOLD = 0.12  # 画面变化阈值，超过此值认为两帧之间有内容切换

# ── 客户端 ────────────────────────────────────────────────────────────


class GeminiClient:
    """封装 Gemini（OpenAI 兼容接口），用于视频片段看图理解。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        request_timeout: int = 120,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        # 直连，不走系统代理（API 是内网 IP，不需要代理）
        self._session.trust_env = False
        self._timeout = request_timeout

    def close(self) -> None:
        self._session.close()

    # ── 对外接口 ──────────────────────────────────────────────────────

    def analyze_full_video(
        self,
        frames: list[FrameFeatures],
        batch_size: int = 20,
        overlap: int = 3,
        transcript: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """一次性看完整视频帧序列，返回语义场景分段（带边界时间戳）。

        这是推荐的模式：把视频所有帧发给 Gemini → Gemini 识别场景边界 →
        算法按边界精确裁切。比"先粗切再逐段问"准得多。

        Parameters
        ----------
        frames : list[FrameFeatures]
            视频所有采样帧（按时间排序），建议 1fps，长度不限
        batch_size : int
            每批最多帧数（超长视频自动分批，带 overlap 避免丢边界）
        overlap : int
            相邻批次重叠帧数
        """
        if not frames:
            return []

        frames_sorted = sorted(frames, key=lambda f: f.time)
        n = len(frames_sorted)

        # 如果帧数不多，直接一次送
        if n <= batch_size:
            result = self._call_full_analysis(frames_sorted, transcript=transcript)
            if result:
                return result
            logger.warning("全帧分析失败，降级为空结果")
            return []

        # ── 分批：每 batch_size 帧一组，相邻组重叠 overlap 帧 ──────
        all_scenes: list[dict[str, Any]] = []
        step = batch_size - overlap
        for batch_start in range(0, n, step):
            batch = frames_sorted[batch_start : batch_start + batch_size]
            if len(batch) < 3:
                continue  # 太少帧没意义
            logger.info("全帧分析批次 %s～%s (%.0f～%.0fs)...",
                        batch[0].path, batch[-1].path, batch[0].time, batch[-1].time)
            scenes = self._call_full_analysis(batch, transcript=transcript)
            if scenes:
                all_scenes.extend(scenes)

        if not all_scenes:
            return []

        # ── 合并去重：按时间排序，合并相邻同类型片段 ────────────────
        all_scenes.sort(key=lambda s: s.get("start", 0))
        merged: list[dict[str, Any]] = [all_scenes[0]]
        for s in all_scenes[1:]:
            last = merged[-1]
            # Batch overlap may duplicate a scene, but a role/action boundary
            # must survive even when both scenes share the same broad type.
            same_type = s.get("scene_type") == last.get("scene_type")
            close = s.get("start", 0) <= last.get("end", 0) + 1.0
            last_characters = {str(value) for value in last.get("characters", []) if str(value)}
            current_characters = {str(value) for value in s.get("characters", []) if str(value)}
            same_characters = (
                not last_characters
                or not current_characters
                or last_characters == current_characters
            )
            semantic_break = str(s.get("boundary_kind") or "") in {
                "shot_change",
                "role_change",
                "action_change",
                "location_change",
                "topic_change",
            }
            if same_type and close and same_characters and not semantic_break:
                last["end"] = max(last["end"], s.get("end", 0))
                if s.get("description"):
                    last["description"] = (
                        last.get("description", "") + " | " + s["description"]
                    )
            else:
                merged.append(s)
        return merged

    def transcribe_audio(self, audio_path: str | Path) -> dict[str, Any] | None:
        """Transcribe the original soundtrack once, preserving speaker/time hints."""
        path = Path(audio_path)
        if not path.exists() or path.stat().st_size <= 0:
            return None
        # DashScope requires the Base64 string to stay below 10MB.
        if path.stat().st_size > 7 * 1024 * 1024:
            logger.warning("音频超过 Base64 单请求限制，跳过在线转写: %s", path)
            return None
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        prompt = (
            "你是短剧音频转写助手。请完整听取音频，按时间顺序输出对白、说话人和声音事件。"
            "角色真名只有在对白明确称呼时才能填写；不确定时 role_name 为空，不要猜。"
            "输出严格 JSON："
            '{"language":"zh","summary":"","utterances":['
            '{"start":0.0,"end":1.2,"speaker":"speaker_1","role_name":"","text":"对白或声音描述"}]}'
        )
        media_content = {
            "type": "input_audio",
            "input_audio": {
                "data": f"data:;base64,{encoded}",
                "format": path.suffix.lower().lstrip(".") or "mp3",
            },
        }
        # Follow the official Qwen-Omni ordering: media first, instruction second.
        contents = [media_content, {"type": "text", "text": prompt}]
        data = self._post(self._build_payload(contents, json_mode=True, max_tokens=8192), timeout=300)
        if not isinstance(data, dict):
            return None
        utterances = data.get("utterances") or data.get("segments")
        if not isinstance(utterances, list):
            return None
        normalized = []
        for raw in utterances:
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            item["start"] = self._time_seconds(item.get("start"))
            item["end"] = self._time_seconds(item.get("end"))
            normalized.append(item)
        data["utterances"] = normalized
        return data

    def analyze_video_url(
        self,
        video_url: str,
        *,
        duration: float,
    ) -> list[dict[str, Any]] | None:
        """Analyze the original video through a public URL."""
        video_url = str(video_url or "").strip()
        if not video_url:
            return None
        prompt = _VIDEO_URL_PROMPT.format(duration=f"{float(duration):.2f}")
        contents = [
            {"type": "video_url", "video_url": {"url": video_url}},
            {"type": "text", "text": prompt},
        ]
        timeout = min(1200, max(300, int(float(duration) * 2.5)))
        data = self._post(self._build_payload(contents, json_mode=True, max_tokens=8192), timeout=timeout)
        if data is None:
            return None
        parsed = self._parse_analysis(data)
        if isinstance(parsed, dict):
            for key in ("scenes", "segments", "results"):
                val = parsed.get(key)
                if isinstance(val, list):
                    return val
            if any(k in parsed for k in ("scene_type", "start", "end")):
                return [parsed]
        return None

    def analyze_mode2_plan(
        self,
        *,
        duration: float,
        video_url: str = "",
        frames: list[FrameFeatures] | None = None,
        transcript: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Return Mode2 story, scene timeline and executable asset evidence in one request.

        ``video_url`` is preferred when supplied. ``frames`` may still be passed
        with a video URL so the model can map evidence times back to the local
        analysis-frame indices without uploading the images a second time.
        """
        video_url = str(video_url or "").strip()
        frame_items = sorted(frames or [], key=lambda frame: frame.time)
        encoded_frames: list[tuple[FrameFeatures, str]] = []
        if not video_url:
            for frame in frame_items:
                encoded = self._frame_to_base64(Path(frame.path), max_width=320)
                if encoded:
                    encoded_frames.append((frame, encoded))
            if not encoded_frames:
                return None
            frame_items = [item[0] for item in encoded_frames]
        if not video_url and not frame_items:
            return None

        frame_manifest = [
            {"index": index, "time": round(float(frame.time), 3)}
            for index, frame in enumerate(frame_items, start=1)
        ]
        prompt = _MODE2_PLAN_PROMPT.format(
            duration=f"{float(duration):.2f}",
            source_mode="原视频 URL（包含连续画面和音频）" if video_url else "按时间排序的采样帧",
            frame_manifest=json.dumps(frame_manifest, ensure_ascii=False),
            audio_context=self._transcript_context(
                transcript,
                0.0,
                float(duration),
            ),
        )
        contents: list[dict[str, Any]] = []
        if video_url:
            contents.append({"type": "video_url", "video_url": {"url": video_url}})
        contents.append({"type": "text", "text": prompt})
        if not video_url:
            for _frame, encoded in encoded_frames:
                contents.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded}",
                        "detail": "low",
                    },
                })

        timeout = min(1200, max(300, int(float(duration) * 2.5)))
        data = self._post(
            self._build_payload(contents, json_mode=True, max_tokens=16384),
            timeout=timeout,
        )
        parsed = self._parse_analysis(data)
        if not isinstance(parsed, dict):
            return None
        return self._normalize_mode2_plan(parsed, frame_manifest)

    @classmethod
    def _normalize_mode2_plan(
        cls,
        data: dict[str, Any],
        frame_manifest: list[dict[str, Any]],
    ) -> dict[str, Any]:
        scenes = [dict(item) for item in data.get("scenes", []) if isinstance(item, dict)]
        raw_manifest = data.get("asset_manifest")
        raw_results = raw_manifest.get("results", []) if isinstance(raw_manifest, dict) else []
        results: list[dict[str, Any]] = []
        frame_count = len(frame_manifest)
        frame_times = [float(item.get("time") or 0) for item in frame_manifest]
        allowed_kinds = {"role", "scene", "prop", "mixed", "ignore"}
        for index, raw in enumerate(raw_results, start=1):
            if not isinstance(raw, dict):
                continue
            kind = str(raw.get("kind") or "mixed").strip().lower()
            if kind not in allowed_kinds:
                kind = "mixed"
            evidence_times = []
            for value in raw.get("evidence_times") or []:
                seconds = cls._time_seconds(value)
                if seconds not in evidence_times:
                    evidence_times.append(seconds)
            try:
                representative_index = int(raw.get("representative_frame_index"))
            except (TypeError, ValueError):
                representative_index = 0
            if representative_index < 1 or representative_index > frame_count:
                if evidence_times and frame_times:
                    target = evidence_times[len(evidence_times) // 2]
                    representative_index = min(
                        range(1, frame_count + 1),
                        key=lambda item: abs(frame_times[item - 1] - target),
                    )
                else:
                    representative_index = 0
            try:
                confidence = max(0.0, min(1.0, float(raw.get("confidence") or 0)))
            except (TypeError, ValueError):
                confidence = 0.0
            visible_props = raw.get("visible_props")
            if not isinstance(visible_props, list):
                visible_props = []
            results.append({
                "group_id": str(raw.get("group_id") or f"G{index:03d}").strip(),
                "kind": kind,
                "name": str(raw.get("name") or "").strip(),
                "identity": str(raw.get("identity") or "").strip(),
                "matched_role": str(raw.get("matched_role") or "").strip(),
                "physical_scene": str(raw.get("physical_scene") or "").strip(),
                "evidence_times": evidence_times,
                "representative_frame_index": representative_index or None,
                "visible_props": visible_props,
                "confidence": confidence,
                "reason": str(raw.get("reason") or "").strip(),
            })
        return {
            "story_summary": str(data.get("story_summary") or "").strip(),
            "scenes": scenes,
            "asset_manifest": {"results": results},
        }

    def _call_full_analysis(
        self,
        frames: list[FrameFeatures],
        *,
        transcript: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]] | None:
        """发一批帧给 Gemini，返回这批帧对应的场景列表。"""
        b64_list: list[str] = []
        for f in frames:
            # 全帧分析用小图省 token
            b64 = self._frame_to_base64(Path(f.path), max_width=320)
            if b64:
                b64_list.append(b64)

        if not b64_list:
            return None

        time_info = ", ".join(f"#{i+1}: {frames[i].time:.2f}s" for i in range(len(frames)))

        prompt = _FULL_VIDEO_PROMPT.format(
            frame_count=len(b64_list),
            time_info=time_info,
            start_time=f"{frames[0].time:.2f}",
            end_time=f"{frames[-1].time:.2f}",
            audio_context=self._transcript_context(transcript, frames[0].time, frames[-1].time),
        )

        contents: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for b64 in b64_list:
            contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
            })

        # 全帧分析：更大 token 上限 + 更长超时
        payload = self._build_payload(contents, json_mode=True, max_tokens=8192)
        # A slow or wedged upstream must not block the whole segmentation job
        # for an hour. Large Omni batches get up to ten minutes, then the
        # pipeline falls back to technical boundaries.
        timeout = min(600, max(180, len(b64_list) * 3))
        data = self._post(payload, timeout=timeout)
        if data is None:
            return None
        parsed = self._parse_analysis(data)
        if parsed is None:
            return None
        # 从 {"scenes": [...]} 中提取列表
        if isinstance(parsed, dict):
            for key in ("scenes", "segments", "results"):
                val = parsed.get(key)
                if isinstance(val, list):
                    return val
            # 如果没有 scenes 键，可能是单场景结果
            if any(k in parsed for k in ("scene_type", "start", "end")):
                return [parsed]
        return None

    @staticmethod
    def _transcript_context(
        transcript: dict[str, Any] | None,
        start: float,
        end: float,
    ) -> str:
        if not transcript:
            return "未提供音频转写，只能根据画面判断人物，不得猜测角色真名。"
        lines: list[str] = []
        for item in transcript.get("utterances") or []:
            if not isinstance(item, dict):
                continue
            item_start = float(item.get("start") or 0)
            item_end = float(item.get("end") or item_start)
            if item_end < start - 0.5 or item_start > end + 0.5:
                continue
            speaker = str(item.get("speaker") or "speaker")
            role = str(item.get("role_name") or "").strip()
            text = str(item.get("text") or "").strip()
            if text:
                lines.append(f"[{item_start:.2f}-{item_end:.2f}] {speaker}{f'({role})' if role else ''}: {text}")
        if not lines:
            return "音频已处理，但本时间窗没有可用对白。"
        return "带时间戳的原视频音频转写：\n" + "\n".join(lines)[:12000]

    @staticmethod
    def _time_seconds(value: Any) -> float:
        if isinstance(value, (int, float)):
            return max(0.0, float(value))
        text = str(value or "").strip()
        if not text:
            return 0.0
        try:
            return max(0.0, float(text))
        except ValueError:
            pass
        parts = text.split(":")
        try:
            total = 0.0
            for part in parts:
                total = total * 60 + float(part)
            return max(0.0, total)
        except ValueError:
            return 0.0

    def analyze_segment_keyframes(
        self,
        keyframe_paths: list[str | Path],
        start: float = 0.0,
        end: float = 0.0,
        retry: int = 1,
        prompt_override: str | None = None,
    ) -> dict[str, Any] | None:
        """发送片段的关键帧给 Gemini，返回结构化理解结果。

        返回示例::

            {
                "has_human": true,
                "has_product": true,
                "main_subject": "human",
                "scene_type": "person_presenting_product",
                "generation_route": "human_driver_with_product_reference",
                "visual_style": "真人手持产品讲解",
                "needs_manual_review": false,
                "description": "一个人站在白色背景前手持产品讲解"
            }
        """
        if not keyframe_paths:
            logger.warning("analyze_segment_keyframes: 没有关键帧，跳过")
            return None

        b64_frames = []
        for p in keyframe_paths:
            b64 = self._frame_to_base64(Path(p), max_width=640)
            if b64:
                b64_frames.append(b64)

        if not b64_frames:
            logger.warning("所有关键帧编码失败")
            return None

        if prompt_override:
            prompt = prompt_override
        else:
            prompt = _SEGMENT_ANALYSIS_PROMPT.format(
                start=f"{start:.2f}",
                end=f"{end:.2f}",
                duration=f"{max(0.01, end - start):.2f}",
            )

        contents: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for b64 in b64_frames:
            contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
            })

        payload = self._build_payload(contents, json_mode=True)

        last_err = None
        for attempt in range(retry + 1):
            data = self._post(payload)
            if data is not None:
                parsed = self._parse_analysis(data)
                if parsed is not None:
                    return parsed
            last_err = data
            if attempt < retry:
                wait = 2.0 * (attempt + 1)
                logger.info("Gemini 返回不合预期，%.1fs 后重试(%d/%d)...", wait, attempt + 1, retry)
                time.sleep(wait)

        logger.warning("Gemini 分析失败（重试 %d 次后）: %s", retry, last_err)
        return None

    def batch_analyze_segments(
        self,
        segments: list[Segment],
        all_frames: list[FrameFeatures],
        keyframes_per_segment: int = 5,
        delay: float = 0.3,
        adaptive: bool = True,
    ) -> list[dict[str, Any]]:
        """批量分析多个片段，每段选关键帧 → Gemini 理解 → 返回结构化结果。

        Parameters
        ----------
        segments : list[Segment]
            待分析的片段列表
        all_frames : list[FrameFeatures]
            全量帧列表（用于选关键帧）
        keyframes_per_segment : int
            每段最多选多少张关键帧
        delay : float
            每段之间的间隔秒数（避免限频）
        adaptive : bool
            True = 用自适应二分选帧（变化多处多取，均匀段少取）
            False = 固定策略（开头+中间+结尾+变化最大+含人最多）
        """
        results: list[dict[str, Any]] = []
        total = len(segments)

        for idx, seg in enumerate(segments):
            logger.info("Gemini 分析片段 %s (%d/%d)...", seg.segment_id, idx + 1, total)
            kf_paths = self._pick_keyframes(
                seg, all_frames,
                max_count=keyframes_per_segment,
                adaptive=adaptive,
            )

            if not kf_paths:
                results.append(self._fallback_result(seg))
                continue

            analysis = self.analyze_segment_keyframes(
                kf_paths, start=seg.start, end=seg.end
            )

            if analysis is None:
                results.append(self._fallback_result(seg))
            else:
                analysis["segment_id"] = seg.segment_id
                analysis["start"] = seg.start
                analysis["end"] = seg.end
                results.append(analysis)

            if idx < total - 1 and delay > 0:
                time.sleep(delay)

        return results

    # ── 关键帧选取 ────────────────────────────────────────────────────

    def _pick_keyframes(
        self,
        segment: Segment,
        all_frames: list[FrameFeatures],
        max_count: int = 5,
        adaptive: bool = True,
    ) -> list[Path]:
        """自适应选帧：二分法思路，变化大的区域多取帧，均匀段少取帧。

        算法（你说的"取1，10，变了再取5"）:
        1. 先取首帧和尾帧
        2. 用已有 pixel diff 数据判断中间是否有显著变化
        3. 变化大 → 在变化点附近二分插入新帧
        4. 变化小 → 就 2-3 帧，不浪费
        5. 保证开头、结尾附近一定有帧
        """
        seg_frames = [f for f in all_frames if segment.start <= f.time <= segment.end]
        if not seg_frames:
            return []

        frames_sorted = sorted(seg_frames, key=lambda f: f.time)
        n = len(frames_sorted)

        if n <= max_count:
            return [Path(f.path) for f in frames_sorted if Path(f.path).exists()]

        if not adaptive:
            return self._pick_keyframes_fixed(seg_frames, frames_sorted, n, max_count)

        # ── 自适应二分 ────────────────────────────────────────────
        # 找到所有 diff_prev 较高的位置（场景变化候选点）
        high_diff_indices = [
            i for i in range(1, n)
            if frames_sorted[i].diff_prev >= DIFF_THRESHOLD
        ]

        picked: set[int] = set()
        picked.add(0)          # 首帧
        picked.add(n - 1)      # 尾帧

        def _bisect(lo: int, hi: int, depth: int) -> None:
            """看看 lo~hi 之间有没有变化，有就在中间插帧再递归。"""
            if depth >= 2 or hi - lo <= 2 or len(picked) >= max_count:
                return
            has_change = any(lo < i < hi for i in high_diff_indices)
            if not has_change:
                return
            mid = (lo + hi) // 2
            if mid not in picked:
                picked.add(mid)
            _bisect(lo, mid, depth + 1)
            _bisect(mid, hi, depth + 1)

        _bisect(0, n - 1, 0)

        # ── 加一张"人在画面最突出"的帧（如果段里有人） ──────────────
        skin_sorted = sorted(
            [(i, f.center_skin_ratio) for i, f in enumerate(frames_sorted) if i not in picked],
            key=lambda x: -x[1],
        )
        if skin_sorted and skin_sorted[0][1] > 0.08 and len(picked) < max_count:
            picked.add(skin_sorted[0][0])

        # ── 只有画面确实有变化才补到 max_count，均匀段不浪费 ──────
        if high_diff_indices:
            candidates = [i for i in range(n) if i not in picked]
            while len(picked) < max_count and candidates:
                best_i = max(candidates, key=lambda i: min(
                    abs(frames_sorted[i].time - frames_sorted[j].time) for j in picked
                ))
                picked.add(best_i)
                candidates.remove(best_i)

        return [Path(frames_sorted[i].path) for i in sorted(picked) if Path(frames_sorted[i].path).exists()]

    def _pick_keyframes_fixed(
        self,
        seg_frames: list[FrameFeatures],
        frames_sorted: list[FrameFeatures],
        n: int,
        max_count: int,
    ) -> list[Path]:
        """固定策略：开头+结尾+中间+变化最大+含人最多。"""
        picked: set[int] = set()
        picked.add(0)
        picked.add(n - 1)
        picked.add(n // 2)

        diff_sorted = sorted(
            [(i, f.diff_prev) for i, f in enumerate(frames_sorted) if i not in picked],
            key=lambda x: -x[1],
        )
        if diff_sorted:
            picked.add(diff_sorted[0][0])

        skin_sorted = sorted(
            [(i, f.center_skin_ratio) for i, f in enumerate(frames_sorted) if i not in picked],
            key=lambda x: -x[1],
        )
        if skin_sorted and skin_sorted[0][1] > 0.05:
            picked.add(skin_sorted[0][0])

        candidates = [i for i in range(n) if i not in picked]
        while len(picked) < max_count and candidates:
            best_i = max(candidates, key=lambda i: min(
                abs(frames_sorted[i].time - frames_sorted[j].time) for j in picked
            ))
            picked.add(best_i)
            candidates.remove(best_i)

        return [Path(frames_sorted[i].path) for i in sorted(picked) if Path(frames_sorted[i].path).exists()]

        result = [Path(frames_sorted[i].path) for i in sorted(picked) if Path(frames_sorted[i].path).exists()]
        return result

    # ── 内部辅助 ──────────────────────────────────────────────────────

    def _build_payload(
        self,
        contents: list[dict[str, Any]],
        json_mode: bool = True,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        msg: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": contents}],
            "max_tokens": max_tokens,
        }
        if json_mode:
            msg["response_format"] = {"type": "json_object"}
        return msg

    def _post(self, payload: dict[str, Any], timeout: int | None = None) -> dict[str, Any] | None:
        use_stream = "qwen" in self.model.lower() and "omni" in self.model.lower()
        request_payload = dict(payload)
        if use_stream:
            request_payload["stream"] = True
            request_payload["stream_options"] = {"include_usage": True}
            request_payload.setdefault("modalities", ["text"])
        try:
            resp = self._session.post(
                f"{self.base_url}/chat/completions",
                json=request_payload,
                timeout=timeout or self._timeout,
                stream=use_stream,
            )
            resp.raise_for_status()
            if use_stream and "text/event-stream" in str(resp.headers.get("Content-Type") or "").lower():
                return self._parse_stream_response(resp)
            data = resp.json()
        except requests.RequestException as e:
            detail = ""
            response = getattr(e, "response", None)
            if response is not None:
                try:
                    detail = str(response.text or "")[:1000]
                except Exception:  # noqa: BLE001
                    detail = ""
            logger.error("Gemini API 请求失败: %s%s", e, f" — {detail}" if detail else "")
            return None
        except json.JSONDecodeError as e:
            logger.error("Gemini 返回非 JSON: %s", e)
            return None

        try:
            content = data["choices"][0]["message"]["content"]
            return self._load_json_content(content)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.warning("解析 Gemini 输出失败: %s — 原文: %s", e, content if 'content' in locals() else 'N/A')
            return None

    def _parse_stream_response(self, response: requests.Response) -> dict[str, Any] | None:
        parts: list[str] = []
        try:
            for line in response.iter_lines(decode_unicode=True):
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                if not line or not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    continue
                chunk = json.loads(raw)
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if isinstance(content, str):
                        parts.append(content)
            if not parts:
                return None
            return self._load_json_content("".join(parts))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("解析 Omni 流式输出失败: %s", exc)
            return None

    @staticmethod
    def _frame_to_base64(path: Path, max_width: int = 640) -> str | None:
        if not path.exists():
            logger.debug("帧文件不存在: %s", path)
            return None
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            logger.debug("无法打开帧 %s: %s", path, e)
            return None

        w, h = img.size
        if w > max_width:
            ratio = max_width / w
            img = img.resize((max_width, int(h * ratio)), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """去掉 ```json ... ``` 等 markdown 代码块包裹。"""
        text = text.strip()
        for prefix in ("```json", "```jsonl", "```"):
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
        return text.strip()

    @classmethod
    def _load_json_content(cls, text: str) -> dict[str, Any]:
        cleaned = cls._strip_markdown(text)
        starts = [index for index in (cleaned.find("{"), cleaned.find("[")) if index >= 0]
        if not starts:
            raise json.JSONDecodeError("No JSON object found", cleaned, 0)
        value, _end = json.JSONDecoder().raw_decode(cleaned[min(starts):])
        if not isinstance(value, dict):
            raise json.JSONDecodeError("Expected JSON object", cleaned, 0)
        return value

    @staticmethod
    def _parse_analysis(data: dict | None) -> dict[str, Any] | None:
        if data is None:
            return None
        if not isinstance(data, dict):
            return None

        # 检查关键字段
        for key in ("scene_type", "generation_route", "has_human"):
            if key in data:
                return data

        # 可能是嵌套格式 {"analysis": {...}} 或 {"result": {...}}
        for wrapper in ("analysis", "result", "data", "segment"):
            val = data.get(wrapper)
            if isinstance(val, dict):
                return val
        return data  # 兜底返回原始 dict

    @staticmethod
    def _fallback_result(segment: Segment) -> dict[str, Any]:
        return {
            "segment_id": segment.segment_id,
            "start": segment.start,
            "end": segment.end,
            "has_human": segment.segment_type == "with_human",
            "has_product": False,
            "main_subject": "unknown",
            "scene_type": "unknown",
            "generation_route": "manual_review",
            "needs_manual_review": True,
            "description": "",
        }


# ── Prompt 模板 ───────────────────────────────────────────────────────

_IDENTITY_CHECK_PROMPT = """这些帧来自同一段视频。图片按上传顺序编号为 1、2、3...
{hint}
请跟踪每张图中的主角（主要完整人物），优先按脸部身份判断。姿势、表情、动作、角度、遮挡和景别变化都不代表换人。其他人的手臂、手、肩膀、背影或局部身体短暂进入画面，也不代表主角换人。
只有当原主角确实离开或不再是主角，并由另一张不同的脸/不同人物接替时，same_person 才为 false。如果同一张脸在前后仍持续出现，必须返回 true。

请仔细看：这段视频里的主角是**同一个人**，还是**中途换成了另一个人**？
如果中途换人了，不要估算秒数。请找出：
1. 只包含旧人物的最后一张图；
2. 只包含新人物的第一张图。
如果中间有叠化、两个人同时出现、身份不清或转场画面，不要把这些图算作任一人物的纯净画面。

输出 JSON 格式（不要 markdown 包裹）：
{{
    "same_person": true,
    "last_old_person_frame": 旧人物最后一张纯净图片的编号或 null,
    "first_new_person_frame": 新人物第一张纯净图片的编号或 null,
    "new_person_frame": 新人物最早出现的图片编号或 null,
    "switch_after_frame": 换人发生在第几张图之后或 null,
    "reasoning": "理由（中文，一句话）"
}}

说明：如果第 1、2 张是旧人物，第 3 张是转场，第 4、5 张是新人物，则 last_old_person_frame=2，first_new_person_frame=4。"""

_FULL_VIDEO_PROMPT = """你是一个专业的视频分镜分析助手。下面是一个完整视频的所有采样帧（共 {frame_count} 帧），按时间顺序排列。

帧编号与时间对应：{time_info}
视频范围：{start_time}s ~ {end_time}s

{audio_context}

请完成以下任务：

**1. 识别场景分段边界**
仔细观察每一帧的画面内容，找出内容/主题发生重大变化的位置（场景切换、人物出场/离场、产品出现、PPT切换、主题变化等）。
在变化点标记边界时间戳。

**2. 判断每个场景的类型**
对每个识别出的场景，从以下类别中选择最匹配的：
- person_talking: 人物口播（有人面对镜头说话）
- person_presenting_product: 人物展示产品（人和产品同屏）
- product_only: 纯产品展示（无人）
- product_demo: 产品操作演示（可能有人手）
- text_slide: 文字页面/PPT/字幕
- certificate: 证书/资质/奖状
- split_screen: 分屏/画中画
- screen_recording: 屏幕录制
- broll: 空镜/风景/环境
- interview: 采访/对话
- lecture: 讲座/培训
- intro: 片头
- outro: 片尾
- unknown: 不确定

**3. 判断每个场景的生成路线**
- human_driver: 人物迁移/口播复刻
- human_driver_with_product_reference: 人物驱动+产品参考
- image_to_video: 图生视频/静图推拉
- product_replication: 产品素材复刻
- graphic_animation: 图文动效
- manual_review: 人工确认

**4. 建立全片角色和动作时间线**
- characters: 当前场景出现的稳定视觉人物称呼，例如“蓝衣女人”“白衣年轻女人”
- character_details: 每个人物的视觉称呼、外观描述，以及只能从对白明确得到的角色名
- key_action: 当前场景最关键的可见动作；没有则为空字符串
- boundary_kind: 该场景开始的原因，只能是 shot_change / role_change / action_change / location_change / topic_change / video_start
- boundary_reason: 为什么这里应当成为语义边界
- boundary_confidence: 0到1；不确定时必须降低，不要猜测

以 JSON 格式输出：
{{
    "scenes": [
        {{
            "start": 边界开始时间（秒，float，基于帧的时间戳推断）,
            "end": 边界结束时间（秒，float）,
            "description": "画面描述（中文，一句话）",
            "scene_type": "类型标签",
            "has_human": true/false,
            "has_product": true/false,
            "main_subject": "human / product / human_product / graphic_text / mixed",
            "generation_route": "生成路线标签",
            "needs_manual_review": true/false,
            "characters": ["稳定视觉人物称呼"],
            "character_details": [{{
                "visual_label": "稳定视觉人物称呼",
                "role_name": "对白明确称呼的角色名；无法确认则为空字符串",
                "description": "外观、服饰、年龄段、声音等可区分特征",
                "confidence": 0.0到1.0
            }}],
            "key_action": "关键可见动作或空字符串",
            "boundary_kind": "shot_change / role_change / action_change / location_change / topic_change / video_start",
            "boundary_reason": "边界理由",
            "boundary_confidence": 0.0到1.0
        }}
    ]
}}

注意：
- 帧是等间隔采样的，两帧之间的精确边界时间请取两帧时间的平均值
- 如果相邻场景的类型相同，说明画面连续，不要强行分割
- 尽量不要产生 <2 秒的过短片段
"""

_VIDEO_URL_PROMPT = """你是一个专业的视频预导演。请直接观看这个原视频，包含画面、动作、人物进出、遮挡、同镜头换人、音频和对白。视频总时长约 {duration}s。

你的任务不是生成视频，而是为后续自动切分、角色资产库、片段导演提供结构化时间线。

请完成：
1. 识别全片故事摘要、人物关系网、角色称呼候选、人物出现时间线。
2. 找出语义软切点：场景切换、角色切换、动作切换、地点切换、话题切换。
3. 对每个场景给出开始/结束时间，时间必须基于原视频全局时间轴，单位秒。
4. role_name 只填对白中明确叫出的姓名或称呼；但 role_candidates 必须结合对白、年龄、互动关系和剧情语义给出候选关系称呼，例如“婆婆”“儿媳”“丈夫”“妻子”“女儿”“父亲”“母亲”“老板”“债主”“小三”“女主”。不要把“美女/男人”当角色候选。
5. 尽量不要产生小于 2 秒的碎片；不确定的边界降低 confidence。

只输出严格 JSON，不要 markdown：
{{
  "story_summary": "全片故事一句话摘要",
  "scenes": [
    {{
      "start": 0.0,
      "end": 2.5,
      "description": "画面和剧情描述",
      "scene_type": "person_talking / person_presenting_product / product_only / product_demo / text_slide / certificate / split_screen / screen_recording / broll / interview / lecture / intro / outro / unknown",
      "has_human": true,
      "has_product": false,
      "main_subject": "human / product / human_product / graphic_text / mixed",
      "generation_route": "human_driver / human_driver_with_product_reference / image_to_video / product_replication / graphic_animation / manual_review",
      "needs_manual_review": false,
      "characters": ["稳定视觉人物称呼"],
      "character_details": [
        {{
          "visual_label": "稳定视觉人物称呼",
          "role_name": "对白明确称呼的姓名或称呼；无法确认则为空字符串",
          "role_candidates": ["根据人物关系和对白推断的候选称呼，如婆婆/儿媳/丈夫/女儿"],
          "relationships": ["与其他人物的关系，如：可能是白衣年轻女人的婆婆"],
          "description": "外观、服装、年龄段、声音等可区分特征",
          "confidence": 0.0
        }}
      ],
      "key_action": "关键可见动作或空字符串",
      "boundary_kind": "shot_change / role_change / action_change / location_change / topic_change / video_start",
      "boundary_reason": "边界理由",
      "boundary_confidence": 0.0
    }}
  ]
}}
"""

_MODE2_PLAN_PROMPT = """你是 Mode2 短剧预导演和视觉资产整理员。请在同一次视觉分析中完成故事理解、语义场景时间线和可执行资产清单。

输入方式：{source_mode}
视频总时长约 {duration}s。
本地分析帧编号与时间：{frame_manifest}
{audio_context}

分析原则：
1. 角色必须按稳定的脸、发型、体型和服装连续性识别，同一人在不同景别、姿势和遮挡下仍使用同一个 identity。matched_role 只有画面或对白有依据时填写，不确定则留空。
2. 场景按真实物理空间识别。卧室、走廊、车内、庭院等不同空间必须分开；同一房间的正反打、远近景可以共用同一 physical_scene。
3. 物品必须在给出的画面或视频中清楚可见。严禁根据剧情摘要、对白、场景常识或关键词猜测物品；只在叙述中提到但画面没有看见的物品不能进入 asset_manifest。
4. evidence_times 必须来自实际可见画面。representative_frame_index 是“本地分析帧编号与时间”中的 1-based 编号，选择最清晰、遮挡最少、最能代表该资产的一帧；没有可靠对应帧时填 null。
5. 每个稳定人物、每个独立物理场景、每个明确可见且对剧情有用的物品分别建立结果。混合不清或无法确认时 kind=mixed，不能伪装成可用资产。
6. scenes 使用原视频全局秒数，描述画面、人物和可见动作；尽量避免小于 2 秒的无意义碎片。

只输出严格 JSON，不要 markdown。结构必须是：
{{
  "story_summary": "全片故事摘要",
  "scenes": [
    {{
      "start": 0.0,
      "end": 4.0,
      "description": "画面和剧情描述",
      "scene_type": "person_talking / interview / broll / unknown",
      "characters": ["稳定视觉人物称呼"],
      "character_details": [
        {{
          "visual_label": "稳定视觉人物称呼",
          "role_name": "对白明确称呼；否则为空",
          "role_candidates": ["有依据的关系称呼候选"],
          "relationships": ["与其他人物的关系"],
          "description": "可区分的外观与服装",
          "confidence": 0.0
        }}
      ],
      "key_action": "关键可见动作",
      "boundary_kind": "shot_change / role_change / action_change / location_change / topic_change / video_start",
      "boundary_reason": "边界理由",
      "boundary_confidence": 0.0
    }}
  ],
  "asset_manifest": {{
    "results": [
      {{
        "group_id": "R001 / S001 / P001",
        "kind": "role / scene / prop / mixed / ignore",
        "name": "视觉确认的资产名称",
        "identity": "稳定人物身份聚类键；非人物为空",
        "matched_role": "有依据的角色名或关系称呼；不确定为空",
        "physical_scene": "稳定物理空间键；非场景可填所属空间",
        "evidence_times": [1.2, 3.4],
        "representative_frame_index": 1,
        "visible_props": [
          {{"name": "画面明确可见的物品", "evidence_times": [3.4], "confidence": 0.9}}
        ],
        "confidence": 0.0,
        "reason": "视觉证据与代表帧选择理由"
      }}
    ]
  }}
}}
"""

_SEGMENT_ANALYSIS_PROMPT = """你是一个专业的视频内容分析助手。下面是一个视频片段的几帧关键画面（{start}s ~ {end}s，时长 {duration}s）。

请仔细观察这几张图，理解这个片段的内容，然后输出结构化的 JSON 分析结果。

按照以下三个层次判断：

**第一层：是否有人**
- has_human: true/false
- has_product: true/false（是否有产品/商品出现在画面中）

**第二层：主体和场景类型**
main_subject 必须是以下之一：
- "human": 人是画面主体
- "product": 产品是画面主体
- "human_product": 人和产品同屏
- "graphic_text": 图文/证书/字幕/PPT
- "mixed": 分屏/混合画面

scene_type 必须是以下之一：
- "person_talking": 人物口播（人面对镜头说话）
- "person_presenting_product": 人物展示产品（手持/操作）
- "product_only": 纯产品展示（无人）
- "product_demo": 产品操作演示（可能有人手入镜）
- "text_slide": 文字页面/PPT/字幕
- "certificate": 证书/资质/奖状
- "split_screen": 分屏画面（画中画/左右分屏）
- "screen_recording": 屏幕录制/软件操作
- "broll": 空镜/B-roll/场景环境
- "interview": 采访/对话
- "lecture": 讲座/培训
- "intro": 片头
- "outro": 片尾
- "unknown": 不确定

**第三层：建议生成路线**
generation_route 必须是以下之一：
- "human_driver": 人物迁移/口播复刻（适合有人说话的场景）
- "human_driver_with_product_reference": 人物驱动+产品参考（人和产品同屏）
- "image_to_video": 图生视频/静图推拉（适合产品静态展示）
- "product_replication": 产品素材复刻/3D渲染（适合产品多角度展示）
- "graphic_animation": 图文动效/文字动画（适合PPT/证书）
- "manual_review": 人工确认（不确定时选这个）

输出格式（严格 JSON，不要 markdown 包裹）：
{{
    "has_human": true,
    "has_product": false,
    "main_subject": "human",
    "scene_type": "person_talking",
    "generation_route": "human_driver",
    "visual_style": "画面描述（一句话，中文）",
    "description": "内容描述（1-2句中文）",
    "needs_manual_review": false
}}"""

# ── 便利入口 ──────────────────────────────────────────────────────────


def create_client(
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
) -> GeminiClient:
    """从环境变量或参数创建 GeminiClient。

    API Key 优先级：参数 > 环境变量 GEMINI_API_KEY > 环境变量 OPENAI_API_KEY
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "需要提供 API Key。可传入 api_key 参数或设置 GEMINI_API_KEY / OPENAI_API_KEY 环境变量"
        )
    return GeminiClient(api_key=key, base_url=base_url, model=model)


# ── 场景分类映射表 ────────────────────────────────────────────────────

# 生成路线 → 目标目录名
GENERATION_DIR_MAP: dict[str, str] = {
    "human_driver": "01_人物驱动",
    "human_driver_with_product_reference": "02_人物加产品",
    "image_to_video": "03_图生视频",
    "product_replication": "04_产品复刻",
    "graphic_animation": "05_图文动效",
    "manual_review": "99_人工确认",
}

# scene_type → 中文标签
SCENE_TYPE_LABELS: dict[str, str] = {
    "person_talking": "人物口播",
    "person_presenting_product": "人物+产品展示",
    "product_only": "纯产品展示",
    "product_demo": "产品操作演示",
    "text_slide": "图文/PPT",
    "certificate": "证书资质",
    "split_screen": "分屏画面",
    "screen_recording": "屏幕录制",
    "broll": "空镜/B-roll",
    "interview": "采访对话",
    "lecture": "讲座培训",
    "intro": "片头",
    "outro": "片尾",
    "unknown": "未识别",
}

# generation_route → 中文标签
GENERATION_ROUTE_LABELS: dict[str, str] = {
    "human_driver": "人物迁移/口播复刻",
    "human_driver_with_product_reference": "人物驱动+产品参考",
    "image_to_video": "图生视频/静图推拉",
    "product_replication": "产品素材复刻",
    "graphic_animation": "图文动效",
    "manual_review": "人工确认",
}


__all__ = [
    "GeminiClient",
    "create_client",
    "GENERATION_DIR_MAP",
    "SCENE_TYPE_LABELS",
    "GENERATION_ROUTE_LABELS",
    "_IDENTITY_CHECK_PROMPT",
]
