from __future__ import annotations

import unittest
import tempfile
from types import SimpleNamespace
from pathlib import Path

from spvideo.gemini_analyzer import GeminiClient


class GeminiParserTests(unittest.TestCase):
    def test_full_video_timeout_formula_is_capped(self) -> None:
        source = Path(__file__).resolve().parents[1] / "spvideo" / "gemini_analyzer.py"
        text = source.read_text(encoding="utf-8")
        self.assertIn("timeout = min(600, max(180, len(b64_list) * 3))", text)

    def test_trailing_markdown_fence_is_ignored(self) -> None:
        value = GeminiClient._load_json_content('{"scenes": []}\n```')
        self.assertEqual(value, {"scenes": []})

    def test_leading_text_and_json_fence_are_ignored(self) -> None:
        value = GeminiClient._load_json_content('结果如下：\n```json\n{"status":"ok"}\n```')
        self.assertEqual(value["status"], "ok")

    def test_same_scene_type_with_different_characters_does_not_merge(self) -> None:
        client = object.__new__(GeminiClient)
        client._call_full_analysis = lambda _frames, **_kwargs: [
            {"start": 0.0, "end": 1.0, "scene_type": "person_talking", "characters": ["A"], "boundary_kind": "video_start"},
            {"start": 1.0, "end": 2.0, "scene_type": "person_talking", "characters": ["B"], "boundary_kind": "role_change"},
        ]
        scenes = client.analyze_full_video(
            [SimpleNamespace(time=0.0), SimpleNamespace(time=1.0)],
            batch_size=10,
            overlap=0,
        )
        self.assertEqual(len(scenes), 2)

    def test_explicit_role_change_does_not_merge_without_character_labels(self) -> None:
        client = object.__new__(GeminiClient)
        client._call_full_analysis = lambda _frames, **_kwargs: [
            {"start": 0.0, "end": 1.0, "scene_type": "person_talking", "boundary_kind": "video_start"},
            {"start": 1.0, "end": 2.0, "scene_type": "person_talking", "boundary_kind": "role_change"},
        ]
        scenes = client.analyze_full_video(
            [SimpleNamespace(time=0.0), SimpleNamespace(time=1.0)],
            batch_size=10,
            overlap=0,
        )
        self.assertEqual(len(scenes), 2)

    def test_audio_transcription_uses_audio_content_and_returns_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio = Path(temp_dir) / "audio.mp3"
            audio.write_bytes(b"fake-mp3")
            client = object.__new__(GeminiClient)
            captured = {}
            client.model = "qwen3.5-omni-plus"
            client._post = lambda payload, timeout=None: captured.setdefault("payload", payload) or {
                "utterances": [{"start": 0.0, "end": 1.0, "text": "你好"}]
            }
            # setdefault returns the payload, so use a regular stub for clarity.
            def fake_post(payload, timeout=None):
                captured["payload"] = payload
                return {"utterances": [{"start": 0.0, "end": 1.0, "text": "你好"}]}
            client._post = fake_post

            result = client.transcribe_audio(audio)

            content = captured["payload"]["messages"][0]["content"]
            self.assertEqual(content[0]["type"], "input_audio")
            self.assertTrue(content[0]["input_audio"]["data"].startswith("data:;base64,"))
            self.assertEqual(result["utterances"][0]["text"], "你好")

    def test_timestamp_strings_are_normalized_to_seconds(self) -> None:
        self.assertAlmostEqual(GeminiClient._time_seconds("00:01.326"), 1.326)
        self.assertAlmostEqual(GeminiClient._time_seconds("01:02:03.5"), 3723.5)

    def test_qwen_omni_post_uses_and_parses_streaming_sse(self) -> None:
        class FakeResponse:
            headers = {"Content-Type": "text/event-stream;charset=utf-8"}

            def raise_for_status(self):
                return None

            def iter_lines(self, decode_unicode=True):
                yield b'data: {"choices":[{"delta":{"content":"{\\"status\\":\\""}}]}'
                yield b'data: {"choices":[{"delta":{"content":"ok\\"}"}}]}'
                yield b"data: [DONE]"

        class FakeSession:
            def __init__(self):
                self.payload = None
                self.stream = None

            def post(self, _url, *, json, timeout, stream):
                self.payload = json
                self.stream = stream
                return FakeResponse()

        client = object.__new__(GeminiClient)
        client.model = "qwen3.5-omni-plus"
        client.base_url = "https://example.invalid/v1"
        client._timeout = 10
        client._session = FakeSession()

        result = client._post({"model": client.model, "messages": []})

        self.assertEqual(result, {"status": "ok"})
        self.assertTrue(client._session.stream)
        self.assertTrue(client._session.payload["stream"])
        self.assertEqual(client._session.payload["modalities"], ["text"])

    def test_transcript_context_filters_to_current_time_window(self) -> None:
        context = GeminiClient._transcript_context({"utterances": [
            {"start": 0.0, "end": 1.0, "speaker": "A", "text": "前段"},
            {"start": 5.0, "end": 6.0, "speaker": "B", "text": "后段"},
        ]}, 4.5, 6.5)
        self.assertNotIn("前段", context)
        self.assertIn("后段", context)

    def test_video_url_prompt_formats_with_json_schema(self) -> None:
        client = object.__new__(GeminiClient)
        client._build_payload = lambda contents, **_kwargs: {"messages": [{"content": contents}]}
        captured = {}

        def fake_post(payload, timeout=None):
            captured["payload"] = payload
            return {"scenes": [{"start": 0, "end": 1, "scene_type": "person_talking"}]}

        client._post = fake_post
        client._parse_analysis = lambda data: data

        result = client.analyze_video_url("https://example.invalid/video.mp4", duration=12.3)

        self.assertEqual(result[0]["scene_type"], "person_talking")
        text_parts = [
            item["text"]
            for item in captured["payload"]["messages"][0]["content"]
            if item.get("type") == "text"
        ]
        self.assertIn("role_candidates", text_parts[0])
        self.assertIn('"story_summary"', text_parts[0])

    def test_mode2_video_url_plan_returns_story_scenes_and_assets_in_one_call(self) -> None:
        client = object.__new__(GeminiClient)
        client._build_payload = lambda contents, **_kwargs: {"messages": [{"content": contents}]}
        client._parse_analysis = lambda data: data
        captured = {"calls": 0}

        def fake_post(payload, timeout=None):
            captured["calls"] += 1
            captured["payload"] = payload
            return {
                "story_summary": "故事摘要",
                "scenes": [{"start": 0, "end": 4, "description": "卧室对话"}],
                "asset_manifest": {"results": [{
                    "group_id": "R001",
                    "kind": "role",
                    "name": "白衣女人",
                    "identity": "person_01",
                    "matched_role": "女主",
                    "physical_scene": "bedroom_01",
                    "evidence_times": [1.0],
                    "representative_frame_index": 1,
                    "visible_props": [],
                    "confidence": 0.94,
                    "reason": "正脸清晰",
                }]},
            }

        client._post = fake_post
        result = client.analyze_mode2_plan(
            duration=4.0,
            video_url="https://example.invalid/video.mp4",
            frames=[SimpleNamespace(time=1.0, path="frame.jpg")],
        )

        self.assertEqual(captured["calls"], 1)
        self.assertEqual(result["story_summary"], "故事摘要")
        self.assertEqual(result["scenes"][0]["description"], "卧室对话")
        self.assertEqual(result["asset_manifest"]["results"][0]["matched_role"], "女主")
        content = captured["payload"]["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "video_url")
        prompt = next(item["text"] for item in content if item.get("type") == "text")
        self.assertIn('"asset_manifest"', prompt)
        self.assertIn("严禁根据剧情摘要", prompt)

    def test_mode2_sampled_frames_plan_uses_one_visual_request(self) -> None:
        client = object.__new__(GeminiClient)
        client._frame_to_base64 = lambda _path, max_width=320: "encoded"
        client._build_payload = lambda contents, **_kwargs: {"messages": [{"content": contents}]}
        client._parse_analysis = lambda data: data
        captured = {"calls": 0}

        def fake_post(payload, timeout=None):
            captured["calls"] += 1
            captured["payload"] = payload
            return {
                "story_summary": "采样帧故事",
                "scenes": [],
                "asset_manifest": {"results": [{
                    "group_id": "S001",
                    "kind": "scene",
                    "name": "卧室",
                    "identity": "",
                    "matched_role": "",
                    "physical_scene": "bedroom_01",
                    "evidence_times": [3.9],
                    "representative_frame_index": None,
                    "visible_props": [],
                    "confidence": 0.9,
                    "reason": "空间完整",
                }]},
            }

        client._post = fake_post
        result = client.analyze_mode2_plan(
            duration=5.0,
            frames=[
                SimpleNamespace(time=0.0, path="f1.jpg"),
                SimpleNamespace(time=4.0, path="f2.jpg"),
            ],
        )

        self.assertEqual(captured["calls"], 1)
        content = captured["payload"]["messages"][0]["content"]
        self.assertEqual(sum(item.get("type") == "image_url" for item in content), 2)
        self.assertEqual(
            result["asset_manifest"]["results"][0]["representative_frame_index"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
