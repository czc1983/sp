import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from spvideo.wan22_client import Wan22MixClient


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        payload: dict | None = None,
        headers: dict | None = None,
        content: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}
        self._content = content
        self.closed = False

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error", response=self)

    def close(self) -> None:
        self.closed = True

    def iter_content(self, chunk_size: int = 8192):
        yield self._content


class Wan22ClientRetryTests(unittest.TestCase):
    def test_upload_retries_file_info_rate_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "subject.png"
            path.write_bytes(b"image")
            responses = [
                FakeResponse(200, {"data": {"uploaded_files": [{"file_id": "file_1"}]}}),
                FakeResponse(429, headers={"Retry-After": "0"}),
                FakeResponse(200, {"data": {"url": "https://oss.example/subject.png"}}),
            ]
            calls: list[tuple[str, str]] = []

            def fake_request(method: str, url: str, **kwargs):
                calls.append((method, url))
                return responses.pop(0)

            client = Wan22MixClient("sk-test")
            with patch("spvideo.wan22_client.requests.request", side_effect=fake_request), patch(
                "spvideo.wan22_client.time.sleep"
            ) as sleep:
                result = client.upload(str(path))

        self.assertEqual(result, "https://oss.example/subject.png")
        self.assertEqual(
            calls,
            [
                ("post", Wan22MixClient.UPLOAD_URL),
                ("get", f"{Wan22MixClient.UPLOAD_URL}/file_1"),
                ("get", f"{Wan22MixClient.UPLOAD_URL}/file_1"),
            ],
        )
        sleep.assert_called_once_with(0.0)

    def test_download_retries_rate_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "out.mp4"
            responses = [
                FakeResponse(429, headers={"Retry-After": "0"}),
                FakeResponse(200, content=b"video"),
            ]

            def fake_request(method: str, url: str, **kwargs):
                return responses.pop(0)

            client = Wan22MixClient("sk-test")
            with patch("spvideo.wan22_client.requests.request", side_effect=fake_request), patch(
                "spvideo.wan22_client.time.sleep"
            ) as sleep:
                result = client.download_video("https://oss.example/out.mp4", str(output))
                content = output.read_bytes()

        self.assertEqual(result, str(output))
        self.assertEqual(content, b"video")
        sleep.assert_called_once_with(0.0)


if __name__ == "__main__":
    unittest.main()
