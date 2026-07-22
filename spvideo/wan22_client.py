"""
Wan2.2-Animate-Mix API 客户端
阿里百炼「视频换人」模型封装

流程:
  1. 上传本地文件 → 获取临时 OSS URL
  2. 创建异步任务
  3. 轮询任务状态
  4. 下载结果视频到本地
"""
import os
import time
import requests
import urllib3
from typing import Optional, Callable

urllib3.disable_warnings()


RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
RETRY_BASE_DELAY = 3.0
RETRY_MAX_DELAY = 45.0
RETRY_ATTEMPTS = 6


def _retry_after_seconds(response: requests.Response | None, attempt: int) -> float:
    if response is not None:
        raw = response.headers.get("Retry-After", "").strip()
        if raw:
            try:
                return max(0.0, min(RETRY_MAX_DELAY, float(raw)))
            except ValueError:
                pass
    return min(RETRY_MAX_DELAY, RETRY_BASE_DELAY * (2 ** attempt))


class Wan22MixClient:
    """Wan2.2-Animate-Mix 视频换人客户端"""

    UPLOAD_URL = "https://dashscope.aliyuncs.com/api/v1/files"
    CREATE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image2video/video-synthesis"
    TASK_URL = "https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"

    def __init__(self, api_key: str, mode: str = "wan-std"):
        """
        Args:
            api_key: 阿里百炼 API Key (sk-...)
            mode: 'wan-std' 标准模式 或 'wan-pro' 专业模式
        """
        self.api_key = api_key
        self.mode = mode
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def _request_with_retries(
        self,
        method: str,
        url: str,
        *,
        before_attempt: Optional[Callable[[], None]] = None,
        attempts: int = RETRY_ATTEMPTS,
        **kwargs,
    ) -> requests.Response:
        """Request DashScope with bounded backoff for rate limits/transient errors."""
        last_error: Exception | None = None
        for attempt in range(max(1, attempts)):
            if before_attempt:
                before_attempt()
            try:
                response = requests.request(method, url, **kwargs)
            except (requests.Timeout, requests.ConnectionError) as error:
                last_error = error
                if attempt >= attempts - 1:
                    raise
                time.sleep(_retry_after_seconds(None, attempt))
                continue

            if response.status_code not in RETRY_STATUS_CODES:
                response.raise_for_status()
                return response
            if attempt >= attempts - 1:
                response.raise_for_status()
                return response

            delay = _retry_after_seconds(response, attempt)
            response.close()
            time.sleep(delay)

        if last_error:
            raise last_error
        raise RuntimeError(f"请求失败: {url}")

    # ------------------------------------------------------------------
    # 文件上传
    # ------------------------------------------------------------------

    def upload(self, file_path: str) -> str:
        """上传本地文件到百炼，返回可供 API 使用的公网 URL

        Args:
            file_path: 本地文件路径 (图片 png/jpg 或视频 mp4/mov/avi)

        Returns:
            带签名的临时 OSS URL（约 24 小时有效）
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # Step 1: 上传 → 获取 file_id
        fname = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            resp = self._request_with_retries(
                "post",
                self.UPLOAD_URL,
                headers=self._headers,
                files={"file": (fname, f)},
                timeout=60,
                before_attempt=lambda: f.seek(0),
            )
        data = resp.json()
        uploaded = data.get("data", {}).get("uploaded_files", [])
        if not uploaded:
            raise RuntimeError(f"上传失败: {data}")
        file_id = uploaded[0]["file_id"]

        # Step 2: 查询 → 获取带签名的 URL
        info = self._request_with_retries(
            "get",
            f"{self.UPLOAD_URL}/{file_id}",
            headers=self._headers,
            timeout=30,
        )
        url = info.json()["data"]["url"]
        return url

    # ------------------------------------------------------------------
    # 视频换人
    # ------------------------------------------------------------------

    def create_task(self, image_url: str, video_url: str) -> str:
        """创建视频换人任务

        Args:
            image_url: 角色图片的公网 URL（或通过 upload() 获取）
            video_url: 参考视频的公网 URL（或通过 upload() 获取）

        Returns:
            task_id，用于后续轮询和下载
        """
        payload = {
            "model": "wan2.2-animate-mix",
            "input": {
                "image_url": image_url,
                "video_url": video_url,
                "watermark": False,
            },
            "parameters": {"mode": self.mode},
        }

        resp = self._request_with_retries(
            "post",
            self.CREATE_URL,
            headers={
                **self._headers,
                "Content-Type": "application/json",
                "X-DashScope-Async": "enable",
            },
            json=payload,
            timeout=30,
        )
        data = resp.json()
        task_id = data["output"]["task_id"]
        return task_id

    def poll_task(
        self,
        task_id: str,
        interval: int = 15,
        timeout: int = 600,
        on_status: Optional[Callable[[str, float], None]] = None,
    ) -> dict:
        """轮询任务直到完成

        Args:
            task_id: 任务 ID
            interval: 轮询间隔（秒）
            timeout: 超时（秒）
            on_status: 状态回调 on_status(status, elapsed_seconds)

        Returns:
            完整的任务响应 dict

        Raises:
            TimeoutError: 超时
            RuntimeError: 任务失败
        """
        start = time.time()
        url = self.TASK_URL.format(task_id=task_id)

        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                raise TimeoutError(f"任务超时 ({timeout}s)")

            resp = self._request_with_retries("get", url, headers=self._headers, timeout=10)
            data = resp.json()
            status = data.get("output", {}).get("task_status", "UNKNOWN")

            if on_status:
                on_status(status, elapsed)

            if status == "SUCCEEDED":
                return data
            elif status == "FAILED":
                err = data.get("output", {})
                raise RuntimeError(
                    f"任务失败: code={err.get('code')}, message={err.get('message')}"
                )
            elif status in ("PENDING", "RUNNING"):
                time.sleep(interval)
            else:
                time.sleep(interval)

    def download_video(self, video_url: str, output_path: str) -> str:
        """下载生成的视频到本地

        Args:
            video_url: 任务结果中的 video_url
            output_path: 本地保存路径

        Returns:
            output_path
        """
        resp = self._request_with_retries("get", video_url, stream=True, timeout=300)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        try:
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
        finally:
            resp.close()

        return output_path

    # ------------------------------------------------------------------
    # 一站式接口
    # ------------------------------------------------------------------

    def transfer(
        self,
        image_path: str,
        video_path: str,
        output_path: str,
        poll_interval: int = 15,
        on_status: Optional[Callable[[str, float], None]] = None,
    ) -> dict:
        """一站式：上传 → 创建 → 轮询 → 下载

        Args:
            image_path: 本地角色图片路径（或 http URL）
            video_path: 本地视频路径（或 http URL）
            output_path: 输出视频路径
            poll_interval: 轮询间隔（秒）
            on_status: 状态回调

        Returns:
            {"output_path": ..., "duration": ..., "mode": ..., "task_id": ...}
        """
        # 1. 准备 URL
        image_url = image_path if image_path.startswith("http") else self.upload(image_path)
        video_url = video_path if video_path.startswith("http") else self.upload(video_path)

        # 2. 创建任务
        task_id = self.create_task(image_url, video_url)

        # 3. 轮询
        result = self.poll_task(task_id, interval=poll_interval, on_status=on_status)

        # 4. 下载
        video_url_out = result["output"]["results"]["video_url"]
        usage = result.get("usage", {})
        self.download_video(video_url_out, output_path)

        return {
            "output_path": output_path,
            "duration": usage.get("video_duration", 0),
            "mode": usage.get("video_ratio", "unknown"),
            "task_id": task_id,
        }
