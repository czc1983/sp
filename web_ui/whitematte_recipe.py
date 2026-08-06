# -*- coding: utf-8 -*-
r"""白膜人偶重生配方 v3（2026-07-31 多轮实测验证）

验证过的成功链路（零画面文字理解）：
  原镜头切片 → 时间拉伸慢放凑时长（setpts 定格复制，不循环、不倒放、不加假内容）
  → 前中后内容锚点（构图/人数/遮挡）+ 人偶风格锚点（造型）
  → 人偶化 + 眼神表情保留提示词
  → 通道1（默认 SD 2.0 720P；SD 2.0 720P_官转 故障时的备用就是它）
  → 抽帧加速还原原节奏（select 每 N 帧取 1 帧）

实测结论（E:\fy\white_matte_test）：
  - 倒放/原速凑时长会把虚化弱信号主体（第二人脸）冲掉；慢放可保住
  - 多锚点+慢放是"保真"取向，人偶化必须显式提示词 + 风格锚图双管齐下
  - 眼神/眼睑/表情最容易被风格化牺牲，必须在提示词里显式声明保留
  - 彩色人偶可行，可按人分配颜色（如：前景深蓝、后景暖红）
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

WM_DIR = Path(__file__).resolve().parent
DEFAULT_STYLE_REF = WM_DIR / "assets" / "whitematte_style_mannequin.png"

DEFAULT_MODEL = "SD 2.0 720P"  # “SD 2.0 720P_官转” 上游故障时的实测可用模型
DEFAULT_RATIO = "9:16"
TARGET_MIN_SECONDS = 4.6   # 慢放后参考视频至少这么长（模型最小时长 4s）
NO_STRETCH_SECONDS = 4.2   # 原片长于这个值就不用慢放
MAX_SPEED = 16             # 实测 16x（0.34s 镜头）可用
REF_FPS = 25               # 慢放参考视频帧率
OUT_FPS = 24               # 生成结果实际帧率（通道1 返回 24fps）

DOLL_COLOR_PRESETS = {
    "white": "白色",
    "red": "红色",
    "blue": "蓝色",
    "black": "黑色",
}

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


# ---------------------------------------------------------------- 基础工具

_FFMPEG_FULL: str | None = None


def _ffmpeg() -> str:
    """选一个带完整滤镜（setpts/fps/select）的 ffmpeg。

    注意：spvideo 候选里的 remotion 精简版编译时 --disable-filters，
    不能用于慢放/抽帧，必须探测后跳过。
    """
    global _FFMPEG_FULL
    if _FFMPEG_FULL:
        return _FFMPEG_FULL
    from spvideo import ffmpeg_tools
    for candidate in ffmpeg_tools.FFMPEG_CANDIDATES:
        found = shutil.which(candidate)
        if not found and Path(candidate).exists():
            found = str(Path(candidate))
        if not found:
            continue
        try:
            probe = subprocess.run(
                [found, "-hide_banner", "-filters"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=30, **ffmpeg_tools.subprocess_no_window_kwargs(),
            )
            if probe.returncode == 0 and "setpts" in (probe.stdout or "") and " fps " in (probe.stdout or ""):
                _FFMPEG_FULL = found
                return found
        except Exception:
            continue
    _FFMPEG_FULL = ffmpeg_tools.ffmpeg_path()
    return _FFMPEG_FULL


def _run_ffmpeg(args: list[str]) -> None:
    from spvideo.ffmpeg_tools import subprocess_no_window_kwargs
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **subprocess_no_window_kwargs(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败: {result.stderr[-400:]}")


def probe_duration(video_path: str | Path) -> float:
    try:
        from spvideo.ffmpeg_tools import probe_video
        return float(probe_video(video_path).duration)
    except Exception:
        pass
    # 兜底：解析 ffmpeg -i 的 stderr
    from spvideo.ffmpeg_tools import subprocess_no_window_kwargs
    result = subprocess.run(
        [_ffmpeg(), "-i", str(video_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        **subprocess_no_window_kwargs(),
    )
    import re
    match = re.search(r"Duration: (\d+):(\d+):([\d.]+)", result.stderr or "")
    if not match:
        raise RuntimeError(f"无法读取视频时长: {video_path}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def has_video_stream(video_path: str | Path) -> bool:
    try:
        from spvideo.ffmpeg_tools import probe_video
        meta = probe_video(video_path)
        return bool(meta.video_codec) and meta.fps > 0
    except Exception:
        # 探测失败时保守放行，交给后续 ffmpeg 报错
        return True


# ---------------------------------------------------------------- 配方步骤

def compute_speed(duration: float) -> int:
    """时间拉伸倍数：1 表示不需要慢放。"""
    if duration >= NO_STRETCH_SECONDS:
        return 1
    return min(MAX_SPEED, max(2, math.ceil(TARGET_MIN_SECONDS / max(duration, 0.04))))


def build_slow_reference(clip_path: Path, work_dir: Path, speed: int) -> tuple[Path, float]:
    """setpts 定格复制式慢放（不插值、不倒放、不循环）。"""
    if speed <= 1:
        return clip_path, probe_duration(clip_path)
    out = work_dir / f"slow_ref_{speed}x.mp4"
    _run_ffmpeg([
        _ffmpeg(), "-y", "-i", str(clip_path),
        "-filter:v", f"setpts={speed}*PTS,fps={REF_FPS}",
        "-an", "-pix_fmt", "yuv420p", str(out),
    ])
    return out, probe_duration(out)


def extract_anchor_frames(clip_path: Path, duration: float, work_dir: Path, count: int = 3) -> list[Path]:
    """前中后均匀抽内容锚点帧。"""
    fractions = {
        1: [0.5],
        2: [0.08, 0.92],
        3: [0.08, 0.5, 0.92],
        4: [0.06, 0.35, 0.65, 0.94],
        5: [0.05, 0.28, 0.5, 0.72, 0.95],
        6: [0.05, 0.24, 0.42, 0.6, 0.78, 0.95],
    }.get(max(1, min(6, count)), [0.08, 0.5, 0.92])
    anchors: list[Path] = []
    for index, frac in enumerate(fractions, 1):
        t = min(max(0.05, duration * frac), max(0.05, duration - 0.05))
        out = work_dir / f"anchor_{index}.jpg"
        _run_ffmpeg([
            _ffmpeg(), "-y", "-ss", f"{t:.3f}", "-i", str(clip_path),
            "-frames:v", "1", "-q:v", "3", str(out),
        ])
        anchors.append(out)
    return anchors


def build_doll_prompt(n_content: int, doll: str = "white", custom_color: str = "",
                      emotion: str = "") -> str:
    """人偶化 + 眼神表情保留提示词。参考图 1~n 为内容锚点，第 n+1 张为风格锚点。"""
    style_idx = n_content + 1
    if doll == "white":
        doll_desc = ("白色关节人偶：木质/树脂模特、光头没有头发、没有眉毛和睫毛、"
                     "身体带球形关节、光滑哑光材质")
    else:
        color_text = DOLL_COLOR_PRESETS.get(doll, "") if doll != "custom" else ""
        color_text = (custom_color or color_text or "彩色").strip()
        doll_desc = (f"彩色树脂关节人偶（{color_text}）：光头没有头发、没有眉毛和睫毛、"
                     "身体带球形关节、光滑哑光质感")
    emotion_clause = f"表情保持原片的{emotion.strip()}神态，" if emotion.strip() else ""
    return (
        f"把视频里的人物全部转换成参考图{style_idx}那样的{doll_desc}，"
        "但眼睛要做成睁开的、有明确视线方向的造型；背景和物体变成纯白色哑光材质。"
        f"参考图1到参考图{n_content}是同一个镜头按时间先后顺序抽取的画面，"
        f"参考图{style_idx}是人物造型的风格示例。"
        f"严格按照参考图1~{n_content}的构图、人物数量、位置、姿态和遮挡关系演变，仅保留柔和光影明暗。"
        "最关键的要求：必须准确保留原片人物的眼神和表情——眼睛保持睁开，眼睑开合程度与参考图一致，"
        f"视线方向与参考图完全一致，{emotion_clause}情绪状态不变，绝不能改成闭眼、垂眼或看向别处；"
        "不要真实皮肤质感、不要头发丝、不要睫毛。"
    )


def restore_original_speed(generated_path: Path, restored_path: Path, speed: int) -> Path:
    """抽帧加速还原：每 speed 帧取 1 帧，时长精确回到原镜头。"""
    if speed <= 1:
        shutil.copyfile(generated_path, restored_path)
        return restored_path
    _run_ffmpeg([
        _ffmpeg(), "-y", "-i", str(generated_path),
        "-vf", f"select=not(mod(n\\,{speed})),setpts=N/({OUT_FPS}*TB)",
        "-r", str(OUT_FPS), "-an", "-pix_fmt", "yuv420p", str(restored_path),
    ])
    return restored_path


# ---------------------------------------------------------------- 任务流

def _work_root(project_dir: str) -> Path:
    base = Path(project_dir) if project_dir else Path(r"E:\sp")
    candidate = base / "04_AI输出成片"
    if not candidate.exists() and base.name == "04_AI输出成片":
        candidate = base
    root = candidate / "whitematte_jobs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def submit_shot_retake(
    payload: dict[str, Any],
    submit_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    clip_path = Path(str(payload.get("clip_path") or payload.get("video_path") or "").strip())
    if not clip_path.is_file():
        raise ValueError(f"镜头切片不存在: {clip_path}")
    if not has_video_stream(clip_path):
        raise ValueError("该切片没有视频流（可能是纯音频切片），无法重生")

    shot_id = str(payload.get("shot_id") or clip_path.stem).strip() or clip_path.stem
    project_dir = str(payload.get("project_dir") or payload.get("projectDir") or "").strip()
    package_id = str(payload.get("package_id") or payload.get("packageId") or "").strip()
    doll = str(payload.get("doll") or "white").strip() or "white"
    custom_color = str(payload.get("custom_color") or "").strip()
    emotion = str(payload.get("emotion") or "").strip()
    anchor_count = int(payload.get("anchor_count") or 6)
    model = str(payload.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    ratio = str(payload.get("ratio") or DEFAULT_RATIO).strip() or DEFAULT_RATIO
    style_ref = Path(str(payload.get("style_ref") or DEFAULT_STYLE_REF))
    if not style_ref.is_file():
        raise ValueError(f"人偶风格锚图缺失: {style_ref}")

    work_dir = _work_root(project_dir) / f"{shot_id}_{uuid.uuid4().hex[:8]}"
    work_dir.mkdir(parents=True, exist_ok=True)

    duration = probe_duration(clip_path)
    speed = compute_speed(duration)
    slow_ref, slow_duration = build_slow_reference(clip_path, work_dir, speed)
    anchors = extract_anchor_frames(clip_path, duration, work_dir, anchor_count)
    prompt = build_doll_prompt(len(anchors), doll=doll, custom_color=custom_color, emotion=emotion)
    output_duration = min(15, max(4, int(round(slow_duration))))

    response = submit_fn({
        "payload": {
            "model": model,
            "prompt": prompt,
            "ratio": ratio,
            "duration": output_duration,
            "video_urls": [str(slow_ref)],
            "video_durations": [round(slow_duration, 3)],
            "image_urls": [str(p) for p in anchors] + [str(style_ref)],
        },
        "project_dir": project_dir,
        "package_id": package_id,
        "source_video_path": str(clip_path),
        "taskName": f"whitematte_doll_{shot_id}",
        "output_role": "mask",
    })
    task_id = str(response.get("task_id") or response.get("taskId") or "").strip()
    if not task_id:
        raise RuntimeError(f"通道1未返回 task_id: {json.dumps(response, ensure_ascii=False)[:300]}")

    job = {
        "task_id": task_id,
        "shot_id": shot_id,
        "clip_path": str(clip_path),
        "work_dir": str(work_dir),
        "speed": speed,
        "slow_ref": str(slow_ref),
        "slow_duration": slow_duration,
        "anchor_count": len(anchors),
        "output_duration": output_duration,
        "model": model,
        "doll": doll,
        "prompt": prompt,
        "created_at": time.time(),
        "restored_path": "",
    }
    with JOBS_LOCK:
        JOBS[task_id] = job
    (work_dir / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "success": True,
        "task_id": task_id,
        "shot_id": shot_id,
        "work_dir": str(work_dir),
        "speed": speed,
        "slow_duration": round(slow_duration, 3),
        "anchor_count": len(anchors),
        "output_duration": output_duration,
        "model": model,
        "prompt": prompt,
    }


def _load_job(task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(task_id)
    if job:
        return job
    # 服务重启后内存丢失：允许前端把 submit 响应里的关键参数带回来兜底
    work_dir = str(payload.get("work_dir") or "").strip()
    job = {
        "task_id": task_id,
        "shot_id": str(payload.get("shot_id") or "").strip(),
        "work_dir": work_dir,
        "speed": int(payload.get("speed") or 1),
        "restored_path": "",
    }
    if work_dir:
        job_file = Path(work_dir) / "job.json"
        if job_file.is_file():
            try:
                job.update(json.loads(job_file.read_text(encoding="utf-8")))
            except Exception:
                pass
    with JOBS_LOCK:
        JOBS[task_id] = job
    return job


def poll_shot_retake(
    payload: dict[str, Any],
    query_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    task_id = str(payload.get("task_id") or payload.get("taskId") or "").strip()
    if not task_id:
        raise ValueError("task_id 必填")
    job = _load_job(task_id, payload)
    status = query_fn({"task_id": task_id})

    restored_path = str(job.get("restored_path") or "")
    if (
        status.get("downloaded")
        and not restored_path
        and int(job.get("speed") or 1) >= 1
    ):
        generated = Path(str(status.get("output_path") or ""))
        work_dir = Path(str(job.get("work_dir") or "")) if job.get("work_dir") else generated.parent
        if generated.is_file():
            restored = work_dir / f"{job.get('shot_id') or 'shot'}_doll_restored.mp4"
            try:
                restore_original_speed(generated, restored, int(job.get("speed") or 1))
                restored_path = str(restored)
                job["restored_path"] = restored_path
                with JOBS_LOCK:
                    JOBS[task_id] = job
                try:
                    (work_dir / "job.json").write_text(
                        json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass
            except Exception as error:  # noqa: BLE001
                status["restore_error"] = str(error)

    status["speed"] = int(job.get("speed") or 1)
    status["shot_id"] = str(job.get("shot_id") or "")
    status["restored_path"] = restored_path
    return status
