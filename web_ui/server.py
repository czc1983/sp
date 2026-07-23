from __future__ import annotations

import copy
import base64
import json
import logging
import math
import mimetypes
import os
import hashlib
import re
import shutil
import threading
import time
import traceback
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import requests

from spvideo.asset_store import (
    add_annotation,
    add_original_asset,
    assign_director_role_annotation,
    delete_character,
    delete_original_asset,
    get_director_plan,
    get_background_config,
    load_asset_store,
    update_annotation,
    update_annotation_mask,
    update_background_config,
    upsert_director_plan,
    upsert_background_config,
    upsert_character,
)
from spvideo.features import analyze_frame
from spvideo.ffmpeg_tools import concat_videos, extract_frame
from spvideo.pipeline import run_segmentation
from spvideo.comfy_inventory import fetch_inventory, load_inventory
from spvideo.comfy_client import COMFY_URL, ComfyClient
from spvideo.auto_director import (
    analyze_auto_director_project,
    answer_auto_director_question,
    load_auto_director,
    resolve_auto_director_project_root,
)


ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "splitter_dashboard.html"
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
SERVER_STARTED_AT = time.time()
RVM_MODEL_LOCK = threading.Lock()
SAM3_TRACK_LOCK = threading.Lock()
SEEDANCE_A_TASKS: dict[str, dict[str, Any]] = {}
SEEDANCE_A_TASKS_LOCK = threading.Lock()
SAM3_PROTECTION_FPS = 15.0
SAM3_PROTECTION_MAX_FRAMES = 180
SAM3_SHAPE_MIN_OVERLAP = 0.12
SAM3_SHAPE_OBJECT_TARGET_OVERLAP = 0.25
SAM3_SHAPE_OBJECT_MATCH_THRESHOLD = 0.35
SCAIL2_COLOR_NAMES = ("蓝色", "红色", "绿色", "紫色", "青色", "黄色")
WAN22_MASK_COLOR_KEYS = ("blue", "red", "green", "magenta", "cyan", "yellow")
DEFAULT_SEEDANCE_A_BASE_URL = "http://152.136.38.202:3000"
DEFAULT_SEEDANCE_A_UPLOAD_BASE_URL = "https://ai.szyqsc.cn"
DEFAULT_WAN22_API_KEY = ""
DEFAULT_WAN22_MULTI_ROLE_LIMIT = 0
DEFAULT_WAN22_ALLOW_EXPERIMENTAL_MULTI_FOCUS = False
DEFAULT_WAN22_FOCUS_FEATHER_PIXELS = 5
DEFAULT_WAN22_FOCUS_ERODE_PIXELS = 3
DEFAULT_WAN22_FOCUS_DILATE_PIXELS = 0
DEFAULT_WAN22_FOCUS_BACKGROUND_DIM = 0.18
DEFAULT_WAN22_FOCUS_BACKGROUND_BLUR_PIXELS = 0
DEFAULT_WAN22_ALLOW_LOCAL_MASK_FALLBACK = False
STORYBOARD_MODE2_PROJECT_ROOT = ROOT.parent / ".storyboard_mode2_projects"
STORYBOARD_MODE2_JOB_ROOT = ROOT.parent / ".storyboard_mode2_jobs"
DEFAULT_SINGLE_ROLE_TRANSFER_BACKEND = "wan22"
DEFAULT_MULTI_ROLE_TRANSFER_BACKEND = "scail2"
TRANSFER_BACKENDS = {"wan22", "scail2", "scail2_colored", "scail2_masked", "bernini", "runninghub_bernini"}
STORYBOARD_STYLE_PRESETS: dict[str, dict[str, str]] = {
    "short_drama_realism": {
        "label": "短剧写实",
        "prompt": "自然写实短剧风格，保留生活化光影和真实人物表演。",
    },
    "cinematic_realism": {
        "label": "电影写实",
        "prompt": "电影感写实风格，层次光影更强，镜头质感更克制。",
    },
    "urban_emotion": {
        "label": "都市情绪",
        "prompt": "现代都市情绪影像，人物干净利落，气氛更集中。",
    },
    "youth_film": {
        "label": "青春胶片",
        "prompt": "低饱和青春胶片风格，柔和自然光，轻微颗粒质感。",
    },
}
STORYBOARD_LANGUAGE_PRESETS: dict[str, dict[str, str]] = {
    "zh-CN": {"label": "中文", "prompt": "当前以中文语境整理分镜与资产命名。"},
    "es": {"label": "西语", "prompt": "后续字幕、配音和人物称谓按西语语境准备。"},
    "en": {"label": "英语", "prompt": "后续字幕、配音和人物称谓按英语语境准备。"},
    "pt-BR": {"label": "葡语（巴西）", "prompt": "后续字幕、配音和人物称谓按巴葡语境准备。"},
    "ja": {"label": "日语", "prompt": "后续字幕、配音和人物称谓按日语语境准备。"},
    "ko": {"label": "韩语", "prompt": "后续字幕、配音和人物称谓按韩语语境准备。"},
}
STORYBOARD_MOTION_PRESETS: dict[str, dict[str, str]] = {
    "strict": {"label": "动作锁定", "prompt": "尽量锁住原视频的人物调度、动作、机位和遮挡关系。"},
    "balanced": {"label": "剧情优先", "prompt": "保留剧情关系和关键动作，允许轻微重拍细节。"},
    "remake": {"label": "风格重拍", "prompt": "保留剧情节点和角色关系，允许较明显的镜头重构。"},
}
STORYBOARD_MASK_SOURCE_PRESETS: dict[str, dict[str, str]] = {
    "remote_sam3_color": {
        "label": "服务器 SAM3 彩色蒙版",
        "prompt": "优先使用 SCAIL-2/ComfyUI 服务器生成的 SAM3 彩色人物蒙版。",
    },
    "local_sam3": {
        "label": "本机 SAM3 蒙版",
        "prompt": "使用本机 SAM3 按手工身份点生成的人物 mask 轨迹。",
    },
}
STORYBOARD_REBUILD_GOALS: dict[str, dict[str, Any]] = {
    "english_europe_foreign_cast": {
        "label": "英语 / 欧洲 / 外国人重制",
        "prompt": [
            "【重制目标】把原视频重生成英文对白、欧洲风格背景、欧美外形角色版本。",
            "【角色重制】人物外形改成外国人风格，肤色、发色、五官、服装换成欧美观感，但动作、表情、站位、遮挡和镜头关系保持与原片一致。",
            "【背景重制】背景重绘为欧洲风格，建筑、街景、室内与色调统一为欧洲观感。",
            "【语言重制】对白改成英语，剧情、情绪和镜头节奏不变。",
        ],
        "defaults": {
            "style_preset": "cinematic_realism",
            "target_language": "en",
            "motion_constraint": "strict",
            "mask_source": "remote_sam3_color",
        },
    },
    "keep_source_locale": {
        "label": "保留原语境重制",
        "prompt": [
            "【重制目标】保留原语言和原人物语境，只重建画面、动作和镜头质感。",
        ],
        "defaults": {
            "style_preset": "short_drama_realism",
            "target_language": "zh-CN",
            "motion_constraint": "balanced",
            "mask_source": "remote_sam3_color",
        },
    },
}
STORYBOARD_REFERENCE_STRATEGIES: dict[str, dict[str, str]] = {
    "new_only": {
        "label": "只按新视频生成",
        "prompt": "忽略旧项目目录，只根据当前原视频重新反推故事、人物、资产和分镜草案。",
    },
    "old_reference": {
        "label": "旧项目作参考",
        "prompt": "读取旧项目里的故事、人物、片段描述和自动导演结果，作为当前新视频的软参考。",
    },
    "hybrid_reference": {
        "label": "旧项目 + 新视频融合",
        "prompt": "当前阶段仍按软参考处理：旧项目提供故事和角色提示，新视频决定实际切法。",
    },
}
STORYBOARD_RATIO_OPTIONS = ("9:16", "16:9", "1:1", "4:3", "3:4", "21:9")
DEFAULT_STORYBOARD_PROJECT_CONFIG = {
    "rebuild_goal": "english_europe_foreign_cast",
    "style_preset": "short_drama_realism",
    "target_language": "en",
    "output_ratio": "9:16",
    "motion_constraint": "strict",
    "mask_source": "remote_sam3_color",
}


def _latest_mode2_control_video(project_dir: str, video_path: str, output_kind: str = "white") -> Path | None:
    try:
        root = _resolve_storyboard_mode2_project_dir(project_dir)
    except Exception:
        return None
    output_dir = root / "04_AI输出成片"
    if not output_dir.exists():
        return None
    stem = Path(str(video_path or "")).stem
    if not stem:
        return None
    kind = str(output_kind or "white").strip().lower()
    if kind in {"background_gray", "bg_gray", "depth", "scene_depth"}:
        patterns = [f"scail2_vda_depth_{stem}_*.mp4"]
    elif kind in {"identity_gray_relief", "gray_relief", "light_gray_control"}:
        patterns = [f"identity_gray_relief_{stem}_*.mp4"]
    else:
        patterns = [f"scail2_vda_white_{stem}_*.mp4", f"scail2_vda_strong_clay_{stem}_*.mp4"]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(path for path in output_dir.glob(pattern) if path.is_file())
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.splitlines() if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled", "enable"}:
        return True
    if text in {"0", "false", "no", "off", "disabled", "disable"}:
        return False
    return default


def _normalize_identity_shape(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    raw_points = value.get("points")
    if not isinstance(raw_points, list):
        return None
    points: list[list[float]] = []
    for point in raw_points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            continue
        try:
            x = max(0.0, min(1.0, float(point[0])))
            y = max(0.0, min(1.0, float(point[1])))
        except (TypeError, ValueError):
            continue
        if points and abs(points[-1][0] - x) < 0.002 and abs(points[-1][1] - y) < 0.002:
            continue
        points.append([x, y])
    if len(points) < 3:
        return None
    return {"type": "freehand", "points": points}


def _identity_shape_center(shape: dict[str, Any] | None) -> list[float] | None:
    clean = _normalize_identity_shape(shape)
    if not clean:
        return None
    xs = [float(point[0]) for point in clean["points"]]
    ys = [float(point[1]) for point in clean["points"]]
    if not xs or not ys:
        return None
    return [
        max(0.0, min(1.0, (min(xs) + max(xs)) / 2.0)),
        max(0.0, min(1.0, (min(ys) + max(ys)) / 2.0)),
    ]


def _identity_shape_seed_points(
    shape: dict[str, Any] | None,
    fallback_point: list[float] | None = None,
) -> list[list[float]]:
    clean = _normalize_identity_shape(shape)
    fallback = None
    if isinstance(fallback_point, (list, tuple)) and len(fallback_point) == 2:
        try:
            fallback = [
                max(0.0, min(1.0, float(fallback_point[0]))),
                max(0.0, min(1.0, float(fallback_point[1]))),
            ]
        except (TypeError, ValueError):
            fallback = None
    if not clean:
        return [fallback] if fallback else []
    import cv2
    import numpy as np

    points = np.asarray([[float(item[0]), float(item[1])] for item in clean["points"]], dtype=np.float32)
    if points.shape[0] < 3:
        return [fallback] if fallback else []
    xs = points[:, 0]
    ys = points[:, 1]
    centroid = np.array([(float(xs.min()) + float(xs.max())) / 2.0, (float(ys.min()) + float(ys.max())) / 2.0], dtype=np.float32)
    poly = np.round(points * 1000.0).astype(np.int32).reshape((-1, 1, 2))

    def inside(point: list[float] | np.ndarray | None) -> bool:
        if point is None:
            return False
        x, y = float(point[0]), float(point[1])
        probe = (int(round(x * 1000.0)), int(round(y * 1000.0)))
        return cv2.pointPolygonTest(poly, probe, False) >= 0

    candidates: list[list[float]] = []
    for point in (fallback, centroid.tolist()):
        if point and inside(point):
            candidates.append([max(0.0, min(1.0, float(point[0]))), max(0.0, min(1.0, float(point[1])))] )

    # 从轮廓点向内收缩，避免把边缘和背景一起当成种子。
    for item in points[:: max(1, len(points) // 6)][:6]:
        inner = centroid + (item - centroid) * 0.58
        inner_point = [float(inner[0]), float(inner[1])]
        if inside(inner_point):
            candidates.append([
                max(0.0, min(1.0, inner_point[0])),
                max(0.0, min(1.0, inner_point[1])),
            ])

    # 如果还不够，再补几个更靠中心的点。
    for ratio in (0.45, 0.68):
        for item in points[:: max(1, len(points) // 5)][:5]:
            inner = centroid + (item - centroid) * ratio
            inner_point = [float(inner[0]), float(inner[1])]
            if inside(inner_point):
                candidates.append([
                    max(0.0, min(1.0, inner_point[0])),
                    max(0.0, min(1.0, inner_point[1])),
                ])

    unique: list[list[float]] = []
    for x, y in candidates:
        point = [max(0.0, min(1.0, float(x))), max(0.0, min(1.0, float(y)))]
        if any(math.hypot(point[0] - old[0], point[1] - old[1]) < 0.02 for old in unique):
            continue
        unique.append(point)
        if len(unique) >= 5:
            break
    return unique or ([fallback] if fallback else [])


def _identity_shape_mask_overlap(mask: Any, shape: dict[str, Any] | None) -> float | None:
    clean = _normalize_identity_shape(shape)
    if not clean:
        return None
    import cv2
    import numpy as np

    mask_array = np.asarray(mask)
    if mask_array.ndim > 2:
        mask_array = np.squeeze(mask_array)
    if mask_array.ndim != 2:
        return None
    height, width = mask_array.shape[:2]
    polygon = np.asarray(
        [
            [
                max(0, min(width - 1, int(round(float(x) * (width - 1))))),
                max(0, min(height - 1, int(round(float(y) * (height - 1))))),
            ]
            for x, y in clean["points"]
        ],
        dtype=np.int32,
    )
    if polygon.shape[0] < 3:
        return None
    seed = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(seed, [polygon], 1)
    seed_area = int(np.count_nonzero(seed))
    if seed_area <= 0:
        return None
    threshold = 127 if float(np.max(mask_array)) > 1.0 else 0
    selected = mask_array > threshold
    return float(np.count_nonzero(selected & (seed > 0))) / float(seed_area)


def _normalize_storyboard_reference_strategy(raw: Any) -> str:
    value = str(raw or "new_only").strip()
    if value not in STORYBOARD_REFERENCE_STRATEGIES:
        value = "new_only"
    return value


def _storyboard_reference_strategy_label(value: Any) -> str:
    key = _normalize_storyboard_reference_strategy(value)
    return STORYBOARD_REFERENCE_STRATEGIES[key]["label"]


def _normalize_storyboard_rebuild_goal(raw: Any) -> str:
    value = str(raw or DEFAULT_STORYBOARD_PROJECT_CONFIG["rebuild_goal"]).strip()
    if value not in STORYBOARD_REBUILD_GOALS:
        value = DEFAULT_STORYBOARD_PROJECT_CONFIG["rebuild_goal"]
    return value


def _storyboard_rebuild_goal_meta(value: Any) -> dict[str, Any]:
    key = _normalize_storyboard_rebuild_goal(value)
    return STORYBOARD_REBUILD_GOALS[key]


def _normalize_storyboard_project_config(raw: Any) -> dict[str, str]:
    data = raw if isinstance(raw, dict) else {}
    rebuild_goal = _normalize_storyboard_rebuild_goal(data.get("rebuild_goal"))
    goal_defaults = _storyboard_rebuild_goal_meta(rebuild_goal).get("defaults") or {}
    style = str(
        data.get("style_preset")
        or goal_defaults.get("style_preset")
        or DEFAULT_STORYBOARD_PROJECT_CONFIG["style_preset"]
    ).strip()
    language = str(
        data.get("target_language")
        or goal_defaults.get("target_language")
        or DEFAULT_STORYBOARD_PROJECT_CONFIG["target_language"]
    ).strip()
    ratio = str(data.get("output_ratio") or DEFAULT_STORYBOARD_PROJECT_CONFIG["output_ratio"]).strip()
    motion = str(
        data.get("motion_constraint")
        or goal_defaults.get("motion_constraint")
        or DEFAULT_STORYBOARD_PROJECT_CONFIG["motion_constraint"]
    ).strip()
    mask_source = str(
        data.get("mask_source")
        or goal_defaults.get("mask_source")
        or DEFAULT_STORYBOARD_PROJECT_CONFIG["mask_source"]
    ).strip()
    if rebuild_goal not in STORYBOARD_REBUILD_GOALS:
        rebuild_goal = DEFAULT_STORYBOARD_PROJECT_CONFIG["rebuild_goal"]
    if style not in STORYBOARD_STYLE_PRESETS:
        style = DEFAULT_STORYBOARD_PROJECT_CONFIG["style_preset"]
    if language not in STORYBOARD_LANGUAGE_PRESETS:
        language = DEFAULT_STORYBOARD_PROJECT_CONFIG["target_language"]
    if ratio not in STORYBOARD_RATIO_OPTIONS:
        ratio = DEFAULT_STORYBOARD_PROJECT_CONFIG["output_ratio"]
    if motion not in STORYBOARD_MOTION_PRESETS:
        motion = DEFAULT_STORYBOARD_PROJECT_CONFIG["motion_constraint"]
    if mask_source not in STORYBOARD_MASK_SOURCE_PRESETS:
        mask_source = DEFAULT_STORYBOARD_PROJECT_CONFIG["mask_source"]
    return {
        "rebuild_goal": rebuild_goal,
        "style_preset": style,
        "target_language": language,
        "output_ratio": ratio,
        "motion_constraint": motion,
        "mask_source": mask_source,
    }


def _storyboard_project_config_summary(config: dict[str, str]) -> str:
    rebuild_goal = STORYBOARD_REBUILD_GOALS[config["rebuild_goal"]]["label"]
    style = STORYBOARD_STYLE_PRESETS[config["style_preset"]]["label"]
    language = STORYBOARD_LANGUAGE_PRESETS[config["target_language"]]["label"]
    motion = STORYBOARD_MOTION_PRESETS[config["motion_constraint"]]["label"]
    mask_source = STORYBOARD_MASK_SOURCE_PRESETS[config["mask_source"]]["label"]
    return f"{rebuild_goal} / {style} / {language} / {config['output_ratio']} / {motion} / {mask_source}"


class JobLogHandler(logging.Handler):
    def __init__(self, job_id: str) -> None:
        super().__init__()
        self.job_id = job_id
        self.setFormatter(logging.Formatter("%(levelname).1s %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        with JOBS_LOCK:
            job = JOBS.get(self.job_id)
            if job is not None:
                job.setdefault("logs", []).append(message)


def _storyboard_job_snapshot_path(job_id: str) -> Path:
    safe_id = "".join(ch for ch in str(job_id or "") if ch.isalnum() or ch in ("-", "_"))
    return STORYBOARD_MODE2_JOB_ROOT / f"{safe_id}.json"


def _write_storyboard_job_snapshot(job: dict[str, Any]) -> None:
    job_id = str(job.get("id") or "").strip()
    if not job_id:
        return
    snapshot = dict(job)
    snapshot.pop("_cancel", None)
    path = _storyboard_job_snapshot_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _load_storyboard_job_snapshot(job_id: str) -> dict[str, Any]:
    path = _storyboard_job_snapshot_path(job_id)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(data, dict):
        return {}
    data.setdefault("id", job_id)
    data["restored_from_snapshot"] = True
    data["snapshot_path"] = str(path)
    if data.get("status") == "running":
        data["status"] = "failed"
        data["error"] = data.get("error") or "任务在后端重启后中断，请重新提交。"
        data.setdefault("logs", []).append("> 任务已中断：后端重启后后台线程不存在，请重新提交")
    return data


class SplitterHandler(BaseHTTPRequestHandler):
    server_version = "SPVideoWeb/0.1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_file(INDEX)
            return
        if parsed.path == "/media":
            query = parse_qs(parsed.query)
            target = unquote((query.get("path") or [""])[0])
            self._send_media(Path(target))
            return
        if parsed.path == "/api/server-status":
            self._send_json({
                "pid": os.getpid(),
                "started_at": SERVER_STARTED_AT,
                "server_file": str(Path(__file__).resolve()),
                "server_mtime": Path(__file__).stat().st_mtime,
                "mode2_project_root": str(STORYBOARD_MODE2_PROJECT_ROOT),
                "mode2_job_root": str(STORYBOARD_MODE2_JOB_ROOT),
            })
            return
        if parsed.path.startswith("/api/jobs/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = dict(JOBS.get(job_id) or {})
                if "logs" in job:
                    job["logs"] = list(job["logs"])
            if not job:
                job = _load_storyboard_job_snapshot(job_id)
            if not job:
                self._send_json({"error": "job_not_found"}, status=404)
                return
            self._send_json(job)
            return
        if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/cancel"):
            job_id = parsed.path.rsplit("/", 2)[-2]
            with JOBS_LOCK:
                job = JOBS.get(job_id, {})
                job["_cancel"] = True
                job["status"] = "cancelled"
                job.setdefault("logs", []).append("> 用户已取消")
            self._send_json({"cancelled": True})
            return
        if parsed.path == "/api/assets":
            query = parse_qs(parsed.query)
            project_dir = unquote((query.get("project_dir") or [""])[0])
            if not project_dir:
                self._send_json({"error": "missing_project_dir"}, status=400)
                return
            project_dir = str(resolve_auto_director_project_root(project_dir))
            self._send_json(load_asset_store(project_dir))
            return
        if parsed.path == "/api/comfy-inventory":
            try:
                inventory = load_inventory(COMFY_URL)
                if inventory is None:
                    inventory = fetch_inventory(COMFY_URL)
                self._send_json(inventory)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=502)
            return
        if parsed.path == "/api/storyboard-assets":
            try:
                query = parse_qs(parsed.query)
                project_dir = unquote((query.get("project_dir") or [""])[0])
                if not project_dir:
                    raise ValueError("missing_project_dir")
                root = _resolve_storyboard_mode2_project_dir(project_dir)
                self._send_json(_load_mode2_storyboard_result(root, _storyboard_mode2_asset_store_path(root)))
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return
        if parsed.path == "/api/storyboard-role-tracks":
            try:
                query = parse_qs(parsed.query)
                project_dir = unquote((query.get("project_dir") or [""])[0])
                if not project_dir:
                    raise ValueError("missing_project_dir")
                root = _resolve_storyboard_mode2_project_dir(project_dir)
                data = _load_storyboard_mode2_store(root)
                self._send_json({
                    "project_dir": str(root),
                    "role_tracks_version": int(data.get("role_tracks_version") or 1),
                    "identity_annotations": data.get("identity_annotations") or [],
                    "role_tracks": data.get("role_tracks") if isinstance(data.get("role_tracks"), dict) else {},
                    "role_track_history": data.get("role_track_history") if isinstance(data.get("role_track_history"), list) else [],
                })
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return
        if parsed.path == "/api/auto-director":
            query = parse_qs(parsed.query)
            project_dir = unquote((query.get("project_dir") or [""])[0])
            if not project_dir:
                self._send_json({"error": "missing_project_dir"}, status=400)
                return
            root = resolve_auto_director_project_root(project_dir)
            plan = load_auto_director(root)
            plan.setdefault("project_dir", str(root))
            self._send_json(plan)
            return

        file_path = (ROOT / unquote(parsed.path.lstrip("/"))).resolve()
        if ROOT in file_path.parents or file_path == ROOT:
            self._send_file(file_path)
            return
        self._send_json({"error": "not_found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        print(f"[web] POST {parsed.path!r}", flush=True)
        if parsed.path.startswith("/api/jobs/") and parsed.path.endswith("/cancel"):
            job_id = parsed.path.rsplit("/", 2)[-2]
            with JOBS_LOCK:
                job = JOBS.get(job_id, {})
                job["_cancel"] = True
                job["status"] = "cancelled"
                job.setdefault("logs", []).append("> 用户已取消")
            self._send_json({"cancelled": True, "job_id": job_id})
            return

        if parsed.path.rstrip("/") == "/api/assets/delete-character":
            try:
                payload = self._read_json()
                project_dir = payload.get("project_dir", "")
                char_id = str(payload.get("id") or "")
                from spvideo import asset_store
                ok = asset_store.delete_character(project_dir, char_id)
                self._send_json({"deleted": ok, "assets": load_asset_store(project_dir)})
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path.rstrip("/") == "/api/assets/delete-original":
            try:
                payload = self._read_json()
                project_dir = payload.get("project_dir", "")
                deleted = delete_original_asset(project_dir, str(payload.get("id") or ""))
                self._send_json({"deleted": deleted, "assets": load_asset_store(project_dir)})
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/open-path":
            try:
                payload = self._read_json()
                target = str(payload.get("path") or "").strip()
                if not target:
                    raise ValueError("缺少路径")
                path = Path(target)
                if not path.exists():
                    raise ValueError(f"路径不存在: {target}")
                os.startfile(str(path))  # type: ignore[attr-defined]
                self._send_json({"ok": True})
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/assistant-chat":
            try:
                payload = self._read_json()
                self._send_json(_assistant_chat(payload))
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=400)
            return

        if parsed.path == "/api/pick-path":
            try:
                payload = self._read_json()
                kind = str(payload.get("kind") or "file")
                initial = str(payload.get("initial") or "").strip()
                picked = _pick_path(kind=kind, initial=initial)
                self._send_json({"path": picked})
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/seedance-a-submit":
            try:
                payload = self._read_json()
                result = _submit_seedance_a_task(payload)
                self._send_json(result)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/seedance-a-status":
            try:
                payload = self._read_json()
                result = _query_seedance_a_task(payload)
                self._send_json(result)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/storyboard-draft":
            try:
                payload = self._read_json()
                job_id = uuid.uuid4().hex
                job = {
                    "id": job_id,
                    "type": "storyboard_draft",
                    "status": "running",
                    "created_at": time.time(),
                    "logs": ["> 已提交分镜草案任务", f"> job_id: {job_id}"],
                    "result": None,
                    "error": None,
                }
                with JOBS_LOCK:
                    JOBS[job_id] = job
                _write_storyboard_job_snapshot(job)
                thread = threading.Thread(target=_run_storyboard_draft_job, args=(job_id, payload), daemon=True)
                thread.start()
                self._send_json({"job_id": job_id})
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/storyboard-assets/refine":
            try:
                payload = self._read_json()
                project_dir = str(payload.get("project_dir") or "").strip()
                if not project_dir:
                    raise ValueError("missing_project_dir")
                root = _resolve_storyboard_mode2_project_dir(project_dir)
                payload["project_dir"] = str(root)
                job_id = uuid.uuid4().hex
                job = {
                    "id": job_id,
                    "type": "storyboard_asset_refine",
                    "status": "running",
                    "created_at": time.time(),
                    "logs": ["> Mode2 资产提纯任务已提交", f"> job_id: {job_id}", f"> project_dir: {root}"],
                    "result": None,
                    "error": None,
                }
                with JOBS_LOCK:
                    JOBS[job_id] = job
                _write_storyboard_job_snapshot(job)
                thread = threading.Thread(target=_run_storyboard_asset_refine_job, args=(job_id, payload), daemon=True)
                thread.start()
                self._send_json({"job_id": job_id})
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/storyboard-assets/split-scene":
            try:
                payload = self._read_json()
                self._send_json(_split_storyboard_mode2_scenes(payload))
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/storyboard-assets/edit-shot":
            try:
                payload = self._read_json()
                self._send_json(_edit_storyboard_mode2_shot(payload))
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/storyboard-assets/update-asset":
            try:
                payload = self._read_json()
                result = _update_storyboard_mode2_asset(payload)
                self._send_json(result)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/storyboard-assets/create-asset":
            try:
                payload = self._read_json()
                result = _create_storyboard_mode2_asset(payload)
                self._send_json(result)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/storyboard-assets/organize":
            try:
                payload = self._read_json()
                result = _organize_storyboard_mode2_asset_library(payload)
                self._send_json(result)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/storyboard-assets/audit":
            try:
                payload = self._read_json()
                result = _audit_storyboard_mode2_assets(payload)
                self._send_json(result)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/storyboard-role-anchors":
            try:
                payload = self._read_json()
                result = _update_storyboard_mode2_role_anchor(payload)
                self._send_json(result)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/storyboard-role-mask-candidates":
            try:
                payload = self._read_json()
                result = _storyboard_mode2_role_mask_candidates(payload)
                self._send_json(result)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/storyboard-role-tracks/run":
            try:
                payload = self._read_json()
                project_dir = str(payload.get("project_dir") or "").strip()
                if not project_dir:
                    raise ValueError("missing_project_dir")
                root = _resolve_storyboard_mode2_project_dir(project_dir)
                payload["project_dir"] = str(root)
                job_id = uuid.uuid4().hex
                job = {
                    "id": job_id,
                    "type": "storyboard_role_track",
                    "status": "running",
                    "created_at": time.time(),
                    "logs": ["> Mode2 SAM3 身份分轨任务已提交", f"> job_id: {job_id}", f"> project_dir: {root}"],
                    "result": None,
                    "error": None,
                }
                with JOBS_LOCK:
                    JOBS[job_id] = job
                _write_storyboard_job_snapshot(job)
                thread = threading.Thread(target=_run_storyboard_role_track_job, args=(job_id, payload), daemon=True)
                thread.start()
                self._send_json({"job_id": job_id})
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/storyboard-role-track-preview":
            try:
                payload = self._read_json()
                result = _storyboard_mode2_role_track_preview(payload)
                self._send_json(result)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/storyboard-reference-mask":
            try:
                payload = self._read_json()
                video_path = str(payload.get("video_path") or "").strip()
                if not video_path or not Path(video_path).exists():
                    raise ValueError("reference_video_not_found")
                raw_ref_images = payload.get("ref_images") or payload.get("images") or []
                ref_images = _string_list(raw_ref_images)
                if not ref_images:
                    raise ValueError("reference_images_required_for_colored_mask")
                if len(ref_images) > 6:
                    raise ValueError("colored_mask_supports_at_most_6_reference_images")
                role_names = _string_list(payload.get("role_names") or payload.get("roles"))
                role_pairs = []
                for index, ref_image in enumerate(ref_images):
                    name = role_names[index] if index < len(role_names) and role_names[index] else f"参考图{index + 1}"
                    role_pairs.append({"name": name, "ref_image": ref_image})
                segment_id = str(payload.get("segment_id") or "").strip()
                pair_warnings: list[str] = []
                role_pairs, pair_warnings = _mode2_reference_mask_role_pairs_from_store(
                    str(payload.get("project_dir") or "").strip(),
                    segment_id,
                    role_pairs,
                    prefer_current_shot_roles=bool(payload.get("prefer_current_shot_roles", True)),
                )
                source_identity_points = payload.get("source_identity_points") or []
                if isinstance(source_identity_points, list):
                    for index, point in enumerate(source_identity_points):
                        if index >= len(role_pairs):
                            break
                        if isinstance(point, (list, tuple)) and len(point) == 2:
                            try:
                                role_pairs[index]["source_point"] = [
                                    max(0.0, min(1.0, float(point[0]))),
                                    max(0.0, min(1.0, float(point[1]))),
                                ]
                            except (TypeError, ValueError):
                                pass
                source_identity_shapes = payload.get("source_identity_shapes") or []
                if isinstance(source_identity_shapes, list):
                    for index, shape in enumerate(source_identity_shapes):
                        if index >= len(role_pairs):
                            break
                        clean_shape = _normalize_identity_shape(shape)
                        if clean_shape:
                            role_pairs[index]["source_shape"] = clean_shape
                sampler_preset = str(payload.get("sampler_preset") or "balanced").strip() or "balanced"
                video_window = payload.get("video_window") or {}
                normalize_size = bool(payload.get("normalize_size", True))
                sam_text = str(payload.get("sam_text") or "").strip()
                require_single_shot = bool(payload.get("require_single_shot", True))
                strict_track_preflight = bool(payload.get("strict_track_preflight", True))
                public_mapping = [
                    {
                        "order": index + 1,
                        "color": SCAIL2_COLOR_NAMES[index],
                        "role": pair["name"],
                        "target_ref": Path(pair["ref_image"]).name,
                    }
                    for index, pair in enumerate(role_pairs)
                ]
                job_id = uuid.uuid4().hex[:8]
                with JOBS_LOCK:
                    JOBS[job_id] = {
                        "id": job_id,
                        "type": "mode2_reference_colored_mask",
                        "status": "running",
                        "created_at": time.time(),
                        "video_path": video_path,
                        "segment_id": segment_id,
                        "mapping": public_mapping,
                        "logs": [
                            f"> Mode2 参考视频彩色蒙版: {Path(video_path).name}",
                            *[
                                f"  {SCAIL2_COLOR_NAMES[i]}: {pair['name']} → {Path(pair['ref_image']).name}"
                                for i, pair in enumerate(role_pairs)
                            ],
                            *[f"> 参考图选择提醒: {warning}" for warning in pair_warnings],
                            *(["> 单镜头校验: 开启，检测到硬切会阻止整段跑蒙版"] if require_single_shot else []),
                        ],
                        "result": None,
                        "error": None,
                    }
                    _write_storyboard_job_snapshot(JOBS[job_id])
                thread = threading.Thread(
                    target=_run_scail2_mask_check_job,
                    args=(
                        job_id,
                        video_path,
                        role_pairs,
                        sampler_preset,
                        video_window,
                        normalize_size,
                        sam_text,
                        strict_track_preflight,
                        require_single_shot,
                    ),
                    daemon=True,
                )
                thread.start()
                self._send_json({"job_id": job_id, "mapping": public_mapping})
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/storyboard-reference-white-mask":
            try:
                payload = self._read_json()
                video_path = str(payload.get("video_path") or "").strip()
                if not video_path or not Path(video_path).exists():
                    raise ValueError("reference_video_not_found")
                project_dir = str(payload.get("project_dir") or "").strip()
                if project_dir:
                    try:
                        project_dir = str(_resolve_storyboard_mode2_project_dir(project_dir))
                    except Exception:
                        project_dir = ""
                video_window = payload.get("video_window") or {}
                normalize_size = bool(payload.get("normalize_size", True))
                model = str(payload.get("model") or "video_depth_anything_vitb.pth").strip()
                input_size = int(payload.get("input_size") or 518)
                max_res = int(payload.get("max_res") or 1280)
                precision = str(payload.get("precision") or "fp16").strip() or "fp16"
                output_kind = str(payload.get("output_kind") or "white").strip().lower() or "white"
                is_background_gray = output_kind in {"background_gray", "bg_gray", "depth", "scene_depth"}
                is_normal_lighting = output_kind in {"normal_lighting", "normal_lit", "light_clay", "strong_clay"}
                if is_normal_lighting:
                    raise ValueError("normal_lighting_white_mask_disabled")
                is_identity_gray_relief = output_kind in {"identity_gray_relief", "gray_relief", "light_gray_control"}
                cached_path = _latest_mode2_control_video(project_dir, video_path, output_kind)
                if cached_path and output_kind == "white":
                    job_id = uuid.uuid4().hex[:8]
                    result = {
                        "workflow_mode": "remote_vda_white_mask_cached",
                        "video_path": video_path,
                        "white_mask_path": str(cached_path),
                        "output_path": str(cached_path),
                        "mask_output_paths": {"white": [str(cached_path)]},
                        "cached": True,
                    }
                    job = {
                        "id": job_id,
                        "type": "mode2_reference_white_mask",
                        "status": "done",
                        "created_at": time.time(),
                        "finished_at": time.time(),
                        "project_dir": project_dir,
                        "video_path": video_path,
                        "logs": [
                            f"> 复用已有 VDA 白膜: {cached_path}",
                            "> 如需强制重跑，请先删除旧白膜文件或换一个输出片段。",
                        ],
                        "result": result,
                        "error": None,
                    }
                    with JOBS_LOCK:
                        JOBS[job_id] = job
                    _write_storyboard_job_snapshot(job)
                    self._send_json({"job_id": job_id, "cached": True, "output_path": str(cached_path)})
                    return
                job_id = uuid.uuid4().hex[:8]
                job = {
                    "id": job_id,
                    "type": (
                        "mode2_reference_background_gray"
                        if is_background_gray
                        else (
                            "mode2_reference_normal_lighting"
                            if is_normal_lighting
                            else (
                        "mode2_reference_identity_gray_relief"
                                if is_identity_gray_relief
                                else "mode2_reference_white_mask"
                            )
                        )
                    ),
                    "status": "running",
                    "created_at": time.time(),
                    "project_dir": project_dir,
                    "video_path": video_path,
                    "logs": [
                        f"> Mode2 {'去身份运动参考' if is_identity_gray_relief else ('VDA ' + ('背景灰模' if is_background_gray else ('强2.5D白模' if is_normal_lighting else '白膜')))}: {Path(video_path).name}",
                        (
                            "> method: local grayscale relief, no SAM/DWPose/generative model"
                            if is_identity_gray_relief
                            else f"> model: {model or 'video_depth_anything_vitb.pth'}"
                        ),
                    ],
                    "result": None,
                    "error": None,
                }
                with JOBS_LOCK:
                    JOBS[job_id] = job
                _write_storyboard_job_snapshot(job)
                thread = threading.Thread(
                    target=_run_scail2_white_mask_job,
                    args=(
                        job_id,
                        video_path,
                        model,
                        video_window,
                        normalize_size,
                        input_size,
                        max_res,
                        precision,
                        output_kind,
                    ),
                    daemon=True,
                )
                thread.start()
                self._send_json({"job_id": job_id})
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/storyboard-reference-expression-mask":
            try:
                payload = self._read_json()
                video_path = str(payload.get("video_path") or "").strip()
                if not video_path or not Path(video_path).exists():
                    raise ValueError("reference_video_not_found")
                project_dir = str(payload.get("project_dir") or "").strip()
                if project_dir:
                    try:
                        project_dir = str(_resolve_storyboard_mode2_project_dir(project_dir))
                    except Exception:
                        project_dir = ""
                video_window = payload.get("video_window") or {}
                normalize_size = bool(payload.get("normalize_size", True))
                model = str(payload.get("model") or "video_depth_anything_vitb.pth").strip()
                input_size = int(payload.get("input_size") or 518)
                max_res = int(payload.get("max_res") or 1280)
                precision = str(payload.get("precision") or "fp16").strip() or "fp16"
                output_mode = str(payload.get("output_mode") or "").strip().lower()
                is_capsule_control = output_mode == "capsule_control"
                max_faces = int(payload.get("max_faces") or 6)
                include_mouth = _bool_value(payload.get("include_mouth"), True)
                safe_mode = _bool_value(payload.get("safe_mode"), True)
                color_faces = False if safe_mode else _bool_value(payload.get("color_faces"), True)
                include_eyes = _bool_value(payload.get("include_eyes"), True)
                include_brows = _bool_value(payload.get("include_brows"), True)
                include_head_pose = _bool_value(payload.get("include_head_pose"), True)
                include_face_outline = _bool_value(payload.get("include_face_outline"), True)
                include_soft_face_relief = _bool_value(
                    payload.get("include_soft_face_relief", payload.get("face_relief")),
                    False,
                )
                normal_lighting_pose_enhance = _bool_value(payload.get("normal_lighting_pose_enhance"), False)
                strong_depth_relief = _bool_value(payload.get("strong_depth_relief"), False)
                include_body_pose = _bool_value(payload.get("include_body_pose"), False)
                pose_backend = str(payload.get("pose_backend") or "server_dwpose").strip().lower() or "server_dwpose"
                pose_render_style = str(
                    payload.get("pose_render_style", payload.get("control_render_mode")) or "capsule"
                ).strip().lower() or "capsule"
                pose_capsule_strength = str(payload.get("pose_capsule_strength") or "strong").strip().lower() or "strong"
                body_color_mode = "none"
                body_colors_enabled = False
                pose_only_light_clay = (
                    not is_capsule_control
                    and
                    include_body_pose
                    and str(pose_render_style).strip().lower() == "capsule"
                    and not include_mouth
                    and not include_eyes
                    and not include_brows
                    and not include_head_pose
                    and not include_face_outline
                    and not include_soft_face_relief
                    and not color_faces
                )
                strong_depth_relief = bool(
                    (strong_depth_relief and not is_capsule_control)
                    or output_mode == "light_clay"
                    or normal_lighting_pose_enhance
                    or pose_only_light_clay
                )
                is_light_clay = output_mode == "light_clay" or strong_depth_relief
                if is_light_clay:
                    raise ValueError("light_clay_expression_mask_disabled")
                job_id = uuid.uuid4().hex[:8]
                job_type = (
                    "mode2_reference_capsule_control"
                    if is_capsule_control
                    else ("mode2_reference_light_clay" if is_light_clay else "mode2_reference_expression_mask")
                )
                job = {
                    "id": job_id,
                    "type": job_type,
                    "status": "running",
                    "created_at": time.time(),
                    "project_dir": project_dir,
                    "video_path": video_path,
                    "output_mode": "capsule_control" if is_capsule_control else ("light_clay" if is_light_clay else (output_mode or "expression")),
                    "strong_depth_relief": bool(strong_depth_relief),
                    "logs": [
                        f"> Mode2 {'强2.5D白模' if is_light_clay else 'VDA 表情线稿'}: {Path(video_path).name}",
                        f"> output mode: {output_mode or ('light_clay' if is_light_clay else 'expression')}",
                        f"> model: {model or 'video_depth_anything_vitb.pth'}",
                        f"> safe mode: {'enabled' if safe_mode else 'disabled'}",
                        f"> mouth control: {'enabled' if include_mouth else 'disabled'}",
                        f"> face colors: {'disabled' if safe_mode else ('enabled' if color_faces else 'disabled')}",
                        f"> face outline: {'enabled' if include_face_outline else 'disabled'}",
                        f"> soft face relief: {'enabled' if include_soft_face_relief else 'disabled'}",
                        f"> strong depth relief: {'enabled' if strong_depth_relief else 'disabled'}",
                        f"> normal lighting pose enhance: {'enabled' if normal_lighting_pose_enhance else 'disabled'}",
                        f"> head pose: {'enabled' if include_head_pose else 'disabled'}",
                        f"> eyes/brows: {'enabled' if include_eyes or include_brows else 'disabled'}",
                        f"> body pose: {'enabled' if include_body_pose else 'disabled'} ({pose_backend})",
                        f"> pose render style: {pose_render_style}",
                        f"> pose capsule strength: {pose_capsule_strength}",
                        f"> body colors: {'enabled' if body_colors_enabled else 'disabled'}",
                    ],
                    "result": None,
                    "error": None,
                }
                with JOBS_LOCK:
                    JOBS[job_id] = job
                _write_storyboard_job_snapshot(job)
                if is_capsule_control:
                    thread = threading.Thread(
                        target=_run_scail2_capsule_control_job,
                        args=(job_id, video_path, video_window, normalize_size, pose_backend, pose_capsule_strength),
                        daemon=True,
                    )
                else:
                    thread = threading.Thread(
                        target=_run_scail2_expression_mask_job,
                        args=(
                            job_id,
                            video_path,
                            model,
                            video_window,
                            normalize_size,
                            input_size,
                            max_res,
                            precision,
                            max_faces,
                            include_mouth,
                            color_faces,
                            include_eyes,
                            include_brows,
                            include_head_pose,
                            include_face_outline,
                            include_soft_face_relief,
                            strong_depth_relief,
                            safe_mode,
                            body_color_mode,
                            include_body_pose,
                            pose_backend,
                            pose_render_style,
                            pose_capsule_strength,
                        ),
                        daemon=True,
                    )
                thread.start()
                self._send_json({"job_id": job_id})
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/storyboard-reference-mask/merge":
            try:
                payload = self._read_json()
                root = _resolve_storyboard_mode2_project_dir(payload.get("project_dir") or "")
                raw_mask_paths = payload.get("mask_paths") or payload.get("input_mask_paths") or []
                mask_paths = _string_list(raw_mask_paths)
                if not mask_paths:
                    raise ValueError("mask_paths_required")
                for mask_path in mask_paths:
                    path = Path(mask_path)
                    if not path.exists() or not path.is_file():
                        raise ValueError(f"mask_path_not_found: {mask_path}")
                    if path.suffix.lower() != ".mp4":
                        raise ValueError(f"mask_path_must_be_mp4: {mask_path}")
                source_video_path = str(payload.get("source_video_path") or payload.get("video_path") or "").strip()
                if source_video_path and not Path(source_video_path).exists():
                    raise ValueError(f"source_video_not_found: {source_video_path}")
                segment_id = str(payload.get("segment_id") or "").strip()
                subclips = payload.get("subclips") if isinstance(payload.get("subclips"), list) else []
                job_id = uuid.uuid4().hex[:8]
                job = {
                    "id": job_id,
                    "type": "mode2_reference_colored_mask_merge",
                    "status": "running",
                    "created_at": time.time(),
                    "project_dir": str(root),
                    "source_video_path": source_video_path,
                    "segment_id": segment_id,
                    "input_mask_paths": mask_paths,
                    "logs": [
                        f"> Mode2 reference colored mask merge: {len(mask_paths)} segments",
                        f"> project_dir: {root}",
                    ],
                    "result": None,
                    "error": None,
                }
                with JOBS_LOCK:
                    JOBS[job_id] = job
                _write_storyboard_job_snapshot(job)
                thread = threading.Thread(
                    target=_run_storyboard_reference_mask_merge_job,
                    args=(job_id, payload),
                    daemon=True,
                )
                thread.start()
                self._send_json({"job_id": job_id})
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/assets/characters":
            try:
                payload = self._read_json()
                item = upsert_character(
                    payload.get("project_dir", ""),
                    name=str(payload.get("name") or "").strip(),
                    label=payload.get("label"),
                    ref_image=str(payload.get("ref_image") or "").strip(),
                    extra_ref_images=(
                        _string_list(payload.get("extra_ref_images"))
                        if "extra_ref_images" in payload
                        else None
                    ),
                    character_id=payload.get("id"),
                )
                self._send_json({"character": item})
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/assets/delete-original":
            try:
                payload = self._read_json()
                project_dir = payload.get("project_dir", "")
                deleted = delete_original_asset(project_dir, str(payload.get("id") or ""))
                self._send_json({"deleted": deleted, "assets": load_asset_store(project_dir)})
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/auto-matting":
            try:
                payload = self._read_json()
                result = _rvm_auto_matting(payload)
                self._send_json(result)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/protection-point":
            try:
                payload = self._read_json()
                project_dir = str(payload.get("project_dir") or "").strip()
                video_path = str(payload.get("video_path") or "").strip()
                point = payload.get("point")
                if not project_dir:
                    raise ValueError("missing_project_dir")
                if not video_path or not Path(video_path).exists():
                    raise ValueError("video_not_found")
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError("protection_point_required")
                normalized_point = [float(point[0]), float(point[1])]
                if not all(0.0 <= value <= 1.0 for value in normalized_point):
                    raise ValueError("protection_point_out_of_bounds")
                annotation = add_annotation(
                    project_dir,
                    video_path=video_path,
                    time_seconds=max(0.0, float(payload.get("time") or 0)),
                    label_id=-1,
                    label_name="保护区域",
                    kind="protection",
                    point=normalized_point,
                    segment_id=str(payload.get("segment_id") or ""),
                )
                annotation = update_annotation(
                    project_dir,
                    str(annotation["id"]),
                    track_status="selected",
                    protection=True,
                ) or annotation
                self._send_json({
                    "annotation": annotation,
                    "assets": load_asset_store(project_dir),
                })
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/assets/sam3-track":
            try:
                payload = self._read_json()
                job_id = uuid.uuid4().hex
                with JOBS_LOCK:
                    JOBS[job_id] = {
                        "id": job_id,
                        "type": "sam3_track",
                        "status": "running",
                        "created_at": time.time(),
                        "logs": ["> SAM3 track job queued"],
                        "result": None,
                        "error": None,
                    }
                threading.Thread(target=_run_sam3_track_job, args=(job_id, payload), daemon=True).start()
                self._send_json({"job_id": job_id})
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/comfy-inventory/refresh":
            try:
                inventory = fetch_inventory(COMFY_URL)
                self._send_json(inventory)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=502)
            return

        if parsed.path == "/api/scail2-mask-check":
            try:
                payload = self._read_json()
                video_path = str(payload.get("video_path") or "").strip()
                project_dir = str(payload.get("project_dir") or "").strip()
                seg_id = str(payload.get("segment_id") or "").strip()
                sampler_preset = str(payload.get("sampler_preset") or "balanced")
                video_window = payload.get("video_window") or {}
                normalize_size = bool(payload.get("normalize_size", True))
                if not video_path or not Path(video_path).exists():
                    self._send_json({"error": "video_not_found"}, status=400)
                    return

                store = load_asset_store(project_dir) if project_dir else {}
                characters = store.get("characters", {})
                originals = store.get("originals", [])
                annotations = store.get("annotations", [])
                director_plan = get_director_plan(
                    project_dir,
                    segment_id=seg_id,
                    video_path=video_path,
                ) if project_dir else None
                char_name = str(payload.get("character") or "").strip()
                raw_char_names = payload.get("characters") or []
                if isinstance(raw_char_names, str):
                    char_names = [name.strip() for name in raw_char_names.split(",") if name.strip()]
                else:
                    char_names = [str(name).strip() for name in raw_char_names if str(name).strip()]

                if director_plan and director_plan.get("roles"):
                    if director_plan.get("status") != "ready":
                        issues = "；".join(director_plan.get("issues") or ["配置不完整"])
                        self._send_json({"error": "片段导演计划未完成: " + issues}, status=400)
                        return
                    char_names = [
                        str(role.get("name") or "").strip()
                        for role in director_plan.get("roles", [])
                        if str(role.get("name") or "").strip()
                    ]
                    char_name = ", ".join(char_names)

                if not char_name:
                    for ann in annotations:
                        if ann.get("type") == "person" and _same_video_path(ann.get("video_path"), video_path):
                            ln = str(ann.get("label_name") or "").strip()
                            if ln:
                                char_name = ln
                                char_names = [ln]
                                break
                    if not char_name:
                        for orig in originals:
                            ln = str(orig.get("label_name") or "").strip()
                            if ln and any(ch.get("name", "").strip() == ln for ch in characters.values()):
                                char_name = ln
                                char_names = [ln]
                                break
                if not char_names and char_name:
                    char_names = [char_name]

                role_pairs, mapping_warning = _resolve_scail2_role_pairs(
                    char_names=char_names,
                    characters=characters,
                    originals=originals,
                    annotations=annotations,
                    video_path=video_path,
                    director_plan=director_plan,
                )
                public_mapping = [
                    {
                        "order": index + 1,
                        "color": SCAIL2_COLOR_NAMES[index],
                        "role": pair["name"],
                        "target_ref": Path(pair["ref_image"]).name,
                        "source_x": pair.get("source_x"),
                        "mark_time": pair.get("source_time"),
                    }
                    for index, pair in enumerate(role_pairs)
                ]

                job_id = uuid.uuid4().hex[:8]
                with JOBS_LOCK:
                    JOBS[job_id] = {
                        "id": job_id,
                        "type": "scail2_mask_check",
                        "status": "running",
                        "video_path": video_path,
                        "mapping": public_mapping,
                        "mapping_warning": mapping_warning,
                        "director_plan_id": str((director_plan or {}).get("id") or ""),
                        "logs": [
                            f"> 远程 SAM3 蒙版检查: {Path(video_path).name}",
                            *[
                                f"  {SCAIL2_COLOR_NAMES[i]}: {pair['name']} → {Path(pair['ref_image']).name}"
                                for i, pair in enumerate(role_pairs)
                            ],
                            *([f"> 映射提醒: {mapping_warning}"] if mapping_warning else []),
                        ],
                        "result": None,
                        "error": None,
                    }

                thread = threading.Thread(
                    target=_run_scail2_mask_check_job,
                    args=(
                        job_id,
                        video_path,
                        role_pairs,
                        sampler_preset,
                        video_window,
                        normalize_size,
                        str((director_plan or {}).get("sam_text") or ""),
                    ),
                    daemon=True,
                )
                thread.start()
                self._send_json({
                    "job_id": job_id,
                    "mapping": public_mapping,
                    "mapping_warning": mapping_warning,
                })
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/director-plan":
            try:
                payload = self._read_json()
                project_dir = str(payload.get("project_dir") or "").strip()
                video_path = str(payload.get("video_path") or "").strip()
                roles = payload.get("roles") or []
                if not project_dir:
                    raise ValueError("missing_project_dir")
                if not video_path or not Path(video_path).exists():
                    raise ValueError("video_not_found")
                if not isinstance(roles, list):
                    raise ValueError("director_roles_invalid")
                plan = upsert_director_plan(
                    project_dir,
                    segment_id=str(payload.get("segment_id") or "").strip(),
                    video_path=video_path,
                    roles=roles,
                    positive_prompt=str(payload.get("positive_prompt") or ""),
                    sam_text=str(payload.get("sam_text") or ""),
                )
                _sync_director_roles_to_characters(project_dir, plan)
                self._send_json({
                    "director": plan,
                    "assets": load_asset_store(project_dir),
                })
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/director-role-point":
            try:
                payload = self._read_json()
                project_dir = str(payload.get("project_dir") or "").strip()
                video_path = str(payload.get("video_path") or "").strip()
                segment_id = str(payload.get("segment_id") or "").strip()
                role_name = str(payload.get("role_name") or "").strip()
                point = payload.get("point")
                if not project_dir:
                    raise ValueError("missing_project_dir")
                if not video_path or not Path(video_path).exists():
                    raise ValueError("video_not_found")
                if not role_name:
                    raise ValueError("director_role_required")
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError("director_point_required")
                normalized_point = [float(point[0]), float(point[1])]
                if not all(0.0 <= value <= 1.0 for value in normalized_point):
                    raise ValueError("director_point_out_of_bounds")

                plan = get_director_plan(
                    project_dir,
                    segment_id=segment_id,
                    video_path=video_path,
                )
                if not plan:
                    raise ValueError("director_plan_not_found")
                role = next(
                    (item for item in plan.get("roles", []) if item.get("name") == role_name),
                    None,
                )
                if role is None:
                    raise ValueError("director_role_not_found")

                annotation = add_annotation(
                    project_dir,
                    video_path=video_path,
                    time_seconds=max(0.0, float(payload.get("time") or 0)),
                    label_id=int(role.get("order") or 0),
                    label_name=role_name,
                    kind="person",
                    point=normalized_point,
                    segment_id=segment_id,
                )
                plan = assign_director_role_annotation(
                    project_dir,
                    segment_id=segment_id,
                    video_path=video_path,
                    role_name=role_name,
                    annotation_id=str(annotation["id"]),
                )
                _sync_director_roles_to_characters(project_dir, plan)
                self._send_json({
                    "annotation": annotation,
                    "director": plan,
                    "assets": load_asset_store(project_dir),
                })
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/auto-director/analyze":
            try:
                payload = self._read_json()
                project_dir = str(payload.get("project_dir") or "").strip()
                if not project_dir:
                    raise ValueError("missing_project_dir")
                project_dir = str(resolve_auto_director_project_root(project_dir))
                payload["project_dir"] = project_dir
                job_id = uuid.uuid4().hex[:10]
                with JOBS_LOCK:
                    JOBS[job_id] = {
                        "id": job_id,
                        "type": "auto_director",
                        "status": "running",
                        "created_at": time.time(),
                        "logs": ["> 自动导演任务已开始"],
                        "result": None,
                        "error": None,
                    }
                threading.Thread(
                    target=_run_auto_director_job,
                    args=(job_id, payload),
                    daemon=True,
                ).start()
                self._send_json({"job_id": job_id})
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/auto-director/answer":
            try:
                payload = self._read_json()
                project_dir = str(payload.get("project_dir") or "").strip()
                question_id = str(payload.get("question_id") or "").strip()
                if not project_dir:
                    raise ValueError("missing_project_dir")
                if not question_id:
                    raise ValueError("auto_director_question_required")
                project_dir = str(resolve_auto_director_project_root(project_dir))
                plan = answer_auto_director_question(
                    project_dir,
                    question_id=question_id,
                    answer=payload.get("answer"),
                )
                synced_roles = _sync_auto_director_roles_to_assets(project_dir, plan)
                self._send_json({
                    "auto_director": plan,
                    "assets": load_asset_store(project_dir),
                    "synced_roles": synced_roles,
                })
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/background-config":
            try:
                payload = self._read_json()
                project_dir = str(payload.get("project_dir") or "").strip()
                video_path = str(payload.get("video_path") or "").strip()
                if not project_dir:
                    raise ValueError("missing_project_dir")
                if not video_path:
                    raise ValueError("missing_video_path")
                item = upsert_background_config(
                    project_dir,
                    segment_id=str(payload.get("segment_id") or "").strip(),
                    video_path=video_path,
                    mode=str(payload.get("mode") or "keep_original"),
                    asset_path=str(payload.get("asset_path") or "").strip(),
                    fit_mode=str(payload.get("fit_mode") or "cover"),
                    feather_pixels=int(payload.get("feather_pixels") or 9),
                    dilate_pixels=int(payload.get("dilate_pixels") or 6),
                )
                self._send_json({"background": item, "assets": load_asset_store(project_dir)})
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/background-recommendations":
            try:
                payload = self._read_json()
                project_dir = str(payload.get("project_dir") or "").strip()
                asset_id = str(payload.get("asset_id") or payload.get("scene_id") or "").strip()
                query_text = str(payload.get("query") or "").strip()
                limit = int(payload.get("limit") or 6)
                if not project_dir:
                    raise ValueError("missing_project_dir")
                result = _mode2_recommend_background_candidates(
                    project_dir,
                    asset_id=asset_id,
                    query_text=query_text,
                    limit=max(1, min(limit, 12)),
                )
                self._send_json(result)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/storyboard-server-transfer":
            try:
                payload = self._read_json()
                project_dir = str(payload.get("project_dir") or "").strip()
                video_path = str(payload.get("video_path") or "").strip()
                segment_id = str(payload.get("segment_id") or "").strip()
                sampler_preset = str(payload.get("sampler_preset") or "balanced")
                video_window = payload.get("video_window") or {}
                normalize_size = bool(payload.get("normalize_size", True))
                raw_transfer_backend = payload.get("transfer_backend") or "scail2"
                transfer_backend = _normalize_transfer_backend(raw_transfer_backend)
                if transfer_backend is None:
                    self._send_json({"error": "invalid_transfer_backend"}, status=400)
                    return
                if transfer_backend == "wan22":
                    self._send_json({"error": "mode2_server_transfer_prefers_scail2"}, status=400)
                    return
                if not project_dir:
                    self._send_json({"error": "project_dir_required"}, status=400)
                    return
                if not video_path:
                    self._send_json({"error": "video_path_required"}, status=400)
                    return
                root = _resolve_storyboard_mode2_project_dir(project_dir)
                store_path = _storyboard_mode2_asset_store_path(root)
                data = json.loads(store_path.read_text(encoding="utf-8-sig"))
                if not segment_id:
                    shot = next((item for item in (data.get("shots") or []) if isinstance(item, dict)), None)
                    segment_id = str((shot or {}).get("segment_id") or "").strip()
                role_pairs, pair_warnings = _mode2_reference_mask_role_pairs_from_store(
                    str(root),
                    segment_id,
                    [],
                    prefer_current_shot_roles=True,
                )
                role_ids = _string_list(payload.get("role_ids"))
                if role_ids:
                    role_id_set = set(role_ids)
                    role_pairs = [
                        pair for pair in role_pairs
                        if str(pair.get("asset_id") or "") in role_id_set
                    ]
                if not role_pairs:
                    self._send_json({"error": "mode2_role_pairs_required"}, status=400)
                    return
                from spvideo.ffmpeg_tools import probe_video
                meta = probe_video(Path(video_path)) if Path(video_path).exists() else None
                if meta and meta.duration and meta.duration > 120:
                    self._send_json({"error": f"segment_too_long: {meta.duration:.1f}s"}, status=400)
                    return
                assets_by_id = {
                    str(asset.get("id") or ""): asset
                    for asset in (data.get("assets") or [])
                    if isinstance(asset, dict)
                }
                status_warnings: list[str] = []
                for pair in role_pairs:
                    asset = assets_by_id.get(str(pair.get("asset_id") or ""))
                    if not asset:
                        continue
                    track_status = str(asset.get("track_status") or asset.get("identity_status") or "").strip()
                    if track_status and track_status not in {"ready", "tracked"}:
                        status_warnings.append(f"{pair.get('name')}: track_status={track_status}")
                if status_warnings and not bool(payload.get("allow_unreviewed_tracks")):
                    self._send_json({
                        "error": "mode2_track_needs_review",
                        "message": "分轨还没通过，已阻止 Scail2 生成。请先查看分轨或重画锚点。",
                        "warnings": status_warnings,
                    }, status=400)
                    return
                job_id = uuid.uuid4().hex[:8]
                public_mapping = [
                    {
                        "order": index + 1,
                        "color": SCAIL2_COLOR_NAMES[index],
                        "role": pair.get("name"),
                        "target_ref": Path(str(pair.get("ref_image") or "")).name,
                        "source_x": (
                            float(pair["source_point"][0])
                            if isinstance(pair.get("source_point"), (list, tuple)) and pair.get("source_point")
                            else None
                        ),
                        "mark_time": pair.get("source_time"),
                        "asset_id": pair.get("asset_id"),
                    }
                    for index, pair in enumerate(role_pairs)
                ]
                with JOBS_LOCK:
                    JOBS[job_id] = {
                        "id": job_id,
                        "type": "mode2_server_transfer",
                        "status": "running",
                        "video_path": video_path,
                        "project_dir": str(root),
                        "segment_id": segment_id,
                        "mapping": public_mapping,
                        "mapping_warning": "；".join([*pair_warnings, *status_warnings]),
                        "transfer_backend": transfer_backend,
                        "logs": [
                            f"> Scail2 生成: {Path(video_path).name}",
                            f"> 当前后端: {transfer_backend}",
                            f"> 角色: {', '.join(str(pair.get('name') or '') for pair in role_pairs)}",
                            *[f"> 提醒: {line}" for line in pair_warnings],
                            *[f"> 分轨需复查: {line}" for line in status_warnings],
                        ],
                        "result": None,
                        "error": None,
                    }
                thread = threading.Thread(
                    target=_run_transfer_job,
                    args=(
                        job_id,
                        video_path,
                        role_pairs,
                        str(root),
                        "",
                        segment_id,
                        False,
                        sampler_preset,
                        video_window,
                        normalize_size,
                        str(payload.get("positive_prompt") or ""),
                        str(payload.get("sam_text") or ""),
                        transfer_backend,
                    ),
                    daemon=True,
                )
                thread.start()
                self._send_json({
                    "job_id": job_id,
                    "backend": transfer_backend,
                    "mapping": public_mapping,
                    "mapping_warning": "；".join([*pair_warnings, *status_warnings]),
                })
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/transfer-segment":
            try:
                payload = self._read_json()
                video_path = str(payload.get("video_path") or "").strip()
                project_dir = str(payload.get("project_dir") or "").strip()
                seg_id = str(payload.get("segment_id") or "").strip()
                protection_annotation_id = str(payload.get("protection_annotation_id") or "").strip()
                use_background = bool(payload.get("use_background_postprocess", False))
                sampler_preset = str(payload.get("sampler_preset") or "balanced")
                video_window = payload.get("video_window") or {}
                normalize_size = bool(payload.get("normalize_size", True))
                raw_transfer_backend = payload.get("transfer_backend")
                transfer_backend = _normalize_transfer_backend(raw_transfer_backend)
                if str(raw_transfer_backend or "").strip() and transfer_backend is None:
                    self._send_json({"error": "invalid_transfer_backend"}, status=400)
                    return

                # 检查视频时长
                from spvideo.ffmpeg_tools import probe_video
                meta = probe_video(Path(video_path)) if Path(video_path).exists() else None
                if meta and meta.duration:
                    dur = meta.duration
                    if dur > 120:
                        self._send_json({"error": f"分段时长 {dur:.1f}s，转绘支持最长 120 秒"}, status=400)
                        return

                job_id = uuid.uuid4().hex[:8]

                # 从资产库查角色对应的参考图
                store = load_asset_store(project_dir) if project_dir else {}
                characters = store.get("characters", {})
                originals = store.get("originals", [])
                annotations = store.get("annotations", [])
                director_plan = get_director_plan(
                    project_dir,
                    segment_id=seg_id,
                    video_path=video_path,
                ) if project_dir else None
                char_name = str(payload.get("character") or "").strip()
                raw_char_names = payload.get("characters") or []
                if isinstance(raw_char_names, str):
                    char_names = [name.strip() for name in raw_char_names.split(",") if name.strip()]
                else:
                    char_names = [str(name).strip() for name in raw_char_names if str(name).strip()]

                if director_plan and director_plan.get("roles"):
                    if director_plan.get("status") != "ready":
                        issues = "；".join(director_plan.get("issues") or ["配置不完整"])
                        self._send_json({"error": "导演计划未完成: " + issues}, status=400)
                        return
                    char_names = [
                        str(role.get("name") or "").strip()
                        for role in director_plan.get("roles", [])
                        if str(role.get("name") or "").strip()
                    ]
                    char_name = ", ".join(char_names)

                protection_annotation = None
                if protection_annotation_id:
                    protection_annotation = next(
                        (item for item in annotations if item.get("id") == protection_annotation_id),
                        None,
                    )
                    if not protection_annotation or protection_annotation.get("type") != "protection":
                        self._send_json({"error": "protection_annotation_not_found"}, status=400)
                        return
                    if not _same_video_path(protection_annotation.get("video_path"), video_path):
                        self._send_json({"error": "protection_video_mismatch"}, status=400)
                        return
                    track_dir = Path(str(protection_annotation.get("track_dir") or ""))
                    if protection_annotation.get("track_status") != "ready" or not (track_dir / "track_summary.json").exists():
                        self._send_json({"error": "protection_track_not_ready"}, status=400)
                        return

                # 如果前端没传角色名，从当前分段视频的标注中自动匹配
                if not char_name:
                    for ann in annotations:
                        if ann.get("type") == "person" and _same_video_path(ann.get("video_path"), video_path):
                            ln = str(ann.get("label_name") or "").strip()
                            if ln:
                                char_name = ln
                                char_names = [ln]
                                break
                    if not char_name:
                        for orig in originals:
                            ln = str(orig.get("label_name") or "").strip()
                            if ln and any(ch.get("name", "").strip() == ln for ch in characters.values()):
                                char_name = ln
                                char_names = [ln]
                                break
                if not char_names and char_name:
                    char_names = [char_name]

                role_pairs, mapping_warning = _resolve_scail2_role_pairs(
                    char_names=char_names,
                    characters=characters,
                    originals=originals,
                    annotations=annotations,
                    video_path=video_path,
                    director_plan=director_plan,
                )
                public_mapping = [
                    {
                        "order": index + 1,
                        "color": SCAIL2_COLOR_NAMES[index],
                        "role": pair["name"],
                        "target_ref": Path(pair["ref_image"]).name,
                        "source_x": pair.get("source_x"),
                        "mark_time": pair.get("source_time"),
                    }
                    for index, pair in enumerate(role_pairs)
                ]
                effective_transfer_backend = _transfer_backend_for_role_count(
                    len(role_pairs),
                    transfer_backend,
                )
                background_config = get_background_config(
                    project_dir,
                    segment_id=seg_id,
                    video_path=video_path,
                ) if project_dir else None
                if use_background:
                    if not background_config or background_config.get("mode") == "keep_original":
                        self._send_json({"error": "background_config_required"}, status=400)
                        return
                    ready_person_tracks = [
                        item
                        for item in annotations
                        if item.get("type") == "person"
                        and _same_video_path(item.get("video_path"), video_path)
                        and item.get("track_status") == "ready"
                        and (Path(str(item.get("track_dir") or "")) / "track_summary.json").exists()
                    ]
                    if not ready_person_tracks:
                        self._send_json({"error": "background_foreground_track_not_ready"}, status=400)
                        return

                with JOBS_LOCK:
                    JOBS[job_id] = {
                        "id": job_id,
                        "type": "transfer",
                        "status": "running",
                        "video_path": video_path,
                        "mapping": public_mapping,
                        "mapping_warning": mapping_warning,
                        "protection_annotation_id": protection_annotation_id,
                        "background_enabled": use_background,
                        "transfer_backend": effective_transfer_backend,
                        "director_plan_id": str((director_plan or {}).get("id") or ""),
                        "logs": [
                            f"> 开始转绘: {Path(video_path).name}",
                            (
                                "> 当前转绘后端: Wan2.2（直接使用云端回传视频）"
                                if effective_transfer_backend == "wan22"
                                else "> 当前转绘后端: SCAIL-2（多人蒙版条件生成）"
                            ),
                            *(["> 已加载片段导演计划"] if director_plan else []),
                            *([f"> 映射提醒: {mapping_warning}"] if mapping_warning else []),
                            *(["> 已启用保护区域，完成后自动还原"] if protection_annotation_id else []),
                            *(["> 已启用背景后处理，将在换人完成后串行执行"] if use_background else []),
                        ],
                        "result": None,
                        "error": None,
                    }

                thread = threading.Thread(
                    target=_run_transfer_job,
                    args=(
                        job_id,
                        video_path,
                        role_pairs,
                        project_dir,
                        protection_annotation_id,
                        seg_id,
                        use_background,
                        sampler_preset,
                        video_window,
                        normalize_size,
                        str((director_plan or {}).get("positive_prompt") or ""),
                        str((director_plan or {}).get("sam_text") or ""),
                        effective_transfer_backend,
                    ),
                    daemon=True,
                )
                thread.start()
                self._send_json({
                    "job_id": job_id,
                    "backend": effective_transfer_backend,
                    "mapping": public_mapping,
                    "mapping_warning": mapping_warning,
                })
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/load-result":
            try:
                payload = self._read_json()
                project_dir = payload.get("project_dir", "")
                project_dir = resolve_auto_director_project_root(project_dir)
                self._send_json(_load_project_result(project_dir))
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path == "/api/segment-edit":
            try:
                payload = self._read_json()
                project_dir = str(payload.get("project_dir") or "").strip()
                segment_id = str(payload.get("segment_id") or "").strip()
                action = str(payload.get("action") or "").strip()
                if not project_dir or not segment_id or not action:
                    self._send_json({"error": "缺少参数"}, status=400)
                    return
                result = _edit_segment(project_dir, segment_id, action)
                self._send_json(result)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, status=400)
            return

        if parsed.path != "/api/split":
            self._send_json({"error": "not_found"}, status=404)
            return
        try:
            payload = self._read_json()
            job_id = uuid.uuid4().hex
            job = {
                "id": job_id,
                "status": "running",
                "created_at": time.time(),
                "logs": ["> Web 后端已接收切分任务"],
                "result": None,
                "error": None,
            }
            with JOBS_LOCK:
                JOBS[job_id] = job
            thread = threading.Thread(target=_run_split_job, args=(job_id, payload), daemon=True)
            thread.start()
            self._send_json({"job_id": job_id})
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=400)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        data = self.rfile.read(length)
        if not data:
            return {}
        return json.loads(data.decode("utf-8"))

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._send_json({"error": "not_found"}, status=404)
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type = f"{content_type}; charset=utf-8"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        if path.suffix.lower() in {".html", ".htm"}:
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_media(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._send_json({"error": "media_not_found"}, status=404)
            return

        file_size = path.stat().st_size
        range_header = self.headers.get("Range")
        content_type = mimetypes.guess_type(str(path))[0] or "video/mp4"

        if range_header:
            try:
                unit, value = range_header.split("=", 1)
                if unit.strip() != "bytes":
                    raise ValueError("unsupported range unit")
                start_text, end_text = value.split("-", 1)
                start = int(start_text) if start_text else 0
                end = int(end_text) if end_text else file_size - 1
                end = min(end, file_size - 1)
                if start > end:
                    raise ValueError("invalid range")
            except Exception:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return

            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            with path.open("rb") as file:
                file.seek(start)
                self.wfile.write(file.read(length))
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(file_size))
        self.end_headers()
        with path.open("rb") as file:
            while True:
                chunk = file.read(1024 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)


def _latest_result_child(root: Path) -> Path | None:
    candidates: list[tuple[float, Path]] = []
    for child in root.iterdir() if root.exists() else []:
        if not child.is_dir():
            continue
        markers = [
            child / "manifest.json",
            child / "01_分析探针" / "two_pass_result.json",
            child / "01_probe" / "two_pass_result.json",
        ]
        existing = [marker for marker in markers if marker.exists()]
        if existing:
            candidates.append((max(marker.stat().st_mtime for marker in existing), child))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mode2_is_mask_delivery_path(value: Any) -> bool:
    text = str(value or "").strip().replace("\\", "/").lower()
    if not text:
        return False
    path_bits = [bit for bit in text.split("/") if bit]
    name = path_bits[-1] if path_bits else text
    mask_tokens = (
        "mask",
        "colored_mask",
        "candidate_mask",
        "maskcand_",
        "identity_candidates",
        "role_tracks",
        "source_mask",
    )
    return any(token in text for token in mask_tokens) or name.startswith("mask_")


def _mode2_filter_seedance_image_inputs(values: list[str], warnings: list[str], *, field: str) -> list[str]:
    kept: list[str] = []
    removed = 0
    for value in values:
        if _mode2_is_mask_delivery_path(value):
            removed += 1
            continue
        kept.append(value)
    if removed:
        warnings.append(f"Mode2 safety: removed {removed} mask-like image input(s) from {field}; masks are not sent to Seedance.")
    return kept


def _mode2_seedance_path_keys(value: Any) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    values = {text.replace("\\", "/").lower()}
    if text.startswith("data:"):
        return values
    try:
        decoded = _decode_media_ref_path(text)
    except Exception:  # noqa: BLE001
        decoded = text
    decoded_text = str(decoded or "").strip()
    if decoded_text:
        values.add(decoded_text.replace("\\", "/").lower())
        if not decoded_text.startswith(("http://", "https://")):
            try:
                values.add(os.path.normcase(os.path.abspath(decoded_text)))
            except OSError:
                pass
    return values


def _mode2_seedance_collect_path_values(value: Any) -> list[str]:
    result: list[str] = []

    def add(item: Any) -> None:
        if isinstance(item, str):
            text = item.strip()
            if text:
                result.append(text)
        elif isinstance(item, dict):
            for key in (
                "path",
                "image",
                "image_path",
                "crop_path",
                "cutout_path",
                "context_path",
                "mask_path",
                "source_image",
                "candidate_source_image",
                "refined_source_image",
                "refined_cutout_image",
                "refined_mask_image",
            ):
                add(item.get(key))
            for key in (
                "paths",
                "images",
                "source_images",
                "candidate_source_images",
                "identity_evidence_images",
                "source_crop_paths",
                "source_mask_paths",
            ):
                add(item.get(key))
        elif isinstance(item, list):
            for child in item:
                add(child)

    add(value)
    return result


def _mode2_seedance_generation_image_deny_keys(project_dir: str) -> set[str]:
    value = str(project_dir or "").strip()
    if not value:
        return set()
    try:
        root = _resolve_storyboard_mode2_project_dir(value)
        store_path = _storyboard_mode2_asset_store_path(root)
        if not store_path.exists():
            return set()
        data = json.loads(store_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:  # noqa: BLE001
        logging.warning("Mode2 Seedance image denylist skipped: %s", exc)
        return set()

    deny_fields = (
        "source_image",
        "source_images",
        "candidate_source_image",
        "candidate_source_images",
        "refined_source_image",
        "refined_source_images",
        "refined_cutout_image",
        "refined_mask_image",
        "identity_evidence",
        "identity_evidence_images",
        "source_crop_paths",
        "source_mask_paths",
        "mask_path",
        "mask_paths",
        "source_mask_path",
        "candidate_mask_path",
        "refined_mask_path",
    )
    deny: set[str] = set()
    for asset in data.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        for field in deny_fields:
            for path_value in _mode2_seedance_collect_path_values(asset.get(field)):
                deny.update(_mode2_seedance_path_keys(path_value))
    return deny


def _mode2_filter_seedance_generation_images(
    values: list[str],
    warnings: list[str],
    *,
    field: str,
    deny_keys: set[str],
) -> list[str]:
    filtered = _mode2_filter_seedance_image_inputs(values, warnings, field=field)
    if not deny_keys:
        return filtered
    kept: list[str] = []
    removed = 0
    for value in filtered:
        keys = _mode2_seedance_path_keys(value)
        if keys and keys.intersection(deny_keys):
            removed += 1
            continue
        kept.append(value)
    if removed:
        warnings.append(f"Mode2 safety: removed {removed} source/evidence image input(s) from {field}; only target replacement images should be sent to Seedance.")
    return kept


def _mode2_asset_layers_summary(assets: list[dict[str, Any]]) -> dict[str, Any]:
    layers: dict[str, Any] = {
        "identity_evidence": {"asset_ids": [], "image_count": 0, "evidence_count": 0, "seedance_delivery": "never"},
        "target_reference": {"asset_ids": [], "image_count": 0, "seedance_delivery": "allowed"},
        "scene_context": {"asset_ids": [], "image_count": 0, "seedance_delivery": "optional"},
        "prop_context": {"asset_ids": [], "image_count": 0, "seedance_delivery": "optional"},
    }
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        asset_id = str(asset.get("id") or "").strip()
        kind = str(asset.get("kind") or "").strip()
        source_images = [
            str(value).strip()
            for value in [
                asset.get("source_image"),
                asset.get("candidate_source_image"),
                asset.get("refined_source_image"),
                *(asset.get("source_images") or []),
                *(asset.get("candidate_source_images") or []),
                *(asset.get("identity_evidence_images") or []),
            ]
            if str(value or "").strip()
        ]
        if kind == "role":
            if asset_id:
                layers["identity_evidence"]["asset_ids"].append(asset_id)
            layers["identity_evidence"]["image_count"] += len(set(source_images))
            layers["identity_evidence"]["evidence_count"] += len([
                item for item in (asset.get("identity_evidence") or []) if isinstance(item, dict)
            ])
            if str(asset.get("target_image") or "").strip():
                layers["target_reference"]["image_count"] += 1
                if asset_id:
                    layers["target_reference"]["asset_ids"].append(asset_id)
        elif kind == "scene":
            if asset_id:
                layers["scene_context"]["asset_ids"].append(asset_id)
            layers["scene_context"]["image_count"] += len(set(source_images))
        elif kind == "prop":
            if asset_id:
                layers["prop_context"]["asset_ids"].append(asset_id)
            layers["prop_context"]["image_count"] += len(set(source_images))
    for layer in layers.values():
        layer["asset_ids"] = list(dict.fromkeys(layer["asset_ids"]))
    return layers


def _mode2_mask_candidate_summary(root: Path, data: dict[str, Any]) -> dict[str, Any]:
    annotations = [item for item in (data.get("identity_annotations") or []) if isinstance(item, dict)]
    selected_ids = {
        str(item.get("mask_candidate_id") or "").strip()
        for item in annotations
        if str(item.get("mask_candidate_id") or "").strip()
    }
    summary: dict[str, Any] = {
        "source": "offline_cache",
        "status": "empty",
        "candidate_count": 0,
        "object_count": 0,
        "selected_candidate_count": 0,
        "selected_object_count": 0,
        "ready_count": 0,
        "items": [],
        "by_asset": {},
        "seedance_delivery": "never",
        "note": "Mask candidates are identity/shot-boundary clues only; they are never Seedance inputs.",
    }
    base = root / "assets" / "identity_candidates"
    if not base.exists():
        summary["cache_dir"] = str(base)
        return summary
    for result_path in sorted(base.glob("maskcand_*/result.json")):
        try:
            item = json.loads(result_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:  # noqa: BLE001
            logging.warning("Mode2 mask candidate summary ignored: %s / %s", result_path, exc)
            continue
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id") or result_path.parent.name).strip()
        objects = [obj for obj in (item.get("objects") or []) if isinstance(obj, dict)]
        selected_objects = [
            obj for obj in objects
            if any(_mode2_anchor_matches_candidate_object(anchor, candidate_id=candidate_id, obj=obj) for anchor in annotations)
        ]
        ready = item.get("ok") is not False and str(item.get("mask_status") or "") in {"", "ready"}
        summary["candidate_count"] += 1
        summary["object_count"] += len(objects)
        summary["selected_object_count"] += len(selected_objects)
        if candidate_id in selected_ids:
            summary["selected_candidate_count"] += 1
        if ready:
            summary["ready_count"] += 1
        asset_id = str(item.get("asset_id") or "").strip()
        if asset_id:
            per_asset = summary["by_asset"].setdefault(asset_id, {
                "candidate_count": 0,
                "object_count": 0,
                "selected_object_count": 0,
                "ready_count": 0,
                "items": [],
                "status": "empty",
            })
            per_asset["candidate_count"] += 1
            per_asset["object_count"] += len(objects)
            per_asset["selected_object_count"] += len(selected_objects)
            if ready:
                per_asset["ready_count"] += 1
            per_asset["items"].append(candidate_id)
        summary["items"].append({
            "candidate_id": candidate_id,
            "time": item.get("time"),
            "asset_id": asset_id,
            "object_count": len(objects),
            "selected_object_count": len(selected_objects),
            "status": "ready" if ready else str(item.get("mask_status") or "failed"),
            "result_path": str(result_path),
            "original_frame": str(item.get("original_frame") or ""),
            "colored_mask": str(item.get("colored_mask") or ""),
            "overlay_image": str(item.get("overlay_image") or ""),
        })
    summary["cache_dir"] = str(base)
    if summary["candidate_count"] <= 0:
        summary["status"] = "empty"
    elif summary["ready_count"] >= summary["candidate_count"]:
        summary["status"] = "ready"
    elif summary["ready_count"] > 0:
        summary["status"] = "partial"
    else:
        summary["status"] = "failed"
    for per_asset in summary["by_asset"].values():
        if per_asset["candidate_count"] <= 0:
            per_asset["status"] = "empty"
        elif per_asset["ready_count"] >= per_asset["candidate_count"]:
            per_asset["status"] = "ready"
        elif per_asset["ready_count"] > 0:
            per_asset["status"] = "partial"
        else:
            per_asset["status"] = "failed"
    return summary


def _mode2_shot_asset_groups(shots: list[dict[str, Any]], assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(asset.get("id") or ""): asset for asset in assets if isinstance(asset, dict)}
    groups: list[dict[str, Any]] = []
    for shot in shots:
        if not isinstance(shot, dict):
            continue
        shot_id = str(shot.get("segment_id") or "").strip()
        asset_ids = [str(value).strip() for value in (shot.get("asset_ids") or []) if str(value).strip()]
        roles: list[dict[str, Any]] = []
        scenes: list[dict[str, Any]] = []
        props: list[dict[str, Any]] = []
        for asset_id in asset_ids:
            asset = by_id.get(asset_id)
            if not asset:
                continue
            ref = {
                "asset_id": asset_id,
                "name": str(asset.get("name") or asset.get("tag") or asset_id),
                "target_image": str(asset.get("target_image") or ""),
                "seedance_reference_image": str(asset.get("seedance_reference_image") or asset.get("target_image") or ""),
                "seedance_reference_role": str(asset.get("seedance_reference_role") or ""),
                "status": str(asset.get("status") or ""),
            }
            kind = str(asset.get("kind") or "")
            if kind == "role":
                roles.append(ref)
            elif kind == "scene":
                scenes.append(ref)
            elif kind == "prop":
                props.append(ref)
        group = {
            "shot_id": shot_id,
            "start": shot.get("start"),
            "end": shot.get("end"),
            "asset_ids": asset_ids,
            "role_asset_ids": [item["asset_id"] for item in roles],
            "scene_asset_ids": [item["asset_id"] for item in scenes],
            "prop_asset_ids": [item["asset_id"] for item in props],
            "roles": roles,
            "scenes": scenes,
            "props": props,
            "seedance_reference_images": [
                item["seedance_reference_image"]
                for item in [*roles, *scenes, *props]
                if item.get("seedance_reference_image") and not _mode2_is_mask_delivery_path(item.get("seedance_reference_image"))
            ],
        }
        shot["asset_group"] = group
        shot["role_asset_ids"] = group["role_asset_ids"]
        shot["scene_asset_ids"] = group["scene_asset_ids"]
        shot["prop_asset_ids"] = group["prop_asset_ids"]
        shot["seedance_reference_images"] = group["seedance_reference_images"]
        groups.append(group)
    return groups


def _mode2_asset_has_issue_code(asset: dict[str, Any], code: str) -> bool:
    return any(
        isinstance(issue, dict) and str(issue.get("code") or "") == code
        for issue in (asset.get("asset_audit_issues") or [])
    )


def _mode2_asset_is_manually_approved(asset: dict[str, Any]) -> bool:
    return str(asset.get("manual_asset_status") or "").strip() == "approved"


def _mode2_scene_frame_is_person_heavy(frame: dict[str, Any]) -> bool:
    metrics = frame.get("metrics") if isinstance(frame.get("metrics"), dict) else {}
    try:
        skin = float(metrics.get("skin_ratio") or 0.0)
        center_skin = float(metrics.get("center_skin_ratio") or 0.0)
    except (TypeError, ValueError):
        skin = center_skin = 0.0
    return skin >= 0.16 or center_skin >= 0.18


def _mode2_scene_group_best_image(frames: list[dict[str, Any]]) -> str:
    if not frames:
        return ""
    ranked = sorted(
        frames,
        key=lambda item: (-float(item.get("score") or 0.0), float(item.get("time") or 0.0)),
    )
    return str(ranked[0].get("path") or ranked[0].get("crop_path") or "").strip()


def _mode2_scene_group_payload(
    group_id: str,
    label: str,
    role: str,
    frames: list[dict[str, Any]],
    note: str,
) -> dict[str, Any]:
    cleaned_frames: list[dict[str, Any]] = []
    for frame in frames:
        path = str(frame.get("path") or frame.get("crop_path") or "").strip()
        if not path:
            continue
        cleaned_frames.append({
            "time": round(float(frame.get("time") or 0.0), 3),
            "path": path,
            "score": round(float(frame.get("score") or 0.0), 4),
            "reason": str(frame.get("reason") or ""),
            "metrics": frame.get("metrics") if isinstance(frame.get("metrics"), dict) else {},
        })
    return {
        "id": group_id,
        "label": label,
        "role": role,
        "note": note,
        "frame_count": len(cleaned_frames),
        "best_image": _mode2_scene_group_best_image(cleaned_frames),
        "frames": sorted(cleaned_frames, key=lambda item: float(item.get("time") or 0.0)),
    }


MODE2_SCENE_VISUAL_GROUP_KEYS: tuple[str, ...] = (
    "visual_group_id",
    "visual_cluster_id",
    "scene_group_id",
    "scene_cluster_id",
    "physical_scene_id",
    "location_id",
    "environment_id",
)


def _mode2_scene_explicit_visual_group_ids(
    asset: dict[str, Any],
    shots: list[dict[str, Any]] | None = None,
) -> list[str]:
    values: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in values:
            values.append(text)

    def add_from(record: dict[str, Any]) -> None:
        for key in MODE2_SCENE_VISUAL_GROUP_KEYS:
            add(record.get(key))
        for key in ("visual_group_ids", "visual_cluster_ids", "scene_group_ids", "scene_cluster_ids"):
            for value in record.get(key) or []:
                add(value)
        for nested_key in ("metadata", "visual_metadata", "semantic_scene", "scene_metadata"):
            nested = record.get(nested_key)
            if isinstance(nested, dict):
                for key in (*MODE2_SCENE_VISUAL_GROUP_KEYS, "scene_id"):
                    add(nested.get(key))

    add_from(asset)
    for frame in asset.get("keyframes") or []:
        if isinstance(frame, dict):
            add_from(frame)
    for group in asset.get("scene_candidate_groups") or []:
        if not isinstance(group, dict):
            continue
        group_id = str(group.get("id") or "").strip()
        if group_id and group_id not in {"environment", "shot_reference"}:
            add(group_id)
        add_from(group)

    asset_source_ids = {
        str(value or "").strip()
        for value in (asset.get("source_segment_ids") or [])
        if str(value or "").strip()
    }
    asset_used_shots = {
        str(value or "").strip()
        for value in (asset.get("used_shots") or [])
        if str(value or "").strip()
    }
    for shot in shots or []:
        if not isinstance(shot, dict):
            continue
        shot_id = str(shot.get("segment_id") or "").strip()
        shot_source_ids = {
            str(value or "").strip()
            for value in (shot.get("source_segment_ids") or [])
            if str(value or "").strip()
        }
        if asset_used_shots and shot_id in asset_used_shots:
            add_from(shot)
        elif asset_source_ids and shot_source_ids and asset_source_ids & shot_source_ids:
            add_from(shot)
    return values


def _mode2_scene_visual_hash_group_count(frames: list[dict[str, Any]]) -> tuple[int, int]:
    hashes: list[str] = []
    evidence_count = 0
    for frame in frames[:12]:
        path = str(frame.get("path") or frame.get("crop_path") or "").strip()
        if not path:
            continue
        evidence_count += 1
        try:
            value = _mode2_asset_image_dhash(path)
        except Exception:  # noqa: BLE001
            continue
        if value not in hashes:
            hashes.append(value)
    if not hashes:
        return 0, evidence_count
    representatives: list[str] = []
    for value in hashes:
        if not representatives or all(
            _mode2_asset_hamming_hex(value, representative) > 20
            for representative in representatives
        ):
            representatives.append(value)
    return len(representatives), evidence_count


def _mode2_scene_frame_explicit_group_id(frame: dict[str, Any]) -> str:
    for key in MODE2_SCENE_VISUAL_GROUP_KEYS:
        value = str(frame.get(key) or "").strip()
        if value:
            return value
    for nested_key in ("metadata", "visual_metadata", "semantic_scene", "scene_metadata"):
        nested = frame.get(nested_key)
        if not isinstance(nested, dict):
            continue
        for key in (*MODE2_SCENE_VISUAL_GROUP_KEYS, "scene_id"):
            value = str(nested.get(key) or "").strip()
            if value:
                return value
    return ""


def _mode2_scene_frame_time(frame: dict[str, Any], fallback: float = 0.0) -> float:
    for key in ("time", "source_time", "timestamp"):
        if frame.get(key) is None:
            continue
        try:
            return float(frame.get(key))
        except (TypeError, ValueError):
            continue
    return fallback


def _mode2_scene_frame_records(asset: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    raw_frames = [
        item for item in (asset.get("keyframes") or [])
        if isinstance(item, dict)
    ]
    if not raw_frames:
        source_time = _mode2_float(asset.get("source_time"), 0.0)
        source_paths = _string_list(asset.get("source_images"))
        if not source_paths and str(asset.get("source_image") or "").strip():
            source_paths = [str(asset.get("source_image") or "").strip()]
        raw_frames = [
            {"path": str(path), "time": source_time + index * 0.001, "metrics": {}}
            for index, path in enumerate(source_paths)
            if str(path or "").strip()
        ]
    for index, frame in enumerate(raw_frames):
        path = str(frame.get("path") or frame.get("crop_path") or "").strip()
        if not path:
            continue
        record = copy.deepcopy(frame)
        record["path"] = path
        record["time"] = round(_mode2_scene_frame_time(frame, float(index) * 0.001), 3)
        try:
            record["_visual_hash"] = _mode2_asset_image_dhash(path)
        except Exception:  # noqa: BLE001
            record["_visual_hash"] = ""
        record["_explicit_group_id"] = _mode2_scene_frame_explicit_group_id(frame)
        records.append(record)
    return records


def _mode2_scene_visual_frame_groups(asset: dict[str, Any]) -> list[dict[str, Any]]:
    """Return actual frame groups; detection alone must never mutate the asset."""
    frames = _mode2_scene_frame_records(asset)
    if len(frames) < 2:
        return []

    explicit_path_groups: dict[str, str] = {}
    for candidate in asset.get("scene_candidate_groups") or []:
        if not isinstance(candidate, dict):
            continue
        group_id = str(candidate.get("id") or "").strip()
        if not group_id or group_id in {"environment", "shot_reference"}:
            continue
        for frame in candidate.get("frames") or []:
            if not isinstance(frame, dict):
                continue
            path = str(frame.get("path") or frame.get("crop_path") or "").strip()
            if path:
                explicit_path_groups[path] = group_id

    groups: list[dict[str, Any]] = []
    by_explicit: dict[str, dict[str, Any]] = {}
    unresolved: list[dict[str, Any]] = []
    for frame in frames:
        group_id = str(frame.get("_explicit_group_id") or explicit_path_groups.get(frame["path"]) or "").strip()
        if not group_id:
            unresolved.append(frame)
            continue
        group = by_explicit.setdefault(group_id, {"id": group_id, "method": "explicit", "frames": []})
        group["frames"].append(frame)
    groups.extend(by_explicit.values())

    for frame in unresolved:
        visual_hash = str(frame.get("_visual_hash") or "")
        best_group: dict[str, Any] | None = None
        best_distance = 999
        for group in groups:
            representative_hash = str(group.get("representative_hash") or "")
            if not representative_hash:
                representative = next(
                    (item for item in group["frames"] if str(item.get("_visual_hash") or "")),
                    None,
                )
                representative_hash = str((representative or {}).get("_visual_hash") or "")
            if not visual_hash or not representative_hash:
                continue
            distance = _mode2_asset_hamming_hex(visual_hash, representative_hash)
            if distance < best_distance:
                best_distance = distance
                best_group = group
        if best_group is not None and best_distance <= 20:
            best_group["frames"].append(frame)
            best_group.setdefault("method", "dhash")
            continue
        groups.append({
            "id": f"visual_{len(groups) + 1}",
            "method": "dhash",
            "representative_hash": visual_hash,
            "frames": [frame],
        })

    cleaned: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        group_frames = sorted(group.get("frames") or [], key=lambda item: _mode2_scene_frame_time(item))
        if not group_frames:
            continue
        for frame in group_frames:
            frame.pop("_visual_hash", None)
            frame.pop("_explicit_group_id", None)
        cleaned.append({
            "id": str(group.get("id") or f"visual_{index}"),
            "method": str(group.get("method") or "dhash"),
            "frames": group_frames,
            "best_image": _mode2_scene_group_best_image(group_frames),
        })
    return cleaned if len(cleaned) >= 2 else []


def _mode2_scene_time_matches_shot(time_value: float, shot: dict[str, Any]) -> bool:
    start = _mode2_float(shot.get("start"), 0.0)
    end = _mode2_float(shot.get("end"), start)
    return start - 0.12 <= time_value <= end + 0.12


def _mode2_scene_asset_is_superseded(asset: dict[str, Any]) -> bool:
    return bool(
        asset.get("superseded")
        or str(asset.get("status") or "").strip() == "superseded"
        or str(asset.get("source_usage_role") or "").strip() == "superseded_parent"
    )


def _mode2_scene_assets_are_duplicates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_frames = _mode2_scene_frame_records(left)
    right_frames = _mode2_scene_frame_records(right)
    left_hashes = [str(item.get("_visual_hash") or "") for item in left_frames if item.get("_visual_hash")]
    right_hashes = [str(item.get("_visual_hash") or "") for item in right_frames if item.get("_visual_hash")]
    if not left_hashes or not right_hashes:
        return False

    matched_right: set[int] = set()
    content_matches = 0
    for left_hash in left_hashes:
        candidate = next(
            (
                index for index, right_hash in enumerate(right_hashes)
                if index not in matched_right and _mode2_asset_hamming_hex(left_hash, right_hash) <= 2
            ),
            None,
        )
        if candidate is None:
            continue
        matched_right.add(candidate)
        content_matches += 1
    if content_matches / max(len(left_hashes), len(right_hashes)) < 0.75:
        return False

    left_segments = set(_string_list(left.get("source_segment_ids")))
    right_segments = set(_string_list(right.get("source_segment_ids")))
    if left_segments and right_segments and left_segments & right_segments:
        return True

    left_times = [_mode2_scene_frame_time(item) for item in left_frames]
    right_times = [_mode2_scene_frame_time(item) for item in right_frames]
    matched_times: set[int] = set()
    time_matches = 0
    for left_time in left_times:
        candidate = next(
            (
                index for index, right_time in enumerate(right_times)
                if index not in matched_times and abs(left_time - right_time) <= 0.12
            ),
            None,
        )
        if candidate is None:
            continue
        matched_times.add(candidate)
        time_matches += 1
    return bool(time_matches / max(len(left_times), len(right_times)) >= 0.75)


def _mode2_scene_canonical_rank(asset: dict[str, Any], position: int) -> tuple[int, int, int, int]:
    trusted = int(bool(
        str(asset.get("target_image") or "").strip()
        or str(asset.get("refined_source_image") or "").strip()
        or str(asset.get("manual_asset_status") or "").strip() == "approved"
    ))
    return (
        trusted,
        len(_mode2_scene_frame_records(asset)),
        len(_string_list(asset.get("used_shots"))),
        -position,
    )


def _mode2_merge_scene_duplicate_metadata(
    canonical: dict[str, Any],
    duplicate: dict[str, Any],
) -> None:
    for key in ("source_segment_ids", "used_shots", "evidence_shots", "source_images", "candidate_source_images"):
        canonical[key] = list(dict.fromkeys([
            *_string_list(canonical.get(key)),
            *_string_list(duplicate.get(key)),
        ]))

    frames = [
        copy.deepcopy(item) for item in (canonical.get("keyframes") or [])
        if isinstance(item, dict)
    ]
    seen: set[tuple[str, float]] = set()
    for frame in frames:
        path = str(frame.get("path") or frame.get("crop_path") or "").strip()
        try:
            visual_hash = _mode2_asset_image_dhash(path)
        except Exception:  # noqa: BLE001
            visual_hash = path
        seen.add((visual_hash, round(_mode2_scene_frame_time(frame), 2)))
    for frame in duplicate.get("keyframes") or []:
        if not isinstance(frame, dict):
            continue
        path = str(frame.get("path") or frame.get("crop_path") or "").strip()
        try:
            visual_hash = _mode2_asset_image_dhash(path)
        except Exception:  # noqa: BLE001
            visual_hash = path
        key = (visual_hash, round(_mode2_scene_frame_time(frame), 2))
        if key in seen:
            continue
        frames.append(copy.deepcopy(frame))
        seen.add(key)
    if frames:
        canonical["keyframes"] = sorted(frames, key=lambda item: _mode2_scene_frame_time(item))

    ranges = [
        value for value in (canonical.get("source_time_range"), duplicate.get("source_time_range"))
        if isinstance(value, (list, tuple)) and len(value) >= 2
    ]
    if ranges:
        canonical["source_time_range"] = [
            round(min(_mode2_float(value[0]) for value in ranges), 3),
            round(max(_mode2_float(value[1]) for value in ranges), 3),
        ]
    duplicate_id = str(duplicate.get("id") or "").strip()
    canonical["deduplicated_asset_ids"] = list(dict.fromkeys([
        *_string_list(canonical.get("deduplicated_asset_ids")),
        duplicate_id,
        *_string_list(duplicate.get("deduplicated_asset_ids")),
    ]))


def _mode2_replace_shot_asset_aliases(
    shots: list[dict[str, Any]],
    aliases: dict[str, str],
) -> None:
    if not aliases:
        return
    for shot in shots:
        if not isinstance(shot, dict):
            continue
        for key in ("asset_ids", "scene_asset_ids"):
            values = _string_list(shot.get(key))
            if not values:
                continue
            shot[key] = list(dict.fromkeys(aliases.get(value, value) for value in values if aliases.get(value, value)))


def _mode2_dedupe_scene_assets(data: dict[str, Any]) -> dict[str, Any]:
    assets = [item for item in (data.get("assets") or []) if isinstance(item, dict)]
    shots = [item for item in (data.get("shots") or []) if isinstance(item, dict)]
    scenes = [
        (position, asset) for position, asset in enumerate(assets)
        if str(asset.get("kind") or "") == "scene" and not _mode2_scene_asset_is_superseded(asset)
    ]
    parent: dict[str, str] = {
        str(asset.get("id") or ""): str(asset.get("id") or "")
        for _, asset in scenes if str(asset.get("id") or "").strip()
    }

    def find(asset_id: str) -> str:
        while parent.get(asset_id, asset_id) != asset_id:
            parent[asset_id] = parent.get(parent[asset_id], parent[asset_id])
            asset_id = parent[asset_id]
        return asset_id

    def union(left_id: str, right_id: str) -> None:
        left_root = find(left_id)
        right_root = find(right_id)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, (_, left) in enumerate(scenes):
        left_id = str(left.get("id") or "").strip()
        if not left_id:
            continue
        for _, right in scenes[left_index + 1:]:
            right_id = str(right.get("id") or "").strip()
            if right_id and _mode2_scene_assets_are_duplicates(left, right):
                union(left_id, right_id)

    components: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for position, asset in scenes:
        asset_id = str(asset.get("id") or "").strip()
        if asset_id:
            components.setdefault(find(asset_id), []).append((position, asset))

    aliases: dict[str, str] = {}
    removed_ids: set[str] = set()
    canonical_ids: list[str] = []
    for component in components.values():
        if len(component) < 2:
            continue
        canonical_position, canonical = max(
            component,
            key=lambda item: _mode2_scene_canonical_rank(item[1], item[0]),
        )
        del canonical_position
        canonical_id = str(canonical.get("id") or "").strip()
        canonical_ids.append(canonical_id)
        for _, duplicate in component:
            duplicate_id = str(duplicate.get("id") or "").strip()
            if not duplicate_id or duplicate_id == canonical_id:
                continue
            _mode2_merge_scene_duplicate_metadata(canonical, duplicate)
            aliases[duplicate_id] = canonical_id
            removed_ids.add(duplicate_id)

    if aliases:
        data["assets"] = [
            asset for asset in assets
            if str(asset.get("id") or "").strip() not in removed_ids
        ]
        _mode2_replace_shot_asset_aliases(shots, aliases)
        data["shots"] = shots
        previous_aliases = data.get("scene_asset_aliases") if isinstance(data.get("scene_asset_aliases"), dict) else {}
        data["scene_asset_aliases"] = {**previous_aliases, **aliases}
        data["scene_dedupe"] = {
            "deduplicated_at": time.time(),
            "removed_count": len(removed_ids),
            "canonical_ids": canonical_ids,
            "aliases": aliases,
        }
    return {
        "changed": bool(aliases),
        "removed_count": len(removed_ids),
        "canonical_ids": canonical_ids,
        "aliases": aliases,
    }


def _mode2_update_scene_visual_groups(
    asset: dict[str, Any],
    shots: list[dict[str, Any]] | None = None,
) -> None:
    if str(asset.get("kind") or "") != "scene":
        return
    keyframes = [
        item for item in (asset.get("keyframes") or [])
        if isinstance(item, dict)
    ]
    environment_frames = [
        item for item in keyframes
        if not _mode2_scene_frame_is_person_heavy(item)
    ]
    shot_reference_frames = [
        item for item in keyframes
        if _mode2_scene_frame_is_person_heavy(item)
    ]
    preserved_explicit_groups = [
        copy.deepcopy(group)
        for group in (asset.get("scene_candidate_groups") or [])
        if isinstance(group, dict)
        and str(group.get("id") or "").strip()
        and str(group.get("id") or "").strip() not in {"environment", "shot_reference"}
    ]
    explicit_group_ids = _mode2_scene_explicit_visual_group_ids(asset, shots)
    visual_hash_group_count, evidence_count = _mode2_scene_visual_hash_group_count(keyframes)
    prior_mixed = bool(asset.get("scene_mixed_visual_groups"))
    groups: list[dict[str, Any]] = []
    if environment_frames:
        groups.append(_mode2_scene_group_payload(
            "environment",
            "环境/场景候选",
            "scene_reference_candidate",
            environment_frames,
            "人物干扰少，可优先当场景/空间参考；真正生成仍建议上传目标场景图。",
        ))
    if shot_reference_frames:
        groups.append(_mode2_scene_group_payload(
            "shot_reference",
            "人物近景/动作参考",
            "shot_motion_reference_only",
            shot_reference_frames,
            "人物占比高，只适合看动作、构图、遮挡关系，不适合作为干净场景替换图。",
        ))
    asset["scene_candidate_groups"] = [
        group for group in [*preserved_explicit_groups, *groups]
        if group.get("frame_count") or group.get("frames")
    ]
    asset["scene_clean_candidate_image"] = _mode2_scene_group_best_image(environment_frames)
    asset["scene_shot_reference_image"] = _mode2_scene_group_best_image(shot_reference_frames)
    unresolved_multiple_evidence = bool(
        evidence_count > 1
        and visual_hash_group_count == 0
        and len(explicit_group_ids) <= 1
    )
    mixed = bool(
        prior_mixed
        or (environment_frames and shot_reference_frames)
        or len(explicit_group_ids) > 1
        or visual_hash_group_count > 1
        or unresolved_multiple_evidence
    )
    asset["scene_visual_group_ids"] = explicit_group_ids
    asset["scene_visual_group_count"] = max(len(explicit_group_ids), visual_hash_group_count)
    asset["scene_visual_hash_group_count"] = visual_hash_group_count
    asset["scene_visual_groups_unverified"] = unresolved_multiple_evidence
    asset["scene_mixed_visual_groups"] = mixed
    if mixed:
        asset["source_quality_status"] = "scene_mixed_visual_groups"
        asset["source_usage_role"] = "mixed_reference_bundle"
        asset["source_quality_warning"] = (
            "这组场景包含多个视觉组、不同物理空间或人物近景证据，不能作为一个单一场景资产替换；"
            "必须先按真实场景拆分，人物近景只作镜头/动作参考。"
        )


def _mode2_scene_is_mixed_visual_bundle(asset: dict[str, Any]) -> bool:
    return (
        bool(asset.get("scene_mixed_visual_groups"))
        or str(asset.get("source_quality_status") or "").strip() == "scene_mixed_visual_groups"
        or str(asset.get("source_usage_role") or "").strip() == "mixed_reference_bundle"
    )


def _mode2_scene_is_shot_reference_only(asset: dict[str, Any]) -> bool:
    return (
        str(asset.get("source_quality_status") or "").strip() == "scene_person_heavy"
        or str(asset.get("source_usage_role") or "").strip() == "shot_reference"
        or str(asset.get("manual_asset_status") or "").strip() == "shot_reference"
    )


def _mode2_scene_refinement_is_untrusted(asset: dict[str, Any]) -> bool:
    if str(asset.get("kind") or "") != "scene":
        return False
    provenance_status = str(asset.get("refinement_provenance_status") or "").strip()
    return (
        bool(asset.get("refinement_quarantined"))
        or str(asset.get("refinement_status") or "").strip() == "quarantined"
        or bool(provenance_status and provenance_status not in {"verified", "not_applicable"})
    )


def _mode2_prop_is_unconfirmed_candidate(asset: dict[str, Any]) -> bool:
    return (
        str(asset.get("source_quality_status") or "").strip() == "prop_needs_visual_check"
        or str(asset.get("source_usage_role") or "").strip() == "object_evidence_candidate"
        or str(asset.get("source_selection_method") or "").strip() == "semantic_keyword_candidate"
        or str(asset.get("source_trust_level") or "").strip() == "semantic_candidate_only"
        or str(asset.get("source_kind") or "").strip() == "prop_keyframe_bundle"
        or str(asset.get("track_status") or "").strip() == "needs_manual_review"
    )


def _mode2_prop_refinement_is_clean(asset: dict[str, Any]) -> bool:
    if str(asset.get("refinement_status") or "") not in {"ready", ""}:
        return False
    if not str(asset.get("refined_cutout_image") or asset.get("refined_source_image") or "").strip():
        return False
    if _mode2_asset_has_issue_code(asset, "mask_quality_unclean"):
        return False
    if _mode2_prop_is_unconfirmed_candidate(asset) and not _mode2_asset_is_manually_approved(asset):
        return False
    quality = asset.get("refinement_quality")
    if not isinstance(quality, dict) or not quality:
        return str(asset.get("refinement_status") or "") == "ready" and not str(asset.get("refinement_warning") or "").strip()
    area_ratio = _mode2_asset_quality_number(quality.get("area_ratio"))
    bbox_ratio = _mode2_asset_quality_number(quality.get("bbox_ratio"))
    component_count = _mode2_asset_quality_number(quality.get("component_count"))
    if area_ratio is not None and not 0.0005 < area_ratio < 0.22:
        return False
    if bbox_ratio is not None and bbox_ratio >= 0.58:
        return False
    if component_count is not None and component_count >= 7:
        return False
    return True


def _mode2_asset_display_fields(asset: dict[str, Any], evidence_images: list[str]) -> dict[str, Any]:
    kind = str(asset.get("kind") or "").strip()
    target_image = str(asset.get("target_image") or "").strip()
    refined_source = str(asset.get("refined_source_image") or "").strip()
    refined_cutout = str(asset.get("refined_cutout_image") or "").strip()
    source_image = str(asset.get("source_image") or "").strip()
    candidate_image = str(asset.get("candidate_source_image") or "").strip()
    refinement_provenance_verified = str(asset.get("refinement_provenance_status") or "") == "verified"
    candidate_preview = ""
    candidate_source = ""
    candidate_label = "候选预览图"
    if kind == "prop":
        candidate_sources = [
            ("物品裁剪候选", refined_cutout),
            ("提纯候选", refined_source),
            ("原片候选帧", source_image),
            ("候选源图", candidate_image),
            ("证据帧", evidence_images[0] if evidence_images else ""),
        ]
    elif kind == "scene":
        scene_clean_candidate = str(asset.get("scene_clean_candidate_image") or "").strip()
        scene_shot_reference = str(asset.get("scene_shot_reference_image") or "").strip()
        candidate_sources = [
            ("环境/场景候选", scene_clean_candidate),
            ("原片候选帧", source_image),
            ("候选源图", candidate_image),
            ("人物近景/动作参考", scene_shot_reference),
            ("提纯候选", refined_source if refinement_provenance_verified else ""),
            ("证据帧", evidence_images[0] if evidence_images else ""),
        ]
    else:
        candidate_sources = [
            ("原片候选帧", source_image),
            ("候选源图", candidate_image),
            ("提纯候选", refined_cutout or refined_source),
            ("证据帧", evidence_images[0] if evidence_images else ""),
        ]
    if (
        (
            kind == "prop"
            and _mode2_prop_is_unconfirmed_candidate(asset)
        )
        or (
            kind == "scene"
            and (
                _mode2_scene_is_mixed_visual_bundle(asset)
                or _mode2_scene_is_shot_reference_only(asset)
            )
        )
    ) and not _mode2_asset_is_manually_approved(asset):
        candidate_sources = []

    for label, value in candidate_sources:
        path = str(value or "").strip()
        if path:
            candidate_preview = path
            candidate_source = label
            candidate_label = label
            break
    display: dict[str, Any] = {
        "image": "",
        "label": "代表图",
        "status": "missing",
        "is_clean": False,
        "source": "",
        "warning": "",
        "evidence_image_count": len(set(evidence_images)),
        "hidden_by_default": False,
        "candidate_image": candidate_preview,
        "candidate_label": candidate_label,
        "candidate_source": candidate_source,
    }

    if kind == "scene" and _mode2_scene_is_mixed_visual_bundle(asset):
        display.update({
            "label": "混合场景候选",
            "status": "mixed_reference_bundle",
            "hidden_by_default": True,
            "warning": str(
                asset.get("source_quality_warning")
                or "这组图属于多个视觉组或不同物理场景，必须先拆分，不能绑定一张目标图或提纯图后直接使用。"
            ),
        })
        return display

    if kind == "scene" and _mode2_scene_is_shot_reference_only(asset) and not _mode2_asset_is_manually_approved(asset):
        display.update({
            "label": "只作镜头参考",
            "status": "shot_reference_only",
            "hidden_by_default": True,
            "warning": str(
                asset.get("source_quality_warning")
                or "这组图是人物近景/动作构图证据，不是干净场景资产；请换空镜、远景或重新提纯。"
            ),
        })
        return display

    if kind == "scene" and _mode2_scene_refinement_is_untrusted(asset):
        display.update({
            "label": "旧提纯结果已隔离",
            "status": "stale_refinement",
            "hidden_by_default": True,
            "warning": str(
                asset.get("refinement_provenance_warning")
                or "提纯结果没有通过源图指纹校验，可能来自旧分析或其他场景，不能作为代表图。"
            ),
        })
        return display

    if target_image:
        display.update({
            "image": target_image,
            "label": "目标替换图" if kind == "role" else "目标参考图",
            "status": "ready",
            "is_clean": True,
            "source": "target_image",
            "warning": "",
        })
        return display

    if kind == "scene":
        if (
            refined_source
            and str(asset.get("refinement_kind") or "") == "clean_background"
            and refinement_provenance_verified
        ):
            display.update({
                "image": refined_source,
                "label": "干净背景代表图" if not _mode2_asset_is_manually_approved(asset) else "人工确认场景图",
                "status": "ready",
                "is_clean": True,
                "source": "refined_source_image",
                "warning": "",
            })
            return display
        if refined_source and str(asset.get("refinement_kind") or "") == "clean_background":
            display.update({
                "label": "旧提纯结果已隔离",
                "status": "stale_refinement",
                "hidden_by_default": True,
                "warning": str(
                    asset.get("refinement_provenance_warning")
                    or "提纯结果没有通过源图指纹校验，可能来自旧分析或其他场景，不能作为代表图。"
                ),
            })
            return display
        if source_image and _mode2_asset_is_manually_approved(asset):
            display.update({
                "image": source_image,
                "label": "人工确认场景图",
                "status": "ready",
                "is_clean": True,
                "source": "source_image",
                "warning": "",
            })
            return display
        if source_image:
            display.update({
                "label": "需补干净场景图",
                "status": "needs_clean_source",
                "warning": str(
                    asset.get("source_quality_warning")
                    or "原片场景帧只是候选证据；未提纯或人工确认前不能作为可用场景代表图。"
                ),
                "hidden_by_default": True,
            })
            return display
        display.update({
            "label": "需补干净场景图",
            "status": "needs_clean_source",
            "warning": str(asset.get("source_quality_warning") or "没有可用的干净场景代表图，请换空镜/远景或重新提纯。"),
            "hidden_by_default": True,
        })
        return display

    if kind == "prop":
        prop_image = refined_cutout or refined_source
        if prop_image and _mode2_prop_refinement_is_clean(asset):
            display.update({
                "image": prop_image,
                "label": "干净物品代表图",
                "status": "ready",
                "is_clean": True,
                "source": "refined_cutout_image" if refined_cutout else "refined_source_image",
                "warning": "",
            })
            return display
        display.update({
            "label": "需换干净物品图",
            "status": "needs_clean_source",
            "warning": str(asset.get("refinement_warning") or asset.get("source_quality_warning") or "没有通过质量检查的物品代表图，请换清晰裁剪图或重新提纯。"),
            "hidden_by_default": True,
        })
        return display

    if kind == "role":
        display.update({
            "label": "需选择目标人物图",
            "status": "needs_target",
            "warning": "角色还没有目标替换图；原片身份图只放在证据详情里。",
        })
        return display

    if source_image:
        display.update({
            "image": source_image,
            "label": "候选代表图",
            "status": "candidate",
            "source": "source_image",
        })
    return display


def _mode2_apply_asset_contract_fields(
    assets: list[dict[str, Any]],
    shots: list[dict[str, Any]],
    mask_summary: dict[str, Any],
) -> None:
    shot_by_id = {
        str(shot.get("segment_id") or ""): shot
        for shot in shots
        if isinstance(shot, dict) and str(shot.get("segment_id") or "")
    }
    mask_by_asset = mask_summary.get("by_asset") if isinstance(mask_summary.get("by_asset"), dict) else {}
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        asset_id = str(asset.get("id") or "").strip()
        kind = str(asset.get("kind") or "").strip()
        if kind == "scene":
            _mode2_update_scene_visual_groups(asset, shots)
            _mode2_apply_scene_refinement_provenance_contract(asset)
        target_image = str(asset.get("target_image") or "").strip()
        scene_mixed = bool(kind == "scene" and _mode2_scene_is_mixed_visual_bundle(asset))
        scene_refinement_untrusted = bool(kind == "scene" and _mode2_scene_refinement_is_untrusted(asset))
        scene_delivery_blocked = bool(scene_mixed or scene_refinement_untrusted)
        used_shots = [
            str(value or "").strip()
            for value in (asset.get("used_shots") or [])
            if str(value or "").strip()
        ]
        time_ranges: list[list[float]] = []
        for shot_id in used_shots:
            shot = shot_by_id.get(shot_id)
            if not shot:
                continue
            try:
                start = round(float(shot.get("start") or 0.0), 3)
                end = round(float(shot.get("end") or start), 3)
            except (TypeError, ValueError):
                continue
            time_ranges.append([start, end])

        evidence_images: list[str] = []
        for value in [
            asset.get("source_image"),
            asset.get("candidate_source_image"),
            asset.get("refined_source_image"),
            asset.get("refined_cutout_image"),
            *(asset.get("source_images") or []),
            *(asset.get("candidate_source_images") or []),
            *(asset.get("identity_evidence_images") or []),
        ]:
            path_text = str(value or "").strip()
            if path_text and path_text not in evidence_images:
                evidence_images.append(path_text)

        warnings = [
            str(value or "").strip()
            for value in [
                asset.get("source_quality_warning"),
                asset.get("refinement_warning"),
                asset.get("asset_quarantine_reason"),
            ]
            if str(value or "").strip()
        ]
        for issue in asset.get("asset_audit_issues") or []:
            if isinstance(issue, dict) and str(issue.get("message") or "").strip():
                warnings.append(str(issue.get("message") or "").strip())
        warnings = list(dict.fromkeys(warnings))

        per_mask = mask_by_asset.get(asset_id) if isinstance(mask_by_asset.get(asset_id), dict) else {}
        mask_status = str(per_mask.get("status") or ("empty" if not per_mask else "unknown"))
        asset["mask_candidate_status"] = mask_status
        asset["mask_candidate_count"] = int(per_mask.get("candidate_count") or 0)
        asset["mask_candidate_object_count"] = int(per_mask.get("object_count") or 0)
        asset["mask_candidates_cached"] = bool(asset["mask_candidate_count"])

        identity_status = str(
            asset.get("identity_status")
            or asset.get("source_identity_status")
            or asset.get("track_status")
            or ("evidence" if evidence_images else "missing")
        )
        asset["source_identity"] = {
            "status": identity_status,
            "kind": kind,
            "evidence_images": evidence_images,
            "mask_candidate_status": mask_status,
            "mask_candidate_count": asset["mask_candidate_count"],
            "mask_candidate_object_count": asset["mask_candidate_object_count"],
            "mask_candidates_cached": asset["mask_candidates_cached"],
            "role_track_id": str(asset.get("role_track_id") or ""),
            "track_dir": str(asset.get("track_dir") or ""),
            "quality": asset.get("track_quality") if isinstance(asset.get("track_quality"), dict) else {},
            "warnings": warnings,
            "seedance_delivery": "never",
        }
        target_status = (
            "blocked_mixed_source"
            if target_image and scene_mixed
            else "blocked_untrusted_scene_source"
            if target_image and scene_refinement_untrusted
            else
            "confirmed"
            if target_image and str(asset.get("manual_asset_status") or "") == "approved"
            else "uploaded"
            if target_image
            else "missing"
        )
        asset["target_asset"] = {
            "status": target_status,
            "active_image": target_image,
            "versions": [target_image] if target_image else [],
            "seedance_delivery": (
                "blocked_until_scene_split"
                if target_image and scene_mixed
                else "blocked_until_scene_refined"
                if target_image and scene_refinement_untrusted
                else "allowed"
                if target_image
                else "blocked_until_uploaded"
            ),
        }
        asset["usage"] = {
            "shot_ids": used_shots,
            "time_ranges": time_ranges,
            "count": len(used_shots),
        }
        asset["seedance_policy"] = {
            "send_source_image": False,
            "send_target_image": bool(target_image and not scene_delivery_blocked),
            "send_mask": False,
            "send_mask_candidates": False,
            "reference_field": "target_image",
            "note": "原片证据、SAM3 mask 和身份轨只用于内部识别；Seedance 只收目标图或显式非 mask 参考。",
        }
        display = _mode2_asset_display_fields(asset, evidence_images)
        asset["asset_display"] = display
        asset["representative_image"] = str(display.get("image") or "")
        asset["representative_label"] = str(display.get("label") or "")
        asset["representative_status"] = str(display.get("status") or "")
        asset["representative_is_clean"] = bool(display.get("is_clean"))
        asset["representative_warning"] = str(display.get("warning") or "")
        asset["evidence_image_count"] = int(display.get("evidence_image_count") or 0)
        asset["asset_hidden_by_default"] = bool(display.get("hidden_by_default"))
        asset["candidate_preview_image"] = str(display.get("candidate_image") or "")
        asset["candidate_preview_label"] = str(display.get("candidate_label") or "")
        asset["candidate_preview_source"] = str(display.get("candidate_source") or "")
        display_status = str(display.get("status") or "")
        display_clean = bool(display.get("is_clean"))
        manually_approved = _mode2_asset_is_manually_approved(asset)
        source_visual_status = display_status or "missing"
        source_trust_level = str(asset.get("source_trust_level") or "").strip()
        visual_confirmed = bool(target_image or (display_clean and display_status == "ready"))
        if scene_mixed:
            visual_confirmed = False
            source_visual_status = "mixed_visual_bundle"
            source_trust_level = "mixed_scene_candidate_only"
        elif kind == "scene" and _mode2_scene_is_shot_reference_only(asset) and not manually_approved:
            visual_confirmed = False
            source_visual_status = "shot_reference_only"
            source_trust_level = "shot_reference_only"
        elif scene_refinement_untrusted:
            visual_confirmed = False
            source_visual_status = "stale_refinement"
            source_trust_level = "stale_refinement_quarantined"
        elif target_image:
            source_visual_status = "target_uploaded"
            source_trust_level = "target_reference"
        elif kind == "prop" and _mode2_prop_is_unconfirmed_candidate(asset) and not manually_approved:
            visual_confirmed = False
            source_visual_status = "unconfirmed_semantic_prop"
            source_trust_level = "semantic_candidate_only"
        elif kind == "scene" and not manually_approved:
            visual_confirmed = False
            if display_status == "needs_clean_source":
                source_visual_status = "unconfirmed_scene_candidate"
                source_trust_level = "scene_candidate_only"
        elif visual_confirmed:
            if kind == "prop":
                source_trust_level = "clean_prop_confirmed"
            elif kind == "scene":
                source_trust_level = "clean_scene_confirmed"
            elif kind == "role":
                source_trust_level = source_trust_level or "target_reference"
            else:
                source_trust_level = source_trust_level or "clean_asset_confirmed"
        elif kind in {"scene", "prop"}:
            source_trust_level = source_trust_level or "unconfirmed_candidate"
        elif kind == "role":
            source_trust_level = source_trust_level or "identity_evidence_only"
        asset["visual_confirmed"] = visual_confirmed
        asset["source_visual_status"] = source_visual_status
        asset["source_trust_level"] = source_trust_level
        delivery_target = target_image if not scene_delivery_blocked else ""
        asset["generation_reference_image"] = delivery_target
        asset["seedance_reference_image"] = delivery_target
        asset["seedance_ready"] = bool(delivery_target)
        asset["asset_hidden_by_default"] = bool(asset["asset_hidden_by_default"] or (
            kind in {"scene", "prop"} and not visual_confirmed and (not target_image or scene_delivery_blocked)
        ))
        asset["scene_candidate_groups"] = asset.get("scene_candidate_groups") if isinstance(asset.get("scene_candidate_groups"), list) else []
        asset["scene_clean_candidate_image"] = str(asset.get("scene_clean_candidate_image") or "")
        asset["scene_shot_reference_image"] = str(asset.get("scene_shot_reference_image") or "")
        asset["scene_mixed_visual_groups"] = bool(asset.get("scene_mixed_visual_groups"))
        asset["scene_visual_group_ids"] = [
            str(value or "").strip()
            for value in (asset.get("scene_visual_group_ids") or [])
            if str(value or "").strip()
        ]
        asset["scene_visual_group_count"] = int(asset.get("scene_visual_group_count") or 0)
        asset["scene_visual_hash_group_count"] = int(asset.get("scene_visual_hash_group_count") or 0)
        asset["scene_visual_groups_unverified"] = bool(asset.get("scene_visual_groups_unverified"))


MODE2_STORYBOARD_PROP_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("粗铁链", ("粗链条", "粗铁链", "铁链", "锁链", "链条")),
    ("车辆", ("车辆", "汽车", "轿车", "车流", "车内")),
    ("床", ("床上", "床边", "床铺", "卧室床", "床")),
)


def _mode2_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mode2_storyboard_duration_from_data(data: dict[str, Any]) -> float:
    duration = _mode2_float(data.get("duration"), 0.0)
    for key in ("shots", "reference_segments", "visual_segments", "semantic_segments", "raw_visual_segments"):
        for item in data.get(key) or []:
            if isinstance(item, dict):
                duration = max(duration, _mode2_float(item.get("end"), 0.0))
    understanding = data.get("understanding") if isinstance(data.get("understanding"), dict) else {}
    for scene in understanding.get("scenes") or []:
        if isinstance(scene, dict):
            duration = max(duration, _mode2_float(scene.get("end"), 0.0))
    return max(0.1, duration)


def _mode2_normalize_semantic_scene_segment(
    item: dict[str, Any],
    *,
    index: int,
    duration: float,
) -> dict[str, Any] | None:
    try:
        start = max(0.0, min(float(item.get("start") or 0.0), duration))
        end = max(start, min(float(item.get("end") or duration), duration))
    except (TypeError, ValueError):
        return None
    if end - start < 0.05:
        return None
    semantic_scene = item.get("semantic_scene") if isinstance(item.get("semantic_scene"), dict) else item
    description = " ".join(
        value
        for value in (
            str(item.get("description") or semantic_scene.get("description") or "").strip(),
            str(item.get("key_action") or semantic_scene.get("key_action") or "").strip(),
        )
        if value
    ).strip()
    if not description:
        return None
    segment_id = str(item.get("segment_id") or "").strip() or f"U{index:03d}"
    characters = _string_list(item.get("characters") or semantic_scene.get("characters"))
    details = [
        detail for detail in (item.get("character_details") or semantic_scene.get("character_details") or [])
        if isinstance(detail, dict)
    ]
    person_count = _safe_int(item.get("person_count"), -1)
    if person_count < 0:
        person_count = len(characters) or len(details) or -1
    return {
        "segment_id": segment_id,
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(end - start, 3),
        "person_count": person_count,
        "segment_type": "semantic_scene",
        "source": str(item.get("source") or "mode2_pre_director"),
        "source_video_path": str(item.get("source_video_path") or semantic_scene.get("source_video_path") or ""),
        "description": description,
        "characters": characters,
        "character_details": details,
        "key_action": str(item.get("key_action") or semantic_scene.get("key_action") or "").strip(),
        "semantic_scene": semantic_scene,
    }


def _mode2_semantic_scene_segments_from_data(data: dict[str, Any]) -> list[dict[str, Any]]:
    duration = _mode2_storyboard_duration_from_data(data)
    raw_segments: list[dict[str, Any]] = []
    for item in data.get("semantic_segments") or []:
        if isinstance(item, dict):
            raw_segments.append(item)
    if not raw_segments:
        understanding = data.get("understanding") if isinstance(data.get("understanding"), dict) else {}
        if understanding.get("scenes"):
            try:
                raw_segments.extend(_storyboard_reference_segments_from_understanding(understanding, duration=duration))
            except Exception:  # noqa: BLE001
                pass
    auto_director = data.get("auto_director") if isinstance(data.get("auto_director"), dict) else {}
    story = auto_director.get("story") if isinstance(auto_director.get("story"), dict) else {}
    for scene in story.get("mode2_understood_scenes") or []:
        if isinstance(scene, dict):
            raw_segments.append(scene)

    result: list[dict[str, Any]] = []
    seen: set[tuple[float, float, str]] = set()
    for index, item in enumerate(raw_segments, 1):
        normalized = _mode2_normalize_semantic_scene_segment(item, index=index, duration=duration)
        if not normalized:
            continue
        key = (
            round(float(normalized["start"]), 2),
            round(float(normalized["end"]), 2),
            str(normalized.get("description") or "")[:80],
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _mode2_semantic_scene_segments_from_sources(
    reference_segments: list[dict[str, Any]],
    *,
    auto_director_plan: dict[str, Any] | None = None,
    understanding: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    data = {
        "reference_segments": reference_segments,
        "semantic_segments": [],
        "understanding": understanding if isinstance(understanding, dict) else {},
        "auto_director": auto_director_plan if isinstance(auto_director_plan, dict) else {},
    }
    return _mode2_semantic_scene_segments_from_data(data)


def _mode2_scene_asset_name(segment: dict[str, Any], index: int) -> str:
    scene = segment.get("semantic_scene") if isinstance(segment.get("semantic_scene"), dict) else {}
    scene_type = str(scene.get("scene_type") or segment.get("scene_type") or "").strip()
    description = str(segment.get("description") or "").strip()
    probes = [
        ("囚禁房间", ("囚禁", "铁链", "锁住", "刑具", "封闭房间")),
        ("卧室", ("卧室", "床上", "床边", "床")),
        ("城市道路", ("城市道路", "车流", "道路")),
        ("海边城市", ("海边", "岛国", "城市黄昏")),
        ("室内", ("室内", "墙面", "房间")),
        ("庭院", ("庭院", "院子", "回廊")),
    ]
    for name, keywords in probes:
        if any(keyword in description for keyword in keywords):
            return name
    return scene_type or f"场景{index}"


def _mode2_make_scene_asset(
    *,
    index: int,
    video_path: str,
    segment: dict[str, Any],
    source_segment_ids: list[str] | None = None,
) -> dict[str, Any]:
    description = str(segment.get("description") or "").strip()
    source_ids = [
        str(value or "").strip()
        for value in (source_segment_ids or [segment.get("segment_id")])
        if str(value or "").strip()
    ]
    return {
        "id": f"scene_{index}",
        "kind": "scene",
        "name": _mode2_scene_asset_name(segment, index),
        "tag": "语义场景",
        "source_video_path": video_path,
        "source_time": round(max(0.1, _mode2_float(segment.get("start"), 0.0) + 0.2), 3),
        "source_segment_ids": source_ids,
        "source_time_range": [
            round(_mode2_float(segment.get("start"), 0.0), 3),
            round(_mode2_float(segment.get("end"), _mode2_float(segment.get("start"), 0.0)), 3),
        ],
        "target_image": "",
        "prompt": (
            "写实短剧场景参考图，只描述环境、空间关系、光影氛围和镜头可用性；"
            "不要把人物脸部特写或剧情动作当成场景资产。"
            + (f" 原片场景：{description}" if description else "")
        ),
        "status": "pending",
        "selection_reason": (
            f"来自全片理解语义场景 {source_ids[0]}，时间 {segment.get('start')}s-{segment.get('end')}s。"
            if source_ids
            else "来自全片理解语义场景。"
        ),
    }


def _mode2_recommendation_text_parts(item: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    for key in (
        "name",
        "tag",
        "prompt",
        "selection_reason",
        "manual_note",
        "source_quality_warning",
        "refinement_prompt",
        "refinement_kind",
        "source_visual_status",
        "source_trust_level",
        "segment_id",
        "video_path",
    ):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    for key in ("source_segment_ids", "scene_visual_group_ids", "used_shots", "evidence_shots"):
        values = item.get(key)
        if isinstance(values, list):
            parts.extend(str(value).strip() for value in values if str(value).strip())
    return parts


MODE2_BACKGROUND_RECOMMENDATION_STOP_TERMS = {
    "scene",
    "asset",
    "image",
    "video",
    "frame",
    "prompt",
    "reference",
    "candidate",
    "通用",
    "场景",
    "资产",
    "参考",
    "参考图",
    "候选",
    "关键帧",
    "写实",
    "短剧",
    "镜头",
    "光影",
    "气氛",
    "可用",
    "可用性",
    "人物",
    "脸部",
    "特写",
    "剧情",
    "动作",
    "环境",
    "描述",
    "不要",
}


def _mode2_recommendation_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for raw in re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+", str(text or "").lower()):
        if len(raw) < 2:
            continue
        if raw in MODE2_BACKGROUND_RECOMMENDATION_STOP_TERMS:
            continue
        if not re.fullmatch(r"[\u4e00-\u9fff]+", raw) or len(raw) <= 8:
            terms.add(raw)
        if re.fullmatch(r"[\u4e00-\u9fff]+", raw):
            max_size = min(4, len(raw))
            for size in range(2, max_size + 1):
                for index in range(0, len(raw) - size + 1):
                    term = raw[index:index + size]
                    if term not in MODE2_BACKGROUND_RECOMMENDATION_STOP_TERMS:
                        terms.add(term)
    return {term for term in terms if len(term) >= 2}


def _mode2_background_candidate_path(asset: dict[str, Any]) -> str:
    for key in (
        "target_image",
        "generation_reference_image",
        "seedance_reference_image",
        "refined_source_image",
        "scene_clean_candidate_image",
        "scene_shot_reference_image",
        "representative_image",
        "asset_path",
    ):
        value = str(asset.get(key) or "").strip()
        if value and Path(value).exists():
            return value
    return ""


def _mode2_background_candidate_reason(candidate: dict[str, Any], matched_terms: list[str]) -> str:
    source = str(candidate.get("source") or "")
    asset = candidate.get("asset") if isinstance(candidate.get("asset"), dict) else {}
    reasons: list[str] = []
    if source == "background_config":
        reasons.append("已在背景配置中使用过")
    if str(asset.get("refinement_kind") or "") == "clean_background":
        reasons.append("干净背景提纯图")
    if str(asset.get("manual_asset_status") or "") == "approved":
        reasons.append("人工确认可用")
    if str(asset.get("manual_asset_status") or "") == "shot_reference":
        reasons.append("镜头参考")
    if matched_terms:
        reasons.append("匹配: " + "、".join(matched_terms[:5]))
    return "；".join(reasons) or "场景候选"


def _mode2_background_candidate_score(
    query_text: str,
    query_terms: set[str],
    candidate: dict[str, Any],
    current_source_ids: set[str],
) -> tuple[float, list[str]]:
    asset = candidate.get("asset") if isinstance(candidate.get("asset"), dict) else {}
    text = " ".join(_mode2_recommendation_text_parts(asset) + [str(candidate.get("label") or ""), str(candidate.get("path") or "")])
    lowered = text.lower()
    matched_terms = sorted(
        {term for term in query_terms if len(term) >= 2 and term in lowered},
        key=lambda value: (-len(value), value),
    )
    score = 0.0
    for term in matched_terms[:20]:
        score += 1.0 + min(2.0, len(term) / 4.0)
    source_ids = {
        str(value or "").strip()
        for value in (asset.get("source_segment_ids") or [])
        if str(value or "").strip()
    }
    if current_source_ids and source_ids and current_source_ids & source_ids:
        score += 4.0
    if candidate.get("source") == "background_config":
        score += 3.0
    if str(asset.get("kind") or "") == "scene":
        score += 1.5
    if str(asset.get("refinement_kind") or "") == "clean_background":
        score += 4.0
    if str(asset.get("manual_asset_status") or "") == "approved":
        score += 3.0
    elif str(asset.get("manual_asset_status") or "") == "shot_reference":
        score += 1.0
    if str(asset.get("target_image") or "").strip():
        score += 2.5
    if str(asset.get("source_quality_status") or "") in {"scene_person_heavy", "scene_refinement_provenance_invalid"}:
        score -= 2.5
    if candidate.get("path") and str(candidate.get("path")).lower() in query_text.lower():
        score += 5.0
    return score, matched_terms


def _mode2_recommend_background_candidates(
    project_dir: str,
    *,
    asset_id: str = "",
    query_text: str = "",
    limit: int = 6,
) -> dict[str, Any]:
    root = _resolve_storyboard_mode2_project_dir(project_dir)
    data = _load_storyboard_mode2_store(root)
    assets = [item for item in (data.get("assets") or []) if isinstance(item, dict)]
    source_asset = next((item for item in assets if str(item.get("id") or "") == asset_id), None)
    if source_asset is None and asset_id:
        raise ValueError(f"asset_not_found: {asset_id}")
    if not query_text and source_asset is not None:
        query_text = " ".join(_mode2_recommendation_text_parts(source_asset))
    if not query_text:
        query_text = " ".join(
            str(value or "")
            for item in assets
            for value in (item.get("name"), item.get("prompt"), item.get("selection_reason"))
            if str(value or "").strip()
        )
    query_terms = _mode2_recommendation_terms(query_text)
    current_source_ids = {
        str(value or "").strip()
        for value in ((source_asset or {}).get("source_segment_ids") or [])
        if str(value or "").strip()
    }

    candidates: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for asset in assets:
        if str(asset.get("id") or "") == asset_id:
            continue
        if str(asset.get("kind") or "") != "scene":
            continue
        path = _mode2_background_candidate_path(asset)
        if not path:
            continue
        path_key = str(Path(path)).lower()
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        candidates.append({
            "source": "scene_asset",
            "asset_id": str(asset.get("id") or ""),
            "label": str(asset.get("name") or Path(path).stem),
            "path": path,
            "asset": asset,
        })

    try:
        background_store = load_asset_store(root).get("backgrounds") or {}
    except Exception:
        background_store = {}
    if isinstance(background_store, dict):
        for key, item in background_store.items():
            if not isinstance(item, dict):
                continue
            path = str(item.get("asset_path") or "").strip()
            if not path or not Path(path).exists():
                continue
            path_key = str(Path(path)).lower()
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            asset = {
                "id": str(key),
                "kind": "scene",
                "name": str(item.get("name") or item.get("segment_id") or Path(path).stem),
                "prompt": " ".join(str(item.get(field) or "") for field in ("mode", "fit_mode", "segment_id", "video_path")),
                "target_image": path,
                "manual_asset_status": "approved",
                "source_segment_ids": [str(item.get("segment_id") or "")],
                "asset_path": path,
            }
            candidates.append({
                "source": "background_config",
                "config_id": str(key),
                "label": str(asset["name"]),
                "path": path,
                "asset": asset,
            })

    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        score, matched_terms = _mode2_background_candidate_score(
            query_text,
            query_terms,
            candidate,
            current_source_ids,
        )
        if score <= 0:
            continue
        scored.append({
            "source": candidate["source"],
            "asset_id": candidate.get("asset_id", ""),
            "config_id": candidate.get("config_id", ""),
            "label": candidate["label"],
            "asset_path": candidate["path"],
            "score": round(score, 3),
            "matched_terms": matched_terms[:8],
            "reason": _mode2_background_candidate_reason(candidate, matched_terms),
        })
    scored.sort(key=lambda item: (-float(item["score"]), item["label"]))
    return {
        "project_dir": str(root),
        "asset_id": asset_id,
        "query": query_text,
        "recommendations": scored[:limit],
        "candidates": scored[:limit],
        "candidate_count": len(candidates),
        "mode": "semi_auto_confirm_required",
    }


def _mode2_make_prop_asset(
    *,
    index: int,
    video_path: str,
    prop_name: str,
    source_time: float,
    source_segment_ids: list[str],
    keywords: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "id": f"prop_{index}",
        "kind": "prop",
        "name": prop_name,
        "tag": "物品",
        "source_video_path": video_path,
        "source_time": round(max(0.1, source_time), 3),
        "source_segment_ids": source_segment_ids,
        "target_image": "",
        "prompt": f"写实{prop_name}道具参考图，材质清晰，适合短剧镜头内复用。",
        "status": "pending",
        "source_quality_status": "prop_needs_visual_check",
        "source_usage_role": "object_evidence_candidate",
        "source_selection_method": "semantic_keyword_candidate",
        "track_status": "needs_manual_review",
        "selection_reason": (
            f"从包含“{'/'.join(keywords[:3])}”的语义段抽取候选帧；仍需人工确认道具本体清晰可见。"
            if source_segment_ids
            else "只从文本中识别到道具关键词，未定位到具体语义段，需人工补图。"
        ),
    }


def _mode2_next_asset_index(assets: list[dict[str, Any]], kind: str) -> int:
    prefix = f"{kind}_"
    value = 0
    for asset in assets:
        asset_id = str(asset.get("id") or "")
        if asset_id.startswith(prefix):
            value = max(value, _storyboard_asset_index(asset_id))
    return value + 1


def _mode2_existing_refined_file(directory: Path, patterns: tuple[str, ...]) -> str:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(
            path
            for path in directory.glob(pattern)
            if path.is_file() and path.stat().st_size > 0
        )
    if not candidates:
        return ""
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return str(candidates[0])


def _mode2_attach_existing_refined_outputs(root: Path, asset: dict[str, Any]) -> bool:
    kind = str(asset.get("kind") or "")
    if kind not in {"scene", "prop"}:
        return False
    asset_id = str(asset.get("id") or asset.get("name") or "").strip()
    if not asset_id:
        return False
    refined_dir = root / "assets" / "refined" / _mode2_safe_asset_slug(asset_id)
    if not refined_dir.exists() or not refined_dir.is_dir():
        return False

    changed = False

    def update(key: str, value: Any) -> None:
        nonlocal changed
        if asset.get(key) != value:
            asset[key] = value
            changed = True

    sidecars = sorted(
        [path for path in refined_dir.glob("*_provenance.json") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not sidecars:
        has_legacy_outputs = any(
            path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            for path in refined_dir.iterdir()
        )
        if has_legacy_outputs:
            update("refinement_provenance_status", "missing_provenance")
            update(
                "refinement_provenance_warning",
                "发现旧提纯文件，但没有 provenance sidecar/source fingerprint；为防止 asset_id 重用串图，未自动附着。",
            )
            update("refinement_status", "quarantined")
            update("refinement_quarantined", True)
        return changed

    invalid_status = "missing_provenance"
    invalid_warning = "没有可验证的提纯 provenance。"
    for sidecar in sidecars:
        try:
            provenance = json.loads(sidecar.read_text(encoding="utf-8-sig"))
        except Exception as exc:  # noqa: BLE001
            invalid_status = "invalid_sidecar"
            invalid_warning = f"提纯 provenance sidecar 无法读取: {exc}"
            continue
        if not isinstance(provenance, dict):
            invalid_status = "invalid_sidecar"
            invalid_warning = "提纯 provenance sidecar 不是有效对象。"
            continue
        output_path = _mode2_refinement_output_path(provenance, "refined_source_image")
        try:
            if not output_path or refined_dir.resolve() not in Path(output_path).resolve().parents:
                invalid_status = "output_outside_asset_dir"
                invalid_warning = "提纯 provenance 输出不在当前资产目录，疑似跨资产串图。"
                continue
        except Exception:  # noqa: BLE001
            invalid_status = "output_path_invalid"
            invalid_warning = "提纯 provenance 输出路径无效。"
            continue
        probe = dict(asset)
        probe.pop("refined_source_image", None)
        verified, status, warning = _mode2_validate_refinement_provenance(probe, provenance)
        if not verified:
            invalid_status = status
            invalid_warning = warning
            continue

        outputs = provenance.get("outputs") if isinstance(provenance.get("outputs"), dict) else {}
        source = provenance.get("source") if isinstance(provenance.get("source"), dict) else {}
        update("refinement_provenance", provenance)
        update("refinement_provenance_path", str(sidecar))
        update("refinement_provenance_status", "verified")
        update("refinement_provenance_verified", True)
        update("refinement_provenance_warning", "")
        update("refinement_source_path", str(source.get("path") or ""))
        update("refinement_source_hash", str(source.get("hash") or ""))
        update("refinement_source_fingerprint", str(source.get("fingerprint") or ""))
        update("refined_source_image", output_path)
        update("refined_source_images", [output_path])
        update("refined_mask_image", _mode2_refinement_output_path(provenance, "refined_mask_image"))
        update("refined_segment_image", _mode2_refinement_output_path(provenance, "refined_segment_image"))
        if kind == "scene":
            update("refined_cutout_image", "")
            update("refinement_kind", "clean_background")
        else:
            update("refined_cutout_image", _mode2_refinement_output_path(provenance, "refined_cutout_image") or output_path)
            update("refinement_kind", "prop_cutout")
            mask_path = str(asset.get("refined_mask_image") or "").strip()
            if mask_path:
                update("refinement_quality", _mode2_mask_quality(mask_path))
        update("refinement_method", str(provenance.get("method") or "verified_existing_output"))
        update("refinement_status", "ready")
        update("refinement_quarantined", False)
        return changed

    update("refinement_provenance_status", invalid_status)
    update("refinement_provenance_verified", False)
    update("refinement_provenance_warning", invalid_warning)
    update("refinement_status", "quarantined")
    update("refinement_quarantined", True)
    return changed


def _mode2_backfill_semantic_scene_prop_assets(root: Path, data: dict[str, Any]) -> bool:
    assets = [item for item in (data.get("assets") or []) if isinstance(item, dict)]
    reference_segments = [item for item in (data.get("reference_segments") or []) if isinstance(item, dict)]
    semantic_segments = _mode2_semantic_scene_segments_from_data(data)
    video_path = str(data.get("video_path") or "")
    changed = False

    existing_scene_source_ids = {
        str(value or "").strip()
        for asset in assets
        if str(asset.get("kind") or "") == "scene"
        for value in (asset.get("source_segment_ids") or [])
        if str(value or "").strip()
    }
    existing_scene_descriptions = {
        str(asset.get("semantic_description") or asset.get("selection_reason") or "")[:120]
        for asset in assets
        if str(asset.get("kind") or "") == "scene"
    }
    next_scene_index = _mode2_next_asset_index(assets, "scene")
    for segment in semantic_segments:
        segment_id = str(segment.get("segment_id") or "").strip()
        description_key = str(segment.get("description") or "")[:120]
        if segment_id and segment_id in existing_scene_source_ids:
            continue
        if description_key and description_key in existing_scene_descriptions:
            continue
        if len([asset for asset in assets if str(asset.get("kind") or "") == "scene"]) >= 8:
            break
        asset = _mode2_make_scene_asset(
            index=next_scene_index,
            video_path=video_path,
            segment=segment,
            source_segment_ids=[segment_id] if segment_id else [],
        )
        asset["semantic_description"] = str(segment.get("description") or "")
        assets.append(asset)
        existing_scene_source_ids.update(asset["source_segment_ids"])
        existing_scene_descriptions.add(description_key)
        next_scene_index += 1
        changed = True

    source_segments = [*reference_segments, *semantic_segments]
    existing_prop_names = {
        str(asset.get("name") or "").strip()
        for asset in assets
        if str(asset.get("kind") or "") == "prop"
    }
    next_prop_index = _mode2_next_asset_index(assets, "prop")
    for prop_name, keywords in MODE2_STORYBOARD_PROP_KEYWORDS:
        if prop_name in existing_prop_names:
            continue
        matched_ids: list[str] = []
        matched_times: list[float] = []
        for segment in source_segments:
            text = " ".join([
                str(segment.get("description") or ""),
                str(segment.get("key_action") or ""),
            ])
            if not any(keyword in text for keyword in keywords):
                continue
            segment_id = str(segment.get("segment_id") or "").strip()
            if segment_id and segment_id not in matched_ids:
                matched_ids.append(segment_id)
            matched_times.append(_mode2_float(segment.get("start"), 0.0) + 0.2)
        if not matched_ids and not matched_times:
            continue
        assets.append(_mode2_make_prop_asset(
            index=next_prop_index,
            video_path=video_path,
            prop_name=prop_name,
            source_time=min(matched_times) if matched_times else 0.1,
            source_segment_ids=matched_ids,
            keywords=keywords,
        ))
        existing_prop_names.add(prop_name)
        next_prop_index += 1
        changed = True

    for asset in assets:
        if _mode2_attach_existing_refined_outputs(root, asset):
            changed = True

    if changed:
        data["assets"] = assets
        data["asset_stage"] = "semantic_asset_manifest_backfilled"
    return changed


def _mode2_metadata_stages(data: dict[str, Any]) -> dict[str, Any]:
    understanding = data.get("understanding") if isinstance(data.get("understanding"), dict) else {}
    return {
        "analysis_original_video": {
            "status": str(understanding.get("status") or "unknown"),
            "source": str(understanding.get("source") or "mode2_pre_director"),
            "cache_path": str(understanding.get("cache_path") or ""),
        },
        "timeline": str(data.get("timeline_stage") or "visual_timeline_built"),
        "asset_library": str(data.get("asset_stage") or "candidate_after_timeline"),
        "storyboard": str(data.get("storyboard_stage") or "draft_shots_built"),
        "mask_candidates": "offline_cache_only",
        "seedance_delivery": "target/reference images only; mask-like paths are filtered server-side",
        "mode_boundary": str(data.get("mode_boundary") or "Mode2 only; no Mode1 transfer/render dependency."),
    }


def _mode2_structured_original_video_analysis(root: Path, data: dict[str, Any]) -> dict[str, Any]:
    assets = [item for item in (data.get("assets") or []) if isinstance(item, dict)]
    shots = [item for item in (data.get("shots") or []) if isinstance(item, dict)]
    asset_layers = data.get("asset_layers") if isinstance(data.get("asset_layers"), dict) else _mode2_asset_layers_summary(assets)
    mask_summary = (
        data.get("mask_candidate_summary")
        if isinstance(data.get("mask_candidate_summary"), dict)
        else _mode2_mask_candidate_summary(root, data)
    )
    shot_groups = _mode2_shot_asset_groups(shots, assets)
    metadata_stages = data.get("metadata_stages") if isinstance(data.get("metadata_stages"), dict) else _mode2_metadata_stages(data)
    return {
        "source_video_path": str(data.get("video_path") or ""),
        "asset_library": {
            "schema_version": MODE2_ASSET_LIBRARY_SCHEMA_VERSION,
            "assets": assets,
            "asset_layers": asset_layers,
            "mask_candidate_summary": mask_summary,
            "seedance_policy": "Only target_image/explicit non-mask references may be sent to Seedance; masks stay internal.",
        },
        "storyboard": {
            "shots": shots,
            "shot_asset_groups": shot_groups,
            "reference_segments": data.get("reference_segments") or [],
            "visual_segments": data.get("visual_segments") or [],
            "semantic_segments": data.get("semantic_segments") or [],
        },
        "metadata_stages": metadata_stages,
    }


def _refresh_mode2_structured_fields(root: Path, data: dict[str, Any]) -> dict[str, Any]:
    assets = [item for item in (data.get("assets") or []) if isinstance(item, dict)]
    shots = [item for item in (data.get("shots") or []) if isinstance(item, dict)]
    data["assets"] = assets
    data["shots"] = shots
    _mode2_ensure_shot_preview_clips(root, data)
    _mode2_backfill_semantic_scene_prop_assets(root, data)
    _mode2_dedupe_scene_assets(data)
    assets = [item for item in (data.get("assets") or []) if isinstance(item, dict)]
    shots = [item for item in (data.get("shots") or []) if isinstance(item, dict)]
    _attach_storyboard_asset_usage(assets, shots)
    project_config = _normalize_storyboard_project_config(data.get("project_config") or {})
    _compile_storyboard_prompts(assets, shots, project_config=project_config)
    mask_summary = _mode2_mask_candidate_summary(root, data)
    _mode2_apply_asset_contract_fields(assets, shots, mask_summary)
    data["asset_layers"] = _mode2_asset_layers_summary(assets)
    data["mask_candidate_summary"] = mask_summary
    data["shot_asset_groups"] = _mode2_shot_asset_groups(shots, assets)
    data["metadata_stages"] = _mode2_metadata_stages(data)
    data["original_video_analysis"] = _mode2_structured_original_video_analysis(root, data)
    data["assets"] = assets
    data["shots"] = shots
    return data


def _mode2_shot_preview_clip_dir(root: Path) -> Path:
    return root / "clips" / "mode2_shots"


def _mode2_shot_preview_clip_path(root: Path, shot: dict[str, Any]) -> Path:
    shot_id = str(shot.get("segment_id") or "shot").strip() or "shot"
    safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in shot_id)[:48] or "shot"
    start_ms = max(0, int(round(_mode2_float(shot.get("start"), 0.0) * 1000)))
    end_ms = max(start_ms + 10, int(round(_mode2_float(shot.get("end"), 0.0) * 1000)))
    return _mode2_shot_preview_clip_dir(root) / f"{safe_id}_{start_ms:08d}_{end_ms:08d}.mp4"


def _mode2_ensure_shot_preview_clips(root: Path, data: dict[str, Any]) -> bool:
    video_path = str(data.get("video_path") or (data.get("meta") or {}).get("source_path") or "").strip()
    if not video_path:
        return False
    source = Path(video_path)
    if not source.exists() or not source.is_file():
        return False
    shots = [item for item in (data.get("shots") or []) if isinstance(item, dict)]
    if not shots:
        return False

    changed = False
    clips_dir = _mode2_shot_preview_clip_dir(root)
    for shot in shots:
        start = max(0.0, _mode2_float(shot.get("start"), 0.0))
        end = _mode2_float(shot.get("end"), start + _mode2_float(shot.get("duration"), 0.0))
        if end <= start + 0.05:
            continue
        clip_path = _mode2_shot_preview_clip_path(root, shot)
        if not (clip_path.exists() and clip_path.stat().st_size > 0):
            try:
                from spvideo.ffmpeg_tools import cut_segment

                cut_segment(source, start, end, clip_path)
            except Exception as exc:  # noqa: BLE001
                logging.warning(
                    "Mode2 shot preview clip skipped for %s %.3f-%.3f: %s",
                    shot.get("segment_id"),
                    start,
                    end,
                    exc,
                )
                continue
        if not (clip_path.exists() and clip_path.stat().st_size > 0):
            continue
        clip_text = str(clip_path)
        updates = {
            "full_source_video_path": video_path,
            "source_video_path": video_path,
            "preview_clip_path": clip_text,
            "clip_output_path": clip_text,
            "output_path": clip_text,
        }
        for key, value in updates.items():
            if shot.get(key) != value:
                shot[key] = value
                changed = True
    if changed:
        data["clips_dir"] = str(clips_dir)
    elif clips_dir.exists():
        data.setdefault("clips_dir", str(clips_dir))
    return changed


def _load_mode2_storyboard_result(root: Path, store_path: Path) -> dict[str, Any]:
    data = json.loads(store_path.read_text(encoding="utf-8-sig"))
    original_assets_signature = json.dumps(
        {
            "assets": data.get("assets") or [],
            "shots": data.get("shots") or [],
            "clips_dir": data.get("clips_dir") or "",
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    _refresh_mode2_structured_fields(root, data)
    refreshed_assets_signature = json.dumps(
        {
            "assets": data.get("assets") or [],
            "shots": data.get("shots") or [],
            "clips_dir": data.get("clips_dir") or "",
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    if refreshed_assets_signature != original_assets_signature:
        try:
            store_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logging.warning("Mode2 storyboard asset store migration write skipped: %s", exc)
    shots = [
        item for item in (data.get("shots") or [])
        if isinstance(item, dict)
    ]
    understanding = data.get("understanding") if isinstance(data.get("understanding"), dict) else {}
    project_config = _normalize_storyboard_project_config(data.get("project_config") or {})
    reference_strategy = _normalize_storyboard_reference_strategy(data.get("reference_strategy"))
    video_path = str(data.get("video_path") or "")
    summary = (
        "Mode2 storyboard draft loaded from cache. It uses visual shot/person timeline "
        "as the motion skeleton and keeps semantic understanding as labels."
    )
    return {
        "project_dir": str(root),
        "clips_dir": str(data.get("clips_dir") or _mode2_shot_preview_clip_dir(root)),
        "report_path": "",
        "segment_count": len(shots),
        "background_count": sum(
            1
            for item in shots
            if _safe_int(item.get("person_count"), -1) == 0
        ),
        "visual_merge": {},
        "sam3_finalize": None,
        "segments": shots,
        "frames": [],
        "meta": {
            "source_path": video_path,
            "project_config": project_config,
            "mode2_project_dir": str(root),
        },
        "auto_director": data.get("auto_director") if isinstance(data.get("auto_director"), dict) else {},
        "assets": data.get("assets") or [],
        "asset_audit": data.get("last_asset_audit") if isinstance(data.get("last_asset_audit"), dict) else {},
        "asset_organize": data.get("last_asset_organize") if isinstance(data.get("last_asset_organize"), dict) else {},
        "reference_segments": data.get("reference_segments") or [],
        "visual_segments": data.get("visual_segments") or [],
        "raw_visual_segments": data.get("raw_visual_segments") or data.get("visual_segments") or [],
        "semantic_segments": data.get("semantic_segments") or [],
        "sam3_identity_snapshots": data.get("sam3_identity_snapshots") or [],
        "sam3_boundary_hints": data.get("sam3_boundary_hints") or [],
        "asset_stage": str(data.get("asset_stage") or "candidate_after_timeline"),
        "timeline_stage": str(data.get("timeline_stage") or "visual_timeline_built"),
        "storyboard_stage": str(data.get("storyboard_stage") or "draft_shots_built"),
        "metadata_stages": data.get("metadata_stages") or {},
        "asset_layers": data.get("asset_layers") or {},
        "mask_candidate_summary": data.get("mask_candidate_summary") or {},
        "shot_asset_groups": data.get("shot_asset_groups") or [],
        "original_video_analysis": data.get("original_video_analysis") or {},
        "mode_boundary": str(data.get("mode_boundary") or "Mode2 only; no Mode1 transfer/render dependency."),
        "reference_strategy": reference_strategy,
        "project_config": project_config,
        "refinement": data.get("last_refinement") if isinstance(data.get("last_refinement"), dict) else {},
        "storyboard": {
            "status": "draft",
            "source": "storyboard_draft_cache",
            "summary": summary,
            "reference_strategy": reference_strategy,
            "project_config": project_config,
            "understanding_status": str(understanding.get("status") or "unknown"),
            "understanding": understanding,
            "sam3_identity_snapshots": data.get("sam3_identity_snapshots") or [],
            "sam3_boundary_hints": data.get("sam3_boundary_hints") or [],
            "cut_rule": (
                "Mode2 uses visual shot/person timeline as the motion skeleton; cached SAM3 "
                "identity candidates and semantics are correction clues, not generation inputs."
            ),
            "asset_stage": str(data.get("asset_stage") or "candidate_after_timeline"),
            "asset_store_path": str(store_path),
        },
    }


def _storyboard_mode2_asset_store_path(root: Path) -> Path:
    return root / "assets" / "storyboard_assets.json"


def _resolve_storyboard_mode2_project_dir(project_dir: str | Path) -> Path:
    value = str(project_dir or "").strip()
    if not value:
        raise ValueError("missing_project_dir")
    path = Path(value)
    if path.is_file() and path.name == "storyboard_assets.json":
        candidate = path.parent.parent
        if _storyboard_mode2_asset_store_path(candidate).exists():
            return candidate
    if path.is_file():
        path = path.parent

    candidates = [path, *list(path.parents)[:8]]
    for candidate in candidates:
        if candidate.name == "assets" and (candidate / "storyboard_assets.json").exists():
            return candidate.parent
        if _storyboard_mode2_asset_store_path(candidate).exists():
            return candidate

    try:
        root = resolve_auto_director_project_root(path)
        if _storyboard_mode2_asset_store_path(root).exists():
            return root
    except Exception:  # noqa: BLE001
        pass

    if path.exists() and path.is_dir():
        return path
    raise ValueError(f"mode2_project_dir_not_found: {value}")


def _load_storyboard_mode2_store(root: Path) -> dict[str, Any]:
    store_path = _storyboard_mode2_asset_store_path(root)
    if not store_path.exists():
        raise ValueError(f"mode2 storyboard asset store not found: {store_path}")
    data = json.loads(store_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("mode2 storyboard asset store is invalid")
    _refresh_mode2_structured_fields(root, data)
    return data


def _write_storyboard_mode2_store(root: Path, data: dict[str, Any]) -> None:
    store_path = _storyboard_mode2_asset_store_path(root)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _mode2_scene_group_shot_bindings(
    group: dict[str, Any],
    shots: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[float]]:
    frame_times = sorted({
        round(_mode2_scene_frame_time(frame), 3)
        for frame in (group.get("frames") or [])
        if isinstance(frame, dict)
    })
    shot_ids: list[str] = []
    source_segment_ids: list[str] = []
    for shot in shots:
        shot_id = str(shot.get("segment_id") or "").strip()
        if not shot_id or not any(_mode2_scene_time_matches_shot(value, shot) for value in frame_times):
            continue
        shot_ids.append(shot_id)
        source_segment_ids.extend(_string_list(shot.get("source_segment_ids")))
    for frame in group.get("frames") or []:
        if not isinstance(frame, dict):
            continue
        source_segment_ids.extend(_string_list(frame.get("source_segment_ids")))
        source_segment_ids.extend(_string_list(frame.get("source_segment_id")))
    return (
        list(dict.fromkeys(shot_ids)),
        list(dict.fromkeys(source_segment_ids)),
        frame_times,
    )


def _mode2_make_split_scene_child(
    parent: dict[str, Any],
    group: dict[str, Any],
    *,
    child_id: str,
    child_index: int,
    shots: list[dict[str, Any]],
) -> dict[str, Any]:
    frames = [
        copy.deepcopy(frame) for frame in (group.get("frames") or [])
        if isinstance(frame, dict)
    ]
    best_image = str(group.get("best_image") or _mode2_scene_group_best_image(frames)).strip()
    if not frames or not best_image:
        raise ValueError("scene_visual_group_has_no_usable_frames")
    shot_ids, source_segment_ids, frame_times = _mode2_scene_group_shot_bindings(group, shots)
    if shot_ids:
        matched_shots = [shot for shot in shots if str(shot.get("segment_id") or "") in set(shot_ids)]
        time_range = [
            round(min(_mode2_float(shot.get("start"), frame_times[0]) for shot in matched_shots), 3),
            round(max(_mode2_float(shot.get("end"), frame_times[-1]) for shot in matched_shots), 3),
        ]
    else:
        time_range = [round(frame_times[0], 3), round(frame_times[-1], 3)]
    parent_id = str(parent.get("id") or "").strip()
    parent_name = str(parent.get("name") or parent_id or "Scene").strip()
    return {
        "id": child_id,
        "kind": "scene",
        "name": f"{parent_name} - {child_index}",
        "tag": str(parent.get("tag") or "scene"),
        "source_video_path": str(parent.get("source_video_path") or ""),
        "source_image": best_image,
        "source_images": list(dict.fromkeys(
            str(frame.get("path") or frame.get("crop_path") or "").strip()
            for frame in frames
            if str(frame.get("path") or frame.get("crop_path") or "").strip()
        )),
        "keyframes": frames,
        "source_time": round(frame_times[0], 3),
        "source_time_range": time_range,
        "source_segment_ids": source_segment_ids,
        "used_shots": shot_ids,
        "split_assigned_shot_ids": shot_ids,
        "usage_count": len(shot_ids),
        "target_image": "",
        "status": "pending",
        "manual_asset_status": "pending",
        "prompt": str(parent.get("prompt") or ""),
        "semantic_description": str(parent.get("semantic_description") or ""),
        "selection_reason": f"Split from {parent_id} visual group {group.get('id') or child_index}.",
        "source_selection_method": "scene_visual_split",
        "split_parent_asset_id": parent_id,
        "split_group_id": str(group.get("id") or child_index),
        "split_group_method": str(group.get("method") or "dhash"),
        "scene_mixed_visual_groups": False,
        "scene_visual_group_ids": [str(group.get("id") or child_index)],
        "scene_visual_group_count": 1,
        "scene_visual_hash_group_count": 1,
        "scene_visual_groups_unverified": False,
    }


def _mode2_split_scene_asset_in_data(data: dict[str, Any], asset_id: str) -> dict[str, Any]:
    assets = [item for item in (data.get("assets") or []) if isinstance(item, dict)]
    shots = [item for item in (data.get("shots") or []) if isinstance(item, dict)]
    parent = next((asset for asset in assets if str(asset.get("id") or "") == asset_id), None)
    if parent is None:
        raise ValueError(f"asset_not_found: {asset_id}")
    if str(parent.get("kind") or "") != "scene":
        raise ValueError(f"asset_not_scene: {asset_id}")
    if _mode2_scene_asset_is_superseded(parent):
        raise ValueError(f"asset_already_superseded: {asset_id}")
    if not _mode2_scene_is_mixed_visual_bundle(parent):
        raise ValueError(f"scene_not_mixed: {asset_id}")

    groups = _mode2_scene_visual_frame_groups(parent)
    if len(groups) < 2:
        raise ValueError(f"scene_visual_groups_not_splittable: {asset_id}")

    next_index = _mode2_next_asset_index(assets, "scene")
    children: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups, start=1):
        child_id = f"scene_{next_index}"
        next_index += 1
        children.append(_mode2_make_split_scene_child(
            parent,
            group,
            child_id=child_id,
            child_index=group_index,
            shots=shots,
        ))
    if len(children) < 2:
        raise ValueError(f"scene_visual_groups_not_splittable: {asset_id}")

    child_ids = [str(child.get("id") or "") for child in children]
    child_by_shot: dict[str, list[str]] = {}
    for child in children:
        child_id = str(child.get("id") or "")
        for shot_id in _string_list(child.get("split_assigned_shot_ids")):
            child_by_shot.setdefault(shot_id, []).append(child_id)
    for shot in shots:
        shot_id = str(shot.get("segment_id") or "").strip()
        retained = [value for value in _string_list(shot.get("asset_ids")) if value != asset_id]
        shot["asset_ids"] = list(dict.fromkeys([*retained, *child_by_shot.get(shot_id, [])]))
        scene_retained = [value for value in _string_list(shot.get("scene_asset_ids")) if value != asset_id]
        shot["scene_asset_ids"] = list(dict.fromkeys([*scene_retained, *child_by_shot.get(shot_id, [])]))

    parent["superseded"] = True
    parent["status"] = "superseded"
    parent["manual_asset_status"] = "ignored"
    parent["source_usage_role"] = "superseded_parent"
    parent["asset_hidden_by_default"] = True
    parent["superseded_by_asset_ids"] = child_ids
    parent["split_child_asset_ids"] = child_ids
    parent["split_at"] = time.time()
    parent["used_shots"] = []
    parent["usage_count"] = 0
    data["assets"] = [*assets, *children]
    data["shots"] = shots
    return {
        "asset_id": asset_id,
        "status": "split",
        "child_ids": child_ids,
        "children_count": len(child_ids),
    }


def _split_storyboard_mode2_scenes(payload: dict[str, Any]) -> dict[str, Any]:
    project_dir = str(payload.get("project_dir") or "").strip()
    if not project_dir:
        raise ValueError("missing_project_dir")
    root = _resolve_storyboard_mode2_project_dir(project_dir)
    store_path = _storyboard_mode2_asset_store_path(root)
    if not store_path.exists():
        raise ValueError(f"mode2 storyboard asset store not found: {store_path}")
    data = json.loads(store_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("mode2 storyboard asset store is invalid")
    _refresh_mode2_structured_fields(root, data)
    dedupe = _mode2_dedupe_scene_assets(data)
    aliases = data.get("scene_asset_aliases") if isinstance(data.get("scene_asset_aliases"), dict) else {}

    requested_ids = _string_list(payload.get("asset_ids"))
    single_id = str(payload.get("asset_id") or "").strip()
    if single_id:
        requested_ids.append(single_id)
    if bool(payload.get("all_mixed")):
        requested_ids.extend(
            str(asset.get("id") or "").strip()
            for asset in (data.get("assets") or [])
            if isinstance(asset, dict)
            and str(asset.get("kind") or "") == "scene"
            and not _mode2_scene_asset_is_superseded(asset)
            and _mode2_scene_is_mixed_visual_bundle(asset)
        )
    requested_ids = list(dict.fromkeys(value for value in requested_ids if value))
    if not requested_ids:
        raise ValueError("missing_asset_id_or_all_mixed")

    results: list[dict[str, Any]] = []
    split_count = 0
    children_count = 0
    processed: set[str] = set()
    working = data
    for requested_id in requested_ids:
        resolved_id = requested_id
        seen_aliases: set[str] = set()
        while resolved_id in aliases and resolved_id not in seen_aliases:
            seen_aliases.add(resolved_id)
            resolved_id = str(aliases[resolved_id] or resolved_id)
        if resolved_id in processed:
            results.append({
                "asset_id": requested_id,
                "resolved_asset_id": resolved_id,
                "status": "skipped",
                "reason": "duplicate_request_after_scene_dedupe",
            })
            continue
        processed.add(resolved_id)
        candidate = copy.deepcopy(working)
        try:
            item_result = _mode2_split_scene_asset_in_data(candidate, resolved_id)
        except ValueError as exc:
            results.append({
                "asset_id": requested_id,
                "resolved_asset_id": resolved_id,
                "status": "skipped",
                "reason": str(exc),
            })
            continue
        except Exception as exc:  # noqa: BLE001
            results.append({
                "asset_id": requested_id,
                "resolved_asset_id": resolved_id,
                "status": "error",
                "reason": str(exc),
            })
            continue
        working = candidate
        split_count += 1
        children_count += int(item_result.get("children_count") or 0)
        results.append({**item_result, "requested_asset_id": requested_id, "resolved_asset_id": resolved_id})

    if split_count or bool(dedupe.get("changed")):
        _refresh_mode2_structured_fields(root, working)
        _write_storyboard_mode2_store(root, working)
    response = _load_mode2_storyboard_result(root, store_path)
    response.update({
        "split_count": split_count,
        "children_count": children_count,
        "results": results,
        "scene_dedupe": working.get("scene_dedupe") or dedupe,
    })
    return response


def _assistant_chat(payload: dict[str, Any]) -> dict[str, Any]:
    base_url = str(payload.get("base_url") or payload.get("baseUrl") or "").strip().rstrip("/")
    api_key = str(payload.get("api_key") or payload.get("apiKey") or "").strip()
    model = str(payload.get("model") or "gpt-5.5").strip()
    message = str(payload.get("message") or "").strip()
    context = str(payload.get("context") or "").strip()
    logs = payload.get("logs") if isinstance(payload.get("logs"), list) else []
    screenshot = payload.get("screenshot") if isinstance(payload.get("screenshot"), dict) else {}
    if not base_url:
        raise ValueError("assistant_base_url_missing")
    if not api_key:
        raise ValueError("assistant_api_key_missing")
    if not model:
        raise ValueError("assistant_model_missing")
    if not message:
        raise ValueError("assistant_message_missing")

    log_text = "\n".join(str(line or "") for line in logs[-80:])[:8000]
    context_text = context[:8000]
    screenshot_url = str(screenshot.get("data_url") or screenshot.get("dataUrl") or "").strip()
    if screenshot_url and not screenshot_url.startswith("data:image/"):
        screenshot_url = ""
    user_text = (
        f"页面上下文:\n{context_text or '-'}\n\n"
        f"最近日志:\n{log_text or '-'}\n\n"
        f"用户问题:\n{message}"
    )
    user_content: Any = user_text
    if screenshot_url:
        user_content = [
            {"type": "text", "text": user_text + "\n\n请优先看截图上真实存在的按钮和位置。"},
            {"type": "image_url", "image_url": {"url": screenshot_url}},
        ]
    url_candidates = []
    if base_url.endswith("/chat/completions"):
        url_candidates.append(base_url)
    elif base_url.endswith("/v1"):
        url_candidates.append(base_url.rstrip("/") + "/chat/completions")
    else:
        url_candidates.append(base_url.rstrip("/") + "/v1/chat/completions")
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是 SP 短剧视频工作台里的视频制作向导。回答要短，最多 3 条，每条尽量一句话。"
                    "不要让用户到处找；能用助手动作完成的，就说“点助手里的按钮”。"
                    "如果有截图，必须按截图上真实存在的按钮名和页面内容回答；不要编造右侧资产详情面板。"
                    "当前资产卡常见按钮是“换图”“描点”“识别轨道”。"
                    "优先给一个最该做的下一步；不知道就说需要先看日志或让用户贴截图。"
                ),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        "temperature": 0.2,
        "max_tokens": 500,
    }
    response = None
    data = None
    attempted_urls = []
    last_error = ""
    for url in dict.fromkeys(url_candidates):
        attempted_urls.append(url)
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
            timeout=120,
        )
        response_text = response.text or ""
        if response.status_code in {404, 405}:
            last_error = f"assistant_http_{response.status_code}: {response_text[:800]}"
            continue
        if response.status_code >= 400:
            tried = " -> ".join(attempted_urls)
            raise RuntimeError(f"assistant_http_{response.status_code}: {response_text[:800]} | tried: {tried}")
        try:
            data = response.json()
            break
        except ValueError:
            content_type = response.headers.get("Content-Type", "")
            preview = response_text[:800] if response_text else "<empty response>"
            last_error = (
                f"assistant_non_json_response: status={response.status_code}, "
                f"content_type={content_type}, body={preview}"
            )
            continue
    if response is None:
        raise RuntimeError("assistant_no_endpoint_attempted")
    if data is None:
        tried = " -> ".join(attempted_urls)
        raise RuntimeError(f"{last_error or 'assistant_no_json_response'} | tried: {tried}")
    text = ""
    choices = data.get("choices") if isinstance(data, dict) else None
    if choices and isinstance(choices, list):
        first = choices[0] if choices else {}
        if isinstance(first, dict):
            msg = first.get("message") if isinstance(first.get("message"), dict) else {}
            text = str(msg.get("content") or first.get("text") or "").strip()
    if not text and isinstance(data, dict):
        text = str(data.get("output_text") or data.get("text") or "").strip()
    return {"ok": True, "text": text, "raw_model": model}


MODE2_TIMELINE_GENERATED_KEYS = {
    "generated_path",
    "seedance_output_path",
    "new_video_path",
    "generated_url",
    "seedance_result_url",
    "seedance_task_id",
    "seedance_task_status",
    "seedance_submitted_at",
    "seedance_finished_at",
    "generated_at",
}

MODE2_TIMELINE_PREVIEW_KEYS = {
    "preview_clip_path",
    "clip_output_path",
    "output_path",
}


def _mode2_unique_list(*values: Any) -> list[str]:
    items: list[str] = []
    for value in values:
        items.extend(_string_list(value))
    return list(dict.fromkeys(items))


def _mode2_shot_time_range(shot: dict[str, Any]) -> tuple[float, float]:
    start = max(0.0, _mode2_float(shot.get("start"), 0.0))
    end = _mode2_float(shot.get("end"), start + _mode2_float(shot.get("duration"), 0.0))
    if end < start:
        end = start
    return start, end


def _mode2_clear_timeline_outputs(shot: dict[str, Any], *, clear_preview: bool = True) -> None:
    for key in MODE2_TIMELINE_GENERATED_KEYS:
        shot.pop(key, None)
    if clear_preview:
        for key in MODE2_TIMELINE_PREVIEW_KEYS:
            shot.pop(key, None)


def _mode2_prune_timeline_fields(shot: dict[str, Any]) -> None:
    start, end = _mode2_shot_time_range(shot)
    shot["start"] = round(start, 3)
    shot["end"] = round(end, 3)
    shot["duration"] = round(max(0.0, end - start), 3)
    for key in ("prompt", "compiled_prompt", "compiled_prompt_long", "analysis_prompt", "seedance_prompt", "asset_refs"):
        shot.pop(key, None)
    _mode2_clear_timeline_outputs(shot)


def _mode2_merge_text(left: Any, right: Any, *, fallback: str = "") -> str:
    values = [str(value or "").strip() for value in (left, right) if str(value or "").strip()]
    if not values:
        return fallback
    return "；".join(list(dict.fromkeys(values)))


def _mode2_merge_person_count(left: dict[str, Any], right: dict[str, Any]) -> int:
    values = [
        _safe_int(item.get("person_count"), -1)
        for item in (left, right)
    ]
    values = [value for value in values if value >= 0]
    return max(values) if values else -1


def _mode2_apply_shot_id_mapping_to_assets(
    assets: list[dict[str, Any]],
    shot_id_map: dict[str, list[str]],
) -> None:
    def mapped_values(values: Any) -> list[str]:
        result: list[str] = []
        for value in _string_list(values):
            result.extend(shot_id_map.get(value, [value]))
        return list(dict.fromkeys([item for item in result if item]))

    list_fields = (
        "present_shots",
        "used_shots",
        "evidence_shots",
        "shot_ids",
        "split_assigned_shot_ids",
        "assigned_shot_ids",
    )
    for asset in assets:
        for key in list_fields:
            if key in asset:
                asset[key] = mapped_values(asset.get(key))
        new_anchors: list[dict[str, Any]] = []
        seen_anchor_keys: set[str] = set()
        for anchor in asset.get("identity_anchors") or []:
            if not isinstance(anchor, dict):
                continue
            old_id = str(anchor.get("shot_id") or "").strip()
            mapped = shot_id_map.get(old_id)
            target_ids = mapped if mapped else ([old_id] if old_id else [])
            if not target_ids:
                new_anchors.append(anchor)
                continue
            for target_id in target_ids:
                cloned = copy.deepcopy(anchor)
                cloned["shot_id"] = target_id
                anchor_key = json.dumps(cloned, ensure_ascii=False, sort_keys=True, default=str)
                if anchor_key in seen_anchor_keys:
                    continue
                seen_anchor_keys.add(anchor_key)
                new_anchors.append(cloned)
        if new_anchors:
            asset["identity_anchors"] = new_anchors


def _mode2_renumber_timeline_shots(
    shots: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    source_to_new: dict[str, list[str]] = {}
    for index, shot in enumerate(shots, start=1):
        old_id = str(shot.get("segment_id") or "").strip()
        new_id = f"S{index:03d}"
        if old_id:
            source_to_new.setdefault(old_id, []).append(new_id)
        if old_id != new_id:
            _mode2_clear_timeline_outputs(shot)
        shot["segment_id"] = new_id
    return shots, source_to_new


def _mode2_finish_timeline_edit(
    root: Path,
    store_path: Path,
    data: dict[str, Any],
    *,
    edit_summary: dict[str, Any],
    selected_segment_id: str,
    selected_index: int,
) -> dict[str, Any]:
    assets = [item for item in (data.get("assets") or []) if isinstance(item, dict)]
    shots = [item for item in (data.get("shots") or []) if isinstance(item, dict)]
    shots.sort(key=lambda item: (_mode2_float(item.get("start"), 0.0), _mode2_float(item.get("end"), 0.0)))
    shots, shot_id_map = _mode2_renumber_timeline_shots(shots)
    _mode2_apply_shot_id_mapping_to_assets(assets, shot_id_map)
    data["assets"] = assets
    data["shots"] = shots
    if selected_segment_id in shot_id_map and shot_id_map[selected_segment_id]:
        selected_segment_id = shot_id_map[selected_segment_id][0]
    selected_index = next(
        (idx for idx, shot in enumerate(shots) if str(shot.get("segment_id") or "") == selected_segment_id),
        max(0, min(selected_index, len(shots) - 1)) if shots else -1,
    )
    if 0 <= selected_index < len(shots):
        selected_segment_id = str(shots[selected_index].get("segment_id") or selected_segment_id)

    edit_summary = {
        **edit_summary,
        "selected_segment_id": selected_segment_id,
        "selected_index": selected_index,
        "shot_count": len(shots),
        "edited_at": time.time(),
    }
    history = data.get("timeline_manual_history") if isinstance(data.get("timeline_manual_history"), list) else []
    history.append(edit_summary)
    data["timeline_manual_history"] = history[-200:]
    data["timeline_stage"] = "manual_timeline_edited"
    data["storyboard_stage"] = "manual_timeline_edited"
    _refresh_mode2_structured_fields(root, data)
    _write_storyboard_mode2_store(root, data)
    response = _load_mode2_storyboard_result(root, store_path)
    response["timeline_edit"] = edit_summary
    response["selected_segment_id"] = selected_segment_id
    response["selected_index"] = selected_index
    return response


def _mode2_split_timeline_shot(
    shots: list[dict[str, Any]],
    index: int,
    split_time: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], str, int]:
    parent = shots[index]
    parent_id = str(parent.get("segment_id") or "").strip()
    start, end = _mode2_shot_time_range(parent)
    min_piece = 0.3
    if split_time <= start + min_piece or split_time >= end - min_piece:
        raise ValueError(f"split_time_too_close_to_edge: {split_time:.3f}, shot={start:.3f}-{end:.3f}")

    left = copy.deepcopy(parent)
    right = copy.deepcopy(parent)
    source_ids = _mode2_unique_list(parent.get("source_segment_ids"), [parent_id])
    for child, child_start, child_end, child_role in (
        (left, start, split_time, "left"),
        (right, split_time, end, "right"),
    ):
        child["source_segment_ids"] = source_ids
        child["manual_parent_segment_id"] = parent_id
        child["manual_timeline_edit"] = {
            "action": "split_at_time",
            "parent_segment_id": parent_id,
            "split_time": round(split_time, 3),
            "part": child_role,
        }
        child["start"] = round(child_start, 3)
        child["end"] = round(child_end, 3)
        child["duration"] = round(child_end - child_start, 3)
        _mode2_prune_timeline_fields(child)

    new_shots = [*shots[:index], left, right, *shots[index + 1:]]
    edit_summary = {
        "action": "split_at_time",
        "parent_segment_id": parent_id,
        "split_time": round(split_time, 3),
        "old_range": [round(start, 3), round(end, 3)],
        "new_ranges": [[round(start, 3), round(split_time, 3)], [round(split_time, 3), round(end, 3)]],
    }
    return new_shots, edit_summary, parent_id, index


def _mode2_merge_timeline_shots(
    shots: list[dict[str, Any]],
    left_index: int,
    right_index: int,
    *,
    action: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], str, int]:
    left = shots[left_index]
    right = shots[right_index]
    left_id = str(left.get("segment_id") or "").strip()
    right_id = str(right.get("segment_id") or "").strip()
    left_start, left_end = _mode2_shot_time_range(left)
    right_start, right_end = _mode2_shot_time_range(right)
    if right_start < left_start:
        raise ValueError("timeline_order_invalid")
    if abs(left_end - right_start) > 0.12:
        raise ValueError(f"shots_not_contiguous: {left_end:.3f}-{right_start:.3f}")

    merged = copy.deepcopy(left)
    merged["start"] = round(left_start, 3)
    merged["end"] = round(right_end, 3)
    merged["duration"] = round(max(0.0, right_end - left_start), 3)
    merged["source_segment_ids"] = _mode2_unique_list(
        left.get("source_segment_ids"),
        right.get("source_segment_ids"),
        [left_id, right_id],
    )
    for key in ("asset_ids", "role_asset_ids", "scene_asset_ids", "prop_asset_ids"):
        merged[key] = _mode2_unique_list(left.get(key), right.get(key))
    boundary_hints: list[Any] = []
    for value in (left.get("boundary_hints"), right.get("boundary_hints")):
        if isinstance(value, list):
            boundary_hints.extend(item for item in value if item)
    merged["boundary_hints"] = boundary_hints
    merged["person_count"] = _mode2_merge_person_count(left, right)
    merged["description"] = _mode2_merge_text(left.get("description"), right.get("description"), fallback="手工合并镜头")
    merged["summary"] = _mode2_merge_text(left.get("summary"), right.get("summary"))
    merged["manual_merged_segment_ids"] = _mode2_unique_list(
        left.get("manual_merged_segment_ids"),
        right.get("manual_merged_segment_ids"),
        [left_id, right_id],
    )
    merged["manual_timeline_edit"] = {
        "action": action,
        "merged_segment_ids": [left_id, right_id],
        "old_ranges": [
            [round(left_start, 3), round(left_end, 3)],
            [round(right_start, 3), round(right_end, 3)],
        ],
    }
    _mode2_prune_timeline_fields(merged)

    new_shots = [*shots[:left_index], merged, *shots[right_index + 1:]]
    edit_summary = {
        "action": action,
        "merged_segment_ids": [left_id, right_id],
        "old_ranges": [
            [round(left_start, 3), round(left_end, 3)],
            [round(right_start, 3), round(right_end, 3)],
        ],
        "new_range": [round(left_start, 3), round(right_end, 3)],
    }
    return new_shots, edit_summary, left_id, left_index


def _edit_storyboard_mode2_shot(payload: dict[str, Any]) -> dict[str, Any]:
    project_dir = str(payload.get("project_dir") or "").strip()
    action = str(payload.get("action") or "").strip()
    segment_id = str(payload.get("segment_id") or "").strip()
    if not project_dir:
        raise ValueError("missing_project_dir")
    if action not in {"split_at_time", "merge_prev", "merge_next"}:
        raise ValueError(f"invalid_timeline_action: {action}")
    if not segment_id:
        raise ValueError("missing_segment_id")

    root = _resolve_storyboard_mode2_project_dir(project_dir)
    store_path = _storyboard_mode2_asset_store_path(root)
    data = _load_storyboard_mode2_store(root)
    shots = [item for item in (data.get("shots") or []) if isinstance(item, dict)]
    if not shots:
        raise ValueError("mode2_timeline_empty")
    index = next((idx for idx, shot in enumerate(shots) if str(shot.get("segment_id") or "") == segment_id), -1)
    if index < 0:
        raise ValueError(f"segment_not_found: {segment_id}")

    if action == "split_at_time":
        split_time = _mode2_float(payload.get("time"), -1.0)
        shots, edit_summary, selected_segment_id, selected_index = _mode2_split_timeline_shot(shots, index, split_time)
    elif action == "merge_prev":
        if index <= 0:
            raise ValueError("no_previous_shot_to_merge")
        shots, edit_summary, selected_segment_id, selected_index = _mode2_merge_timeline_shots(
            shots,
            index - 1,
            index,
            action=action,
        )
    else:
        if index >= len(shots) - 1:
            raise ValueError("no_next_shot_to_merge")
        shots, edit_summary, selected_segment_id, selected_index = _mode2_merge_timeline_shots(
            shots,
            index,
            index + 1,
            action=action,
        )

    data["shots"] = shots
    return _mode2_finish_timeline_edit(
        root,
        store_path,
        data,
        edit_summary=edit_summary,
        selected_segment_id=selected_segment_id,
        selected_index=selected_index,
    )


MODE2_ASSET_AUDIT_SEVERITY_RANK = {"info": 0, "warning": 1, "blocked": 2}


def _mode2_asset_display_name(asset: dict[str, Any]) -> str:
    name = str(asset.get("name") or "").strip()
    asset_id = str(asset.get("id") or "").strip()
    return name or asset_id or "未命名资产"


def _mode2_asset_issue(
    code: str,
    severity: str,
    message: str,
    *,
    related_asset_ids: list[str] | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    level = severity if severity in MODE2_ASSET_AUDIT_SEVERITY_RANK else "warning"
    issue: dict[str, Any] = {
        "code": code,
        "severity": level,
        "message": message,
    }
    if related_asset_ids:
        issue["related_asset_ids"] = related_asset_ids
    if evidence:
        issue["evidence"] = evidence
    return issue


def _mode2_asset_add_issue(asset: dict[str, Any], issue: dict[str, Any]) -> None:
    issues = asset.setdefault("asset_audit_issues", [])
    if not isinstance(issues, list):
        issues = []
        asset["asset_audit_issues"] = issues
    issues.append(issue)


def _mode2_asset_status_from_issues(issues: list[dict[str, Any]]) -> str:
    worst = "ok"
    worst_rank = -1
    for issue in issues:
        severity = str(issue.get("severity") or "warning")
        rank = MODE2_ASSET_AUDIT_SEVERITY_RANK.get(severity, 1)
        if rank > worst_rank:
            worst = severity
            worst_rank = rank
    return worst if worst_rank >= 0 else "ok"


def _mode2_asset_repair_action(asset: dict[str, Any], issues: list[dict[str, Any]]) -> dict[str, str]:
    kind = str(asset.get("kind") or "")
    codes = {str(issue.get("code") or "") for issue in issues}
    if kind == "role":
        if "manual_replacement_missing" in codes or "role_target_image_missing" in codes:
            return {
                "kind": "target_image",
                "label": "替换图片",
                "hint": "给这个原片角色绑定真正要送进 Seedance 的目标角色参考图。",
            }
        if "role_missing_identity_anchor" in codes:
            return {
                "kind": "identity_anchor",
                "label": "先标身份点",
                "hint": "在正确时间帧点到这个角色本人身上，再跑 SAM3 身份分轨。",
            }
        if "role_missing_sam3_identity_track" in codes:
            return {
                "kind": "identity_track",
                "label": "跑分轨",
                "hint": "已有身份点，但还没有可用身份轨，请运行 SAM3 分轨后再入库。",
            }
        if "role_duplicate_source_image" in codes or "role_identity_collision" in codes:
            return {
                "kind": "identity_collision",
                "label": "重建身份",
                "hint": "当前角色和其他角色串图，请重新标身份点并跑分轨，或直接绑定正确目标图后确认。",
            }
        if "manual_replacement_missing" in codes or "role_target_image_missing" in codes:
            return {
                "kind": "target_image",
                "label": "替换图片",
                "hint": "给这个角色绑定正确目标参考图。",
            }
        return {
            "kind": "role_review",
            "label": "修复角色",
            "hint": "核对角色身份、目标图和身份轨后再放行。",
        }
    if kind == "scene":
        return {
            "kind": "scene_reference",
            "label": "换空镜/远景",
            "hint": "当前图只适合镜头参考；需要干净背景时请换空镜或远景。",
        }
    if kind == "prop":
        return {
            "kind": "prop_refine",
            "label": "重提纯/换物品图",
            "hint": "物品候选不干净，请重提纯或换成清晰物体图。",
        }
    return {
        "kind": "asset_review",
        "label": "修复资产",
        "hint": "核对资产后再放行。",
    }


def _mode2_asset_file_hash(path_value: Any) -> dict[str, Any]:
    path_text = str(path_value or "").strip()
    if not path_text:
        raise ValueError("empty_path")
    path = Path(path_text)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path_text)
    digest = hashlib.md5()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {
        "hash": digest.hexdigest(),
        "path": str(path),
        "size": size,
        "mtime": path.stat().st_mtime,
    }


def _mode2_asset_image_dhash(path_value: Any) -> str:
    from PIL import Image

    path = Path(str(path_value or "").strip())
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(str(path))
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8))
        pixels = list(gray.getdata())
    value = 0
    for y in range(8):
        row = y * 9
        for x in range(8):
            value = (value << 1) | (1 if pixels[row + x] > pixels[row + x + 1] else 0)
    return f"{value:016x}"


def _mode2_asset_hamming_hex(left: str, right: str) -> int:
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except Exception:  # noqa: BLE001
        return 999


def _mode2_refinement_hash_record(path_value: Any) -> dict[str, Any]:
    info = _mode2_asset_file_hash(path_value)
    return {
        "path": str(info["path"]),
        "algorithm": "md5",
        "hash": str(info["hash"]),
        "size": int(info["size"]),
        "fingerprint": f"md5:{info['hash']}:{info['size']}",
    }


def _mode2_refinement_output_path(provenance: dict[str, Any], key: str) -> str:
    outputs = provenance.get("outputs") if isinstance(provenance.get("outputs"), dict) else {}
    value = outputs.get(key)
    if isinstance(value, dict):
        return str(value.get("path") or "").strip()
    return str(value or "").strip()


def _mode2_refinement_provenance_payload(
    asset: dict[str, Any],
    *,
    source_path: str | Path,
    outputs: dict[str, Any],
    job_id: str,
    prompt_id: str,
    method: str,
    kind: str,
) -> dict[str, Any]:
    source = _mode2_refinement_hash_record(source_path)
    output_records: dict[str, Any] = {}
    for key, value in outputs.items():
        path_text = str(value or "").strip()
        if not path_text:
            continue
        output_records[key] = _mode2_refinement_hash_record(path_text)
    if "refined_source_image" not in output_records:
        raise RuntimeError("refinement provenance missing refined source output")
    return {
        "schema_version": 1,
        "asset_id": str(asset.get("id") or "").strip(),
        "asset_kind": str(asset.get("kind") or kind).strip(),
        "refinement_kind": kind,
        "method": method,
        "job_id": str(job_id or ""),
        "prompt_id": str(prompt_id or ""),
        "created_at": time.time(),
        "source": source,
        "source_segment_ids": [
            str(value or "").strip()
            for value in (asset.get("source_segment_ids") or [])
            if str(value or "").strip()
        ],
        "source_time_range": asset.get("source_time_range") or asset.get("time_span") or {},
        "outputs": output_records,
    }


def _mode2_write_refinement_provenance(
    output_dir: Path,
    prefix: str,
    asset: dict[str, Any],
    *,
    source_path: str | Path,
    outputs: dict[str, Any],
    job_id: str,
    prompt_id: str,
    method: str,
    kind: str,
) -> tuple[dict[str, Any], str]:
    provenance = _mode2_refinement_provenance_payload(
        asset,
        source_path=source_path,
        outputs=outputs,
        job_id=job_id,
        prompt_id=prompt_id,
        method=method,
        kind=kind,
    )
    sidecar = output_dir / f"{prefix}_provenance.json"
    sidecar.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    return provenance, str(sidecar)


def _mode2_validate_refinement_provenance(
    asset: dict[str, Any],
    provenance: dict[str, Any] | None = None,
) -> tuple[bool, str, str]:
    provenance = provenance if isinstance(provenance, dict) else (
        asset.get("refinement_provenance")
        if isinstance(asset.get("refinement_provenance"), dict)
        else {}
    )
    if not provenance:
        return False, "missing_provenance", "提纯结果没有 provenance sidecar/source fingerprint，可能是重分析前的旧产物。"
    asset_id = str(asset.get("id") or "").strip()
    if not asset_id or str(provenance.get("asset_id") or "").strip() != asset_id:
        return False, "asset_mismatch", "提纯 provenance 的 asset_id 与当前资产不一致，疑似跨资产串图。"
    asset_kind = str(asset.get("kind") or "").strip()
    provenance_kind = str(provenance.get("asset_kind") or "").strip()
    if provenance_kind and provenance_kind != asset_kind:
        return False, "kind_mismatch", "提纯 provenance 的资产类型与当前资产不一致。"

    source_record = provenance.get("source") if isinstance(provenance.get("source"), dict) else {}
    recorded_path = str(source_record.get("path") or "").strip()
    recorded_hash = str(source_record.get("hash") or "").strip()
    if not recorded_path or not recorded_hash:
        return False, "missing_source_fingerprint", "提纯 provenance 没有明确记录源图路径和 hash。"
    current_source = _storyboard_asset_primary_source_path(asset, prefer_crop=False)
    if current_source is None:
        return False, "current_source_missing", "当前资产没有可校验的源图，旧提纯结果不能复用。"
    try:
        if Path(recorded_path).resolve() != current_source.resolve():
            return False, "source_path_mismatch", "提纯使用的源图路径与当前资产代表源帧不一致。"
        current_hash = _mode2_asset_file_hash(current_source)
    except Exception as exc:  # noqa: BLE001
        return False, "source_hash_failed", f"当前源图无法完成指纹校验: {exc}"
    if str(current_hash.get("hash") or "") != recorded_hash:
        return False, "source_hash_mismatch", "当前资产源图内容已变化，提纯结果属于旧分析或其他场景。"
    recorded_size = source_record.get("size")
    if recorded_size not in {None, ""} and int(recorded_size) != int(current_hash.get("size") or 0):
        return False, "source_size_mismatch", "当前资产源图大小与提纯 provenance 不一致。"

    refined_path = str(asset.get("refined_source_image") or "").strip()
    output_path = _mode2_refinement_output_path(provenance, "refined_source_image")
    if not output_path:
        return False, "missing_output_provenance", "提纯 provenance 没有记录代表图输出。"
    if refined_path:
        try:
            if Path(refined_path).resolve() != Path(output_path).resolve():
                return False, "output_path_mismatch", "当前挂载的提纯图与 provenance 输出不一致。"
        except Exception:  # noqa: BLE001
            return False, "output_path_mismatch", "当前挂载的提纯图路径无效。"
    output_record = (
        provenance.get("outputs", {}).get("refined_source_image")
        if isinstance(provenance.get("outputs"), dict)
        else {}
    )
    output_hash = str(output_record.get("hash") or "").strip() if isinstance(output_record, dict) else ""
    try:
        current_output_hash = _mode2_asset_file_hash(output_path)
    except Exception as exc:  # noqa: BLE001
        return False, "output_missing", f"提纯代表图不存在或无法读取: {exc}"
    if not output_hash or str(current_output_hash.get("hash") or "") != output_hash:
        return False, "output_hash_mismatch", "提纯代表图内容与 provenance 不一致。"

    recorded_segments = {
        str(value or "").strip()
        for value in (provenance.get("source_segment_ids") or [])
        if str(value or "").strip()
    }
    current_segments = {
        str(value or "").strip()
        for value in (asset.get("source_segment_ids") or [])
        if str(value or "").strip()
    }
    if recorded_segments and current_segments and recorded_segments != current_segments:
        return False, "source_segments_mismatch", "提纯 provenance 对应的分段与当前场景资产不一致。"
    return True, "verified", ""


def _mode2_apply_scene_refinement_provenance_contract(asset: dict[str, Any]) -> None:
    if str(asset.get("kind") or "") != "scene":
        return
    refined_source = str(asset.get("refined_source_image") or "").strip()
    refinement_kind = str(asset.get("refinement_kind") or "").strip()
    if not refined_source or refinement_kind != "clean_background":
        asset["refinement_provenance_status"] = str(asset.get("refinement_provenance_status") or "not_applicable")
        return
    verified, status, warning = _mode2_validate_refinement_provenance(asset)
    asset["refinement_provenance_status"] = status
    asset["refinement_provenance_verified"] = verified
    asset["refinement_provenance_warning"] = warning
    if verified:
        provenance = asset.get("refinement_provenance") or {}
        source = provenance.get("source") if isinstance(provenance.get("source"), dict) else {}
        asset["refinement_source_path"] = str(source.get("path") or asset.get("refinement_source_path") or "")
        asset["refinement_source_hash"] = str(source.get("hash") or "")
        asset["refinement_source_fingerprint"] = str(source.get("fingerprint") or "")
        return
    asset["refinement_status"] = "quarantined"
    asset["refinement_warning"] = warning
    asset["refinement_quarantined"] = True
    asset["asset_quarantined"] = True
    asset["asset_quarantine_reason"] = warning
    asset["source_quality_status"] = "scene_refinement_provenance_invalid"
    asset["source_trust_level"] = "stale_refinement_quarantined"


def _mode2_asset_quality_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _mode2_asset_quality_issue(asset: dict[str, Any]) -> dict[str, Any] | None:
    quality = asset.get("refinement_quality")
    if not isinstance(quality, dict):
        return None
    area_ratio = _mode2_asset_quality_number(quality.get("area_ratio"))
    bbox_ratio = _mode2_asset_quality_number(quality.get("bbox_ratio"))
    component_count = _mode2_asset_quality_number(quality.get("component_count"))
    bad = (
        (area_ratio is not None and area_ratio >= 0.22)
        or (bbox_ratio is not None and bbox_ratio >= 0.58)
        or (component_count is not None and component_count >= 7)
    )
    if not bad:
        return None
    return _mode2_asset_issue(
        "mask_quality_unclean",
        "warning",
        "提纯 mask 范围过大或过散，可能混入背景、人物或多个同类物，需要换更干净的源图/裁剪图。",
        evidence={
            "area_ratio": area_ratio,
            "bbox_ratio": bbox_ratio,
            "component_count": component_count,
        },
    )


MODE2_ASSET_LIBRARY_SCHEMA_VERSION = 2


def _mode2_path_text(value: Any, *, require_file: bool = False) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if require_file:
        path = Path(text)
        if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
            return ""
    return text


def _mode2_append_unique_path(values: list[str], value: Any, *, require_file: bool = True) -> None:
    text = _mode2_path_text(value, require_file=require_file)
    if not text:
        return
    key = str(Path(text)).lower()
    existing = {str(Path(item)).lower() for item in values}
    if key not in existing:
        values.append(text)


def _mode2_append_identity_evidence(
    evidence: list[dict[str, Any]],
    item: dict[str, Any],
) -> None:
    path = str(item.get("path") or item.get("image_path") or item.get("crop_path") or "").strip()
    mask_path = str(item.get("mask_path") or "").strip()
    key = "|".join([
        str(item.get("type") or ""),
        str(item.get("candidate_id") or ""),
        str(item.get("object_id") or ""),
        str(Path(path)).lower() if path else "",
        str(Path(mask_path)).lower() if mask_path else "",
    ])
    for existing in evidence:
        existing_key = "|".join([
            str(existing.get("type") or ""),
            str(existing.get("candidate_id") or ""),
            str(existing.get("object_id") or ""),
            str(Path(str(existing.get("path") or existing.get("image_path") or existing.get("crop_path") or ""))).lower()
            if str(existing.get("path") or existing.get("image_path") or existing.get("crop_path") or "").strip()
            else "",
            str(Path(str(existing.get("mask_path") or ""))).lower()
            if str(existing.get("mask_path") or "").strip()
            else "",
        ])
        if existing_key == key:
            return
    evidence.append(item)


def _mode2_role_identity_evidence_images(asset: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for value in asset.get("identity_evidence_images") or []:
        _mode2_append_unique_path(paths, value, require_file=True)
    for value in asset.get("candidate_source_images") or []:
        _mode2_append_unique_path(paths, value, require_file=True)
    for value in asset.get("source_images") or []:
        _mode2_append_unique_path(paths, value, require_file=True)
    for key in (
        "source_image",
        "candidate_source_image",
        "refined_source_image",
        "refinement_source_path",
    ):
        _mode2_append_unique_path(paths, asset.get(key), require_file=True)
    return paths


def _mode2_anchor_matches_candidate_object(
    anchor: dict[str, Any],
    *,
    candidate_id: str,
    obj: dict[str, Any],
) -> bool:
    if str(anchor.get("mask_candidate_id") or "").strip() != candidate_id:
        return False
    anchor_object_id = anchor.get("sam3_object_id")
    object_id = obj.get("object_id")
    if anchor_object_id not in (None, ""):
        try:
            if int(anchor_object_id) == int(object_id):
                return True
        except (TypeError, ValueError):
            if str(anchor_object_id) == str(object_id):
                return True
    anchor_color = str(anchor.get("mask_color") or "").strip().lower()
    object_color = str(obj.get("color_key") or "").strip().lower()
    return bool(anchor_color and object_color and anchor_color == object_color)


def _mode2_identity_candidate_evidence_by_role(
    root: Path,
    annotations: list[dict[str, Any]],
    role_ids: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    by_role: dict[str, list[dict[str, Any]]] = {role_id: [] for role_id in role_ids}
    stats = {"result_files": 0, "objects": 0, "assigned_objects": 0}
    base = root / "assets" / "identity_candidates"
    if not base.exists():
        return by_role, stats
    for result_path in sorted(base.glob("maskcand_*/result.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:  # noqa: BLE001
            logging.warning("Mode2 identity candidate ignored: %s / %s", result_path, exc)
            continue
        if not isinstance(result, dict):
            continue
        stats["result_files"] += 1
        candidate_id = str(result.get("candidate_id") or result_path.parent.name).strip()
        result_asset_id = str(result.get("asset_id") or "").strip()
        objects = [item for item in (result.get("objects") or []) if isinstance(item, dict)]
        if not objects:
            continue
        for obj in objects:
            stats["objects"] += 1
            matched_roles = [
                str(anchor.get("role_id") or "").strip()
                for anchor in annotations
                if _mode2_anchor_matches_candidate_object(anchor, candidate_id=candidate_id, obj=obj)
                and str(anchor.get("role_id") or "").strip() in role_ids
            ]
            selected_by_anchor = bool(matched_roles)
            if not matched_roles and result_asset_id in role_ids:
                matched_roles = [result_asset_id]
            if not matched_roles:
                continue
            item = {
                "type": "sam3_mask_candidate_object",
                "source": "identity_candidates",
                "trust": "identity_evidence",
                "candidate_id": candidate_id,
                "candidate_result": str(result_path),
                "time": result.get("time"),
                "role_hint_asset_id": result_asset_id,
                "object_id": obj.get("object_id"),
                "color_key": str(obj.get("color_key") or ""),
                "color_label": str(obj.get("color_label") or ""),
                "color_rgb": obj.get("color_rgb") or [],
                "bbox": obj.get("bbox") or obj.get("bbox_xywh") or [],
                "center_point": obj.get("center_point") or [],
                "area_ratio": obj.get("area_ratio"),
                "score": obj.get("score"),
                "image_path": str(obj.get("crop_path") or ""),
                "crop_path": str(obj.get("crop_path") or ""),
                "mask_path": str(obj.get("mask_path") or ""),
                "original_frame": str(result.get("original_frame") or ""),
                "colored_mask": str(result.get("colored_mask") or ""),
                "overlay_image": str(result.get("overlay_image") or ""),
                "selected_by_identity_anchor": selected_by_anchor,
            }
            for role_id in matched_roles:
                _mode2_append_identity_evidence(by_role.setdefault(role_id, []), dict(item))
                stats["assigned_objects"] += 1
    return by_role, stats


def _organize_storyboard_mode2_asset_library(payload: dict[str, Any]) -> dict[str, Any]:
    project_dir = str(payload.get("project_dir") or "").strip()
    if not project_dir:
        raise ValueError("missing_project_dir")
    root = _resolve_storyboard_mode2_project_dir(project_dir)
    data = _load_storyboard_mode2_store(root)
    assets = [item for item in (data.get("assets") or []) if isinstance(item, dict)]
    shots = [item for item in (data.get("shots") or []) if isinstance(item, dict)]
    annotations = [item for item in (data.get("identity_annotations") or []) if isinstance(item, dict)]
    role_assets = [
        item for item in assets
        if str(item.get("kind") or "") == "role"
    ]
    role_ids = {str(item.get("id") or "").strip() for item in role_assets if str(item.get("id") or "").strip()}
    candidate_evidence_by_role, candidate_stats = _mode2_identity_candidate_evidence_by_role(root, annotations, role_ids)
    now = time.time()
    summary: dict[str, Any] = {
        "schema_version": MODE2_ASSET_LIBRARY_SCHEMA_VERSION,
        "route": "mode2_seedance",
        "project_dir": str(root),
        "organized_at": now,
        "role_count": len(role_assets),
        "roles_with_target_image": 0,
        "roles_missing_target_image": 0,
        "roles_with_identity_evidence": 0,
        "roles_downgraded_face_sources": 0,
        "identity_candidate_results": candidate_stats["result_files"],
        "identity_candidate_objects": candidate_stats["objects"],
        "identity_candidate_objects_assigned": candidate_stats["assigned_objects"],
        "items": [],
    }

    for asset in role_assets:
        asset_id = str(asset.get("id") or "").strip()
        name = _mode2_asset_display_name(asset)
        target_image = str(asset.get("target_image") or "").strip()
        source_kind = str(asset.get("source_kind") or "").strip()
        track_source = str(asset.get("track_source") or "").strip()
        refinement_method = str(asset.get("refinement_method") or "").strip()
        legacy_face_source = (
            source_kind == "role_face_bundle"
            or track_source == "face_detect"
            or refinement_method == "role_face_bundle"
        )
        evidence: list[dict[str, Any]] = [
            item for item in (asset.get("identity_evidence") or [])
            if isinstance(item, dict)
        ]
        candidate_images: list[str] = []
        evidence_images: list[str] = []

        def add_path_evidence(
            path_value: Any,
            *,
            evidence_type: str,
            source: str,
            trust: str = "identity_evidence",
            extra: dict[str, Any] | None = None,
        ) -> None:
            path_text = _mode2_path_text(path_value, require_file=True)
            if not path_text:
                return
            item = {
                "type": evidence_type,
                "source": source,
                "trust": trust,
                "path": path_text,
            }
            if extra:
                item.update(extra)
            _mode2_append_identity_evidence(evidence, item)
            _mode2_append_unique_path(evidence_images, path_text)

        primary_source = str(asset.get("source_image") or "").strip()
        if primary_source:
            add_path_evidence(
                primary_source,
                evidence_type=(
                    "sam3_identity_track_bundle"
                    if source_kind == "mode2_role_identity_track_bundle"
                    else "face_detect_fallback_source"
                    if legacy_face_source
                    else "legacy_role_source"
                ),
                source=source_kind or track_source or "source_image",
                trust="trusted_identity_evidence" if source_kind == "mode2_role_identity_track_bundle" else "identity_evidence_only",
                extra={"was_primary_source_image": True},
            )
        for path_value in asset.get("source_images") or []:
            add_path_evidence(
                path_value,
                evidence_type="role_source_image_item",
                source=source_kind or track_source or "source_images",
                trust="trusted_identity_evidence" if source_kind == "mode2_role_identity_track_bundle" else "identity_evidence_only",
            )
        add_path_evidence(
            asset.get("candidate_source_image"),
            evidence_type="face_cluster_candidate_sheet",
            source=str(asset.get("candidate_source_kind") or "candidate_source_image"),
            trust="identity_evidence_only",
        )
        for path_value in asset.get("candidate_source_images") or []:
            _mode2_append_unique_path(candidate_images, path_value)
            add_path_evidence(
                path_value,
                evidence_type="face_cluster_candidate_crop",
                source=str(asset.get("candidate_source_kind") or "candidate_source_images"),
                trust="identity_evidence_only",
            )
        if refinement_method == "role_face_bundle":
            add_path_evidence(
                asset.get("refined_source_image"),
                evidence_type="role_face_refined_bundle",
                source="role_face_bundle",
                trust="identity_evidence_only",
            )

        for item in candidate_evidence_by_role.get(asset_id, []):
            _mode2_append_identity_evidence(evidence, item)
            _mode2_append_unique_path(evidence_images, item.get("image_path") or item.get("crop_path"))
            _mode2_append_unique_path(candidate_images, item.get("image_path") or item.get("crop_path"))
            _mode2_append_unique_path(evidence_images, item.get("mask_path"))

        for path_value in asset.get("identity_evidence_images") or []:
            _mode2_append_unique_path(evidence_images, path_value)
        for path_value in evidence_images:
            _mode2_append_unique_path(candidate_images, path_value)

        if legacy_face_source:
            asset["legacy_source_kind"] = source_kind or asset.get("legacy_source_kind") or ""
            asset["legacy_track_source"] = track_source or asset.get("legacy_track_source") or ""
            asset["source_kind"] = "identity_evidence_only"
            asset["source_trust_level"] = "identity_evidence_only"
            asset["source_usage_role"] = "identity_evidence_only"
            asset["source_image_is_generation_source"] = False
            asset["identity_evidence_only"] = True
            if str(asset.get("refinement_method") or "") == "role_face_bundle":
                asset["refinement_trust_level"] = "identity_evidence_only"
            summary["roles_downgraded_face_sources"] += 1
        else:
            asset["source_image_is_generation_source"] = False
            asset["identity_evidence_only"] = True
            asset["source_usage_role"] = "identity_evidence_only"
            if source_kind == "mode2_role_identity_track_bundle":
                asset["source_trust_level"] = "trusted_identity_evidence"
            else:
                asset["source_trust_level"] = str(asset.get("source_trust_level") or "identity_evidence")

        asset["identity_evidence"] = evidence
        asset["identity_evidence_images"] = evidence_images
        asset["candidate_source_images"] = candidate_images
        asset["generation_reference_kind"] = "target_image"
        asset["generation_reference_image"] = target_image
        asset["seedance_reference_image"] = target_image
        asset["seedance_reference_role"] = "target_image_only"
        asset["seedance_ready"] = bool(target_image)
        asset["asset_layering"] = {
            "identity_evidence": {
                "purpose": "original video identity proof only; not sent as Seedance target character reference",
                "image_count": len(evidence_images),
                "evidence_count": len(evidence),
            },
            "target_reference": {
                "purpose": "Seedance character reference image",
                "field": "target_image",
                "path": target_image,
                "ready": bool(target_image),
            },
        }
        asset["asset_library_schema_version"] = MODE2_ASSET_LIBRARY_SCHEMA_VERSION
        asset["asset_organized_at"] = now
        if target_image:
            summary["roles_with_target_image"] += 1
        else:
            summary["roles_missing_target_image"] += 1
        if evidence_images:
            summary["roles_with_identity_evidence"] += 1
        summary["items"].append({
            "asset_id": asset_id,
            "name": name,
            "target_image": bool(target_image),
            "identity_evidence_images": len(evidence_images),
            "candidate_source_images": len(candidate_images),
            "downgraded_face_source": legacy_face_source,
            "source_kind": str(asset.get("source_kind") or ""),
        })

    data["asset_library_schema"] = {
        "version": MODE2_ASSET_LIBRARY_SCHEMA_VERSION,
        "route": "mode2_seedance",
        "updated_at": now,
        "layers": {
            "identity_evidence": (
                "原片身份/SAM3/face_detect 证据，只用于识别谁是谁、选帧和内部校验；"
                "不作为 Seedance 的目标角色参考。"
            ),
            "target_image": "Seedance 生成真正使用的目标角色参考图。",
        },
        "generation_reference_field": "target_image",
        "mode_boundary": "Mode2 only; does not depend on Mode1 transfer/render workflow.",
    }
    data["last_asset_organize"] = summary
    history = data.get("asset_organize_history") if isinstance(data.get("asset_organize_history"), list) else []
    history.append(summary)
    data["asset_organize_history"] = history[-50:]
    data["assets"] = assets
    data["shots"] = shots
    _attach_storyboard_asset_usage(assets, shots)
    project_config = _normalize_storyboard_project_config(data.get("project_config") or {})
    _compile_storyboard_prompts(assets, shots, project_config=project_config)
    _refresh_mode2_structured_fields(root, data)
    _write_storyboard_mode2_store(root, data)
    try:
        audited = _audit_storyboard_mode2_assets({"project_dir": str(root)})
    except Exception as exc:  # noqa: BLE001
        logging.warning("Mode2 asset audit skipped after asset organize: %s", exc)
        audited = _load_mode2_storyboard_result(root, _storyboard_mode2_asset_store_path(root))
    audited["asset_organize"] = summary
    return audited


def _audit_storyboard_mode2_assets(payload: dict[str, Any]) -> dict[str, Any]:
    project_dir = str(payload.get("project_dir") or "").strip()
    if not project_dir:
        raise ValueError("missing_project_dir")
    root = _resolve_storyboard_mode2_project_dir(project_dir)
    data = _load_storyboard_mode2_store(root)
    assets = [item for item in (data.get("assets") or []) if isinstance(item, dict)]
    shots = [item for item in (data.get("shots") or []) if isinstance(item, dict)]
    annotations = [item for item in (data.get("identity_annotations") or []) if isinstance(item, dict)]
    role_tracks = data.get("role_tracks") if isinstance(data.get("role_tracks"), dict) else {}
    role_anchor_count: dict[str, int] = {}
    for anchor in annotations:
        role_id = str(anchor.get("role_id") or "").strip()
        if role_id:
            role_anchor_count[role_id] = role_anchor_count.get(role_id, 0) + 1
    asset_schema = data.get("asset_library_schema") if isinstance(data.get("asset_library_schema"), dict) else {}
    asset_route = str(asset_schema.get("route") or "mode2_seedance").strip().lower()
    seedance_route = asset_route in {"", "mode2_seedance", "seedance"}
    identity_issue_severity = "warning" if seedance_route else "blocked"
    target_missing_severity = "blocked" if seedance_route else "warning"

    for asset in assets:
        asset["asset_audit_issues"] = []
        asset["asset_audit_status"] = "ok"
        asset["asset_audit_summary"] = ""
        asset["asset_audit_checked_at"] = time.time()

    role_primary_hashes: dict[str, list[dict[str, Any]]] = {}
    role_visual_hashes: list[tuple[dict[str, Any], str]] = []
    for asset in assets:
        if str(asset.get("kind") or "") != "role":
            continue
        evidence_images = _mode2_role_identity_evidence_images(asset)
        source_image = str(asset.get("source_image") or "").strip() or (evidence_images[0] if evidence_images else "")
        if not source_image:
            _mode2_asset_add_issue(asset, _mode2_asset_issue(
                "role_identity_evidence_missing",
                identity_issue_severity,
                "角色没有源图，不能作为身份资产进入生成。",
            ))
            continue
        try:
            hash_info = _mode2_asset_file_hash(source_image)
            asset["asset_source_hash"] = hash_info["hash"]
            asset["asset_source_hash_path"] = hash_info["path"]
            role_primary_hashes.setdefault(hash_info["hash"], []).append(asset)
            try:
                visual_hash = _mode2_asset_image_dhash(source_image)
                asset["asset_source_visual_hash"] = visual_hash
                role_visual_hashes.append((asset, visual_hash))
            except Exception as exc:  # noqa: BLE001
                asset["asset_source_visual_hash_error"] = str(exc)
        except FileNotFoundError:
            _mode2_asset_add_issue(asset, _mode2_asset_issue(
                "missing_role_identity_evidence_file",
                identity_issue_severity,
                f"角色源图文件不存在: {source_image}",
                evidence={"path": source_image},
            ))
        except Exception as exc:  # noqa: BLE001
            _mode2_asset_add_issue(asset, _mode2_asset_issue(
                "role_source_hash_failed",
                "warning",
                f"角色源图无法计算 hash: {exc}",
                evidence={"path": source_image},
            ))

    for digest, same_assets in role_primary_hashes.items():
        if len(same_assets) < 2:
            continue
        for asset in same_assets:
            related = [
                str(other.get("id") or "")
                for other in same_assets
                if other is not asset and str(other.get("id") or "")
            ]
            related_names = [
                _mode2_asset_display_name(other)
                for other in same_assets
                if other is not asset
            ]
            _mode2_asset_add_issue(asset, _mode2_asset_issue(
                "role_duplicate_source_image",
                identity_issue_severity,
                f"角色源图和 {('、'.join(related_names) or '其他角色')} 完全相同，说明身份资产已经串人。",
                related_asset_ids=related,
                evidence={"hash": digest, "path": str(asset.get("source_image") or "")},
            ))
            asset["source_collision"] = True
            asset["source_collision_with"] = related_names
            asset["source_identity_status"] = "duplicate_source_image"

    for left_index, (left_asset, left_hash) in enumerate(role_visual_hashes):
        for right_asset, right_hash in role_visual_hashes[left_index + 1:]:
            if str(left_asset.get("asset_source_hash") or "") and str(left_asset.get("asset_source_hash") or "") == str(right_asset.get("asset_source_hash") or ""):
                continue
            distance = _mode2_asset_hamming_hex(left_hash, right_hash)
            if distance > 6:
                continue
            left_name = _mode2_asset_display_name(left_asset)
            right_name = _mode2_asset_display_name(right_asset)
            _mode2_asset_add_issue(left_asset, _mode2_asset_issue(
                "role_near_duplicate_source_image",
                "warning",
                f"角色源图和 {right_name} 视觉上高度相似，疑似同一人物或同一批错误抽帧。",
                related_asset_ids=[str(right_asset.get("id") or "")],
                evidence={"visual_hash_distance": distance},
            ))
            _mode2_asset_add_issue(right_asset, _mode2_asset_issue(
                "role_near_duplicate_source_image",
                "warning",
                f"角色源图和 {left_name} 视觉上高度相似，疑似同一人物或同一批错误抽帧。",
                related_asset_ids=[str(left_asset.get("id") or "")],
                evidence={"visual_hash_distance": distance},
            ))

    for asset in assets:
        asset_id = str(asset.get("id") or "").strip()
        kind = str(asset.get("kind") or "").strip()
        name = _mode2_asset_display_name(asset)
        manual_status = str(asset.get("manual_asset_status") or "").strip()
        target_image = str(asset.get("target_image") or "").strip()
        source_kind = str(asset.get("source_kind") or "").strip()
        track_source = str(asset.get("track_source") or "").strip()
        track_status = str(asset.get("track_status") or "").strip()
        identity_status = str(asset.get("identity_status") or "").strip()
        source_quality_status = str(asset.get("source_quality_status") or "").strip()
        refinement_status = str(asset.get("refinement_status") or "").strip()

        if manual_status == "needs_replacement" and not target_image:
            _mode2_asset_add_issue(asset, _mode2_asset_issue(
                "manual_replacement_missing",
                "blocked",
                "你已标记需要换图，但还没有绑定替换图，生成前必须补图。",
            ))
        if manual_status == "ignored":
            _mode2_asset_add_issue(asset, _mode2_asset_issue(
                "manual_ignored",
                "info",
                "该资产已被人工标记忽略，分镜提示词会尽量跳过它。",
            ))

        if kind == "role":
            anchors_in_asset = [
                item for item in (asset.get("identity_anchors") or [])
                if isinstance(item, dict)
            ]
            anchor_total = role_anchor_count.get(asset_id, 0) + len(anchors_in_asset)
            tracked = (
                identity_status == "tracked"
                and bool(str(asset.get("role_track_id") or "").strip())
                and source_kind == "mode2_role_identity_track_bundle"
                and track_source.startswith("mode2_sam3")
            )
            if source_kind == "role_face_bundle" or track_source == "face_detect":
                _mode2_asset_add_issue(asset, _mode2_asset_issue(
                    "role_face_detect_fallback",
                    identity_issue_severity,
                    "当前角色仍来自 face_detect 人脸兜底，不是 SAM3 身份分轨，容易把同框人物抽成同一个人。",
                    evidence={"source_kind": source_kind, "track_source": track_source},
                ))
            if not anchor_total:
                _mode2_asset_add_issue(asset, _mode2_asset_issue(
                    "role_missing_identity_anchor",
                    identity_issue_severity,
                    "角色还没有身份点。请在正确时间帧点到这个人身上，再跑 SAM3 身份分轨。",
                ))
            elif not tracked:
                _mode2_asset_add_issue(asset, _mode2_asset_issue(
                    "role_missing_sam3_identity_track",
                    identity_issue_severity,
                    "角色有身份点但还没有可用的 SAM3 身份轨，不能信任当前源图。",
                    evidence={
                        "identity_status": identity_status,
                        "role_track_id": str(asset.get("role_track_id") or ""),
                        "source_kind": source_kind,
                        "track_source": track_source,
                    },
                ))
            if track_status in {"failed", "needs_anchor"} or identity_status == "failed":
                _mode2_asset_add_issue(asset, _mode2_asset_issue(
                    "role_track_failed",
                    identity_issue_severity,
                    f"{name} 身份分轨状态不可用: {identity_status or track_status}",
                ))
            elif track_status in {"needs_manual_review", "identity_ambiguous"} or identity_status == "needs_review":
                _mode2_asset_add_issue(asset, _mode2_asset_issue(
                    "role_track_needs_review",
                    "warning",
                    f"{name} 身份分轨需要人工核对，当前源图不要直接信任。",
                    evidence={"identity_status": identity_status, "track_status": track_status},
                ))
            if asset.get("source_collision") or str(asset.get("source_identity_status") or "") in {"ambiguous_same_frames", "duplicate_source_image"}:
                _mode2_asset_add_issue(asset, _mode2_asset_issue(
                    "role_identity_collision",
                    identity_issue_severity,
                    f"{name} 与其他角色源图/源帧发生撞车，需要重新标身份点或手工替换。",
                    evidence={"source_collision_with": asset.get("source_collision_with") or []},
                ))
            if not target_image:
                _mode2_asset_add_issue(asset, _mode2_asset_issue(
                    "role_target_image_missing",
                    target_missing_severity,
                    "角色还没有绑定目标参考图。送 Seedance 时不会有稳定的新人物参考。",
                ))
            elif manual_status != "approved":
                _mode2_asset_add_issue(asset, _mode2_asset_issue(
                    "role_target_not_approved",
                    "info",
                    "角色已有目标参考图，但还没有人工点“确认可用”。",
                    evidence={"target_image": target_image},
                ))

        elif kind == "scene":
            if _mode2_scene_is_mixed_visual_bundle(asset):
                _mode2_asset_add_issue(asset, _mode2_asset_issue(
                    "scene_mixed_visual_groups",
                    "blocked",
                    "场景资产包含多个视觉组或不同物理空间，必须先拆分，不能用一张目标图或提纯图整体替换。",
                    evidence={
                        "visual_group_ids": asset.get("scene_visual_group_ids") or [],
                        "visual_group_count": asset.get("scene_visual_group_count") or 0,
                    },
                ))
            provenance_status = str(asset.get("refinement_provenance_status") or "").strip()
            if (
                str(asset.get("refined_source_image") or "").strip()
                and str(asset.get("refinement_kind") or "") == "clean_background"
                and provenance_status != "verified"
            ):
                _mode2_asset_add_issue(asset, _mode2_asset_issue(
                    "scene_refinement_provenance_invalid",
                    "blocked",
                    str(
                        asset.get("refinement_provenance_warning")
                        or "场景提纯结果缺少匹配的源图 provenance，可能来自旧分析或其他资产。"
                    ),
                    evidence={"provenance_status": provenance_status or "missing"},
                ))
            if not str(asset.get("source_image") or "").strip():
                _mode2_asset_add_issue(asset, _mode2_asset_issue(
                    "scene_source_missing",
                    "warning",
                    "场景没有源图，只能按原视频/提示词生成，不能当干净背景参考。",
                ))
            if source_quality_status == "scene_person_heavy" or str(asset.get("source_usage_role") or "") == "shot_reference":
                _mode2_asset_add_issue(asset, _mode2_asset_issue(
                    "scene_person_heavy",
                    "warning",
                    "场景源图人物占比过高，更适合当镜头/构图/动作参考，不适合作为干净背景或环填参考。",
                    evidence={"source_quality_status": source_quality_status},
                ))

        elif kind == "prop":
            if source_quality_status == "prop_needs_visual_check" or track_status == "needs_manual_review":
                _mode2_asset_add_issue(asset, _mode2_asset_issue(
                    "prop_needs_visual_check",
                    "warning",
                    "物品源图只是候选证据，可能混有其他画面；床、粗链条这类资产需要人工确认或先提纯。",
                    evidence={"source_quality_status": source_quality_status, "track_status": track_status},
                ))
            if refinement_status == "failed":
                _mode2_asset_add_issue(asset, _mode2_asset_issue(
                    "prop_refinement_failed",
                    "warning",
                    f"物品提纯失败: {asset.get('refinement_error') or '未知错误'}",
                ))
            elif refinement_status == "needs_manual_review":
                _mode2_asset_add_issue(asset, _mode2_asset_issue(
                    "prop_refinement_needs_review",
                    "warning",
                    asset.get("refinement_warning") or "物品提纯结果需要人工核对。",
                ))
            quality_issue = _mode2_asset_quality_issue(asset)
            if quality_issue:
                _mode2_asset_add_issue(asset, quality_issue)

        issues = [
            issue for issue in (asset.get("asset_audit_issues") or [])
            if isinstance(issue, dict)
        ]
        status = "ignored" if manual_status == "ignored" else _mode2_asset_status_from_issues(issues)
        asset["asset_audit_status"] = status
        asset["asset_repair_action"] = _mode2_asset_repair_action(asset, issues)
        blocked = [issue for issue in issues if str(issue.get("severity") or "") == "blocked"]
        target_missing_blocked = any(
            str(issue.get("code") or "") in {"manual_replacement_missing", "role_target_image_missing"}
            for issue in blocked
        )
        contract_blocked = any(
            str(issue.get("code") or "") in {
                "scene_mixed_visual_groups",
                "scene_refinement_provenance_invalid",
            }
            for issue in blocked
        )
        asset["asset_quarantined"] = (
            target_missing_blocked
            or contract_blocked
            or (bool(blocked) and manual_status != "approved")
        )
        asset["asset_quarantine_reason"] = "; ".join(
            str(issue.get("message") or "").strip()
            for issue in blocked[:3]
            if str(issue.get("message") or "").strip()
        )
        if asset["asset_quarantined"]:
            asset["status"] = "needs_repair"
        elif manual_status == "approved" and target_image:
            asset["status"] = "ready"
        elif kind == "role" and target_image and manual_status != "needs_replacement":
            asset["status"] = "ready"
        blocking = [issue for issue in issues if str(issue.get("severity") or "") == "blocked"]
        warnings = [issue for issue in issues if str(issue.get("severity") or "") == "warning"]
        if blocking:
            asset["asset_audit_summary"] = f"{len(blocking)} 个阻断问题，{len(warnings)} 个警告"
        elif warnings:
            asset["asset_audit_summary"] = f"{len(warnings)} 个警告"
        elif status == "ignored":
            asset["asset_audit_summary"] = "已忽略"
        else:
            asset["asset_audit_summary"] = "未发现硬性问题"

    _attach_storyboard_asset_usage(assets, shots)
    project_config = _normalize_storyboard_project_config(data.get("project_config") or {})
    _compile_storyboard_prompts(assets, shots, project_config=project_config)
    counts = {"ok": 0, "warning": 0, "blocked": 0, "ignored": 0}
    for asset in assets:
        status = str(asset.get("asset_audit_status") or "ok")
        counts[status if status in counts else "warning"] += 1
    summary_status = "blocked" if counts["blocked"] else "warning" if counts["warning"] else "ok"
    summary = {
        "status": summary_status,
        "project_dir": str(root),
        "checked_at": time.time(),
        "total": len(assets),
        **counts,
        "rule_version": 2,
        "route": "mode2_seedance" if seedance_route else asset_route,
        "rules": [
            "target_image is the Seedance character reference gate",
            "original role/SAM3/face_detect images are identity evidence only",
            "role identity anchor/track readiness is warning-only for Seedance",
            "role identity evidence collision is warning-only for Seedance",
            "scene person-heavy source warning",
            "prop visual-check and mask-quality warning",
        ],
    }
    data["assets"] = assets
    data["shots"] = shots
    data["last_asset_audit"] = summary
    history = data.get("asset_audit_history") if isinstance(data.get("asset_audit_history"), list) else []
    history.append(summary)
    data["asset_audit_history"] = history[-50:]
    _refresh_mode2_structured_fields(root, data)
    _write_storyboard_mode2_store(root, data)
    result = _load_mode2_storyboard_result(root, _storyboard_mode2_asset_store_path(root))
    result["asset_audit"] = summary
    return result


def _update_storyboard_mode2_role_anchor(payload: dict[str, Any]) -> dict[str, Any]:
    project_dir = str(payload.get("project_dir") or "").strip()
    role_id = str(payload.get("role_id") or payload.get("asset_id") or "").strip()
    if not project_dir:
        raise ValueError("missing_project_dir")
    if not role_id:
        raise ValueError("missing_role_id")
    root = _resolve_storyboard_mode2_project_dir(project_dir)
    data = _load_storyboard_mode2_store(root)
    assets = [item for item in (data.get("assets") or []) if isinstance(item, dict)]
    role = next((item for item in assets if str(item.get("id") or "") == role_id and str(item.get("kind") or "") == "role"), None)
    if role is None:
        raise ValueError(f"role_not_found: {role_id}")

    delete = bool(payload.get("delete"))
    anchor_id = str(payload.get("anchor_id") or payload.get("id") or "").strip()
    annotations = [
        item for item in (data.get("identity_annotations") or [])
        if isinstance(item, dict)
    ]
    if delete:
        if not anchor_id:
            annotations = [item for item in annotations if str(item.get("role_id") or "") != role_id]
        else:
            annotations = [item for item in annotations if str(item.get("id") or "") != anchor_id]
    else:
        try:
            time_seconds = max(0.0, float(payload.get("time") or payload.get("time_seconds") or role.get("source_time") or 0.1))
        except (TypeError, ValueError):
            time_seconds = 0.1
        point = payload.get("point")
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("identity_point_required")
        px = max(0.0, min(1.0, float(point[0])))
        py = max(0.0, min(1.0, float(point[1])))
        if not anchor_id:
            anchor_id = f"anchor_{uuid.uuid4().hex[:10]}"
        anchor = {
            "id": anchor_id,
            "role_id": role_id,
            "role_name": str(role.get("name") or role_id),
            "time": round(time_seconds, 3),
            "point": [round(px, 6), round(py, 6)],
            "shot_id": str(payload.get("shot_id") or ""),
            "source_segment_id": str(payload.get("source_segment_id") or ""),
            "frame_path": str(payload.get("frame_path") or ""),
            "note": str(payload.get("note") or ""),
            "source": "manual",
            "updated_at": time.time(),
        }
        source_kind = str(payload.get("source_kind") or "").strip()
        if source_kind:
            anchor["source_kind"] = source_kind
        sam3_object_id = payload.get("sam3_object_id")
        if sam3_object_id not in (None, ""):
            try:
                anchor["sam3_object_id"] = int(sam3_object_id)
            except (TypeError, ValueError):
                anchor["sam3_object_id"] = str(sam3_object_id)
        for source_key, anchor_key in (
            ("mask_color", "mask_color"),
            ("mask_color_key", "mask_color"),
            ("mask_candidate_id", "mask_candidate_id"),
            ("mask_candidate_path", "mask_candidate_path"),
            ("candidate_mask_path", "mask_candidate_path"),
            ("candidate_overlay_path", "candidate_overlay_path"),
        ):
            value = str(payload.get(source_key) or "").strip()
            if value:
                anchor[anchor_key] = value
        object_center = payload.get("object_center_point") or payload.get("center_point")
        if isinstance(object_center, (list, tuple)) and len(object_center) == 2:
            try:
                anchor["object_center_point"] = [
                    round(max(0.0, min(1.0, float(object_center[0]))), 6),
                    round(max(0.0, min(1.0, float(object_center[1]))), 6),
                ]
            except (TypeError, ValueError):
                pass
        mask_color_rgb = payload.get("mask_color_rgb")
        if isinstance(mask_color_rgb, (list, tuple)) and len(mask_color_rgb) >= 3:
            try:
                anchor["mask_color_rgb"] = [
                    max(0, min(255, int(float(value))))
                    for value in list(mask_color_rgb)[:3]
                ]
            except (TypeError, ValueError):
                pass
        source_shape = _normalize_identity_shape(payload.get("source_shape") or payload.get("shape"))
        if source_shape:
            anchor["source_shape"] = source_shape
            anchor["shape"] = source_shape
        anchor_frame_path = str(anchor.get("frame_path") or "")
        anchor_time = float(anchor.get("time") or 0.0)

        def same_anchor_slot(item: dict[str, Any]) -> bool:
            if str(item.get("role_id") or "") != role_id:
                return False
            item_id = str(item.get("id") or "")
            if item_id and item_id == anchor_id:
                return True
            item_frame_path = str(item.get("frame_path") or "")
            if anchor_frame_path and item_frame_path and anchor_frame_path == item_frame_path:
                return True
            try:
                if abs(float(item.get("time") or 0.0) - anchor_time) < 0.05:
                    return True
            except (TypeError, ValueError):
                pass
            return False

        replaced = False
        for index, item in enumerate(annotations):
            if same_anchor_slot(item):
                annotations[index] = anchor
                replaced = True
                break
        if not replaced:
            annotations.append(anchor)

    role_anchors = sorted(
        [item for item in annotations if str(item.get("role_id") or "") == role_id],
        key=lambda item: (float(item.get("time") or 0.0), float(item.get("updated_at") or 0.0)),
    )
    role["identity_anchors"] = role_anchors
    role["identity_status"] = "annotated" if role_anchors else "needs_anchor"
    role["track_status"] = str(role.get("track_status") or "pending")
    data["assets"] = assets
    data["identity_annotations"] = annotations
    data["role_tracks_version"] = int(data.get("role_tracks_version") or 1)
    _write_storyboard_mode2_store(root, data)
    try:
        _audit_storyboard_mode2_assets({"project_dir": str(root)})
    except Exception as exc:  # noqa: BLE001
        logging.warning("Mode2 asset audit skipped after role anchor update: %s", exc)
    result = _load_mode2_storyboard_result(root, _storyboard_mode2_asset_store_path(root))
    result["updated_role_id"] = role_id
    return result


def _storyboard_mode2_role_mask_candidates(payload: dict[str, Any]) -> dict[str, Any]:
    project_dir = str(payload.get("project_dir") or "").strip()
    if not project_dir:
        raise ValueError("missing_project_dir")
    root = _resolve_storyboard_mode2_project_dir(project_dir)
    data = _load_storyboard_mode2_store(root)
    assets = [item for item in (data.get("assets") or []) if isinstance(item, dict)]
    project_config = _normalize_storyboard_project_config(data.get("project_config") or {})
    prefer_remote = project_config.get("mask_source") == "remote_sam3_color"
    asset_id = str(payload.get("asset_id") or payload.get("role_id") or "").strip()
    asset = next((item for item in assets if str(item.get("id") or "") == asset_id), {}) if asset_id else {}

    video_path = str(payload.get("video_path") or data.get("video_path") or asset.get("source_video_path") or "").strip()
    frame_path_value = str(payload.get("frame_path") or "").strip()
    try:
        time_seconds = float(
            payload.get("time")
            or payload.get("time_seconds")
            or asset.get("source_time")
            or 0.1
        )
    except (TypeError, ValueError):
        time_seconds = 0.1
    if video_path and Path(video_path).exists():
        time_seconds = _clamp_video_time(video_path, time_seconds)

    cache_key = hashlib.md5(
        "|".join([
            str(Path(video_path)) if video_path else "",
            frame_path_value,
            f"{time_seconds:.3f}",
            asset_id,
        ]).encode("utf-8", errors="ignore")
    ).hexdigest()[:12]
    time_ms = int(round(max(0.0, time_seconds) * 1000))
    candidate_id = f"maskcand_{time_ms:08d}_{cache_key}"
    cache_dir = root / "assets" / "identity_candidates" / candidate_id
    result_path = cache_dir / "result.json"
    force = bool(payload.get("force"))
    cached_fallback: dict[str, Any] | None = None

    if not force and result_path.exists():
        try:
            cached = json.loads(result_path.read_text(encoding="utf-8-sig"))
            colored_path = str(cached.get("colored_mask") or "").strip()
            if (
                isinstance(cached, dict)
                and Path(str(cached.get("original_frame") or "")).exists()
                and colored_path
                and Path(colored_path).exists()
            ):
                cached["cached"] = True
                candidate_source = str(cached.get("candidate_source") or "").strip()
                if not prefer_remote or candidate_source == "remote_sam3_color":
                    return cached
                cached_fallback = cached
        except Exception:  # noqa: BLE001
            pass

    cache_dir.mkdir(parents=True, exist_ok=True)
    original_frame = cache_dir / "original.jpg"
    if video_path and Path(video_path).exists():
        extract_frame(video_path, time_seconds, original_frame)
    elif frame_path_value and Path(frame_path_value).exists():
        shutil.copyfile(frame_path_value, original_frame)
    else:
        raise ValueError(f"source_video_or_frame_not_found: {video_path or frame_path_value}")

    result: dict[str, Any] = {
        "ok": False,
        "project_dir": str(root),
        "candidate_id": candidate_id,
        "asset_id": asset_id,
        "time": round(time_seconds, 3),
        "video_path": video_path,
        "source_frame_path": frame_path_value,
        "original_frame": str(original_frame),
        "colored_mask": "",
        "overlay_image": "",
        "objects": [],
        "cached": False,
        "candidate_source": "",
        "preferred_source": "remote_sam3_color" if prefer_remote else "local_sam3",
        "fallback_used": False,
        "fallback_reason": "",
        "mask_status": "failed",
        "error": "",
    }
    if not video_path or not Path(video_path).exists():
        result["error"] = "source_video_missing; only original frame is available"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return result

    remote_error = ""
    if prefer_remote:
        try:
            result.update(_build_storyboard_mode2_remote_role_mask_candidates(
                root=root,
                assets=assets,
                asset_id=asset_id,
                video_path=video_path,
                time_seconds=time_seconds,
                original_frame=original_frame,
                cache_dir=cache_dir,
                candidate_id=candidate_id,
            ))
            result["ok"] = True
            result["mask_status"] = "ready"
            result["candidate_source"] = "remote_sam3_color"
        except Exception as exc:  # noqa: BLE001
            remote_error = str(exc)
            logging.warning("Mode2 remote SAM3 role mask candidate failed, fallback local: %s", exc)

    if not result.get("ok"):
        if cached_fallback:
            cached_fallback["cached"] = True
            cached_fallback["preferred_source"] = "remote_sam3_color" if prefer_remote else "local_sam3"
            cached_fallback["fallback_used"] = bool(prefer_remote)
            cached_fallback["fallback_reason"] = remote_error or "remote_sam3_color_unavailable"
            return cached_fallback
        try:
            result.update(_build_storyboard_mode2_role_mask_candidates(
                video_path=video_path,
                time_seconds=time_seconds,
                original_frame=original_frame,
                cache_dir=cache_dir,
                candidate_id=candidate_id,
            ))
            result["ok"] = True
            result["mask_status"] = "ready"
            result["candidate_source"] = "local_sam3"
            if prefer_remote:
                result["fallback_used"] = True
                result["fallback_reason"] = remote_error or "remote_sam3_color_unavailable"
        except Exception as exc:  # noqa: BLE001
            if remote_error:
                result["remote_error"] = remote_error
            raise exc

    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return result


def _storyboard_mode2_role_pairs_for_remote_mask(
    assets: list[dict[str, Any]],
    *,
    asset_id: str,
    time_seconds: float,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    role_assets = [item for item in assets if str(item.get("kind") or "") == "role"]
    role_assets.sort(key=lambda item: 0 if str(item.get("id") or "") == asset_id else 1)
    seen_refs: set[str] = set()
    for role in role_assets:
        role_id = str(role.get("id") or "").strip()
        ref_image = str(role.get("target_image") or role.get("seedance_reference_image") or "").strip()
        if not ref_image or not Path(ref_image).exists():
            continue
        ref_key = str(Path(ref_image)).lower()
        if ref_key in seen_refs:
            continue
        seen_refs.add(ref_key)
        anchors = [item for item in (role.get("identity_anchors") or []) if isinstance(item, dict)]
        preferred = min(
            anchors,
            key=lambda item: abs(float(item.get("time") or time_seconds) - float(time_seconds)),
            default={},
        )
        source_shape = _normalize_identity_shape(preferred.get("source_shape") or preferred.get("shape"))
        source_point = preferred.get("point") if isinstance(preferred.get("point"), (list, tuple)) else None
        pairs.append({
            "name": str(role.get("name") or role.get("tag") or role_id or f"角色{len(pairs) + 1}"),
            "ref_image": ref_image,
            "source_shape": source_shape,
            "source_point": source_point,
        })
        if len(pairs) >= len(WAN22_MASK_COLOR_KEYS):
            break
    return pairs


def _extract_first_video_frame_to_png(video_path: str | Path, output_path: str | Path) -> None:
    import cv2

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"cannot_read_remote_mask_video: {video_path}")
        cv2.imwrite(str(output), frame)
    finally:
        capture.release()
    if not output.exists() or output.stat().st_size <= 0:
        raise ValueError(f"remote_mask_frame_missing: {output}")


def _storyboard_mode2_objects_from_colored_mask(mask_path: Path, cache_dir: Path) -> list[dict[str, Any]]:
    import cv2
    import numpy as np

    image = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot_read_colored_mask: {mask_path}")
    height, width = image.shape[:2]
    palette_rgb: dict[str, tuple[int, int, int]] = {
        "blue": (40, 120, 255),
        "red": (255, 64, 64),
        "green": (52, 211, 153),
        "magenta": (216, 80, 255),
        "cyan": (45, 212, 255),
        "yellow": (250, 204, 21),
    }
    color_labels = {
        "blue": "蓝色",
        "red": "红色",
        "green": "绿色",
        "magenta": "紫色",
        "cyan": "青色",
        "yellow": "黄色",
    }
    object_dir = cache_dir / "objects"
    object_dir.mkdir(parents=True, exist_ok=True)
    objects: list[dict[str, Any]] = []
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    min_area = max(16, int(width * height * 0.001))
    for index, color_key in enumerate(WAN22_MASK_COLOR_KEYS):
        color_rgb = np.array(palette_rgb[color_key], dtype=np.float32)
        distance = np.sqrt(np.sum((image_rgb - color_rgb) ** 2, axis=2))
        selected = distance < 90
        area = int(np.count_nonzero(selected))
        if area < min_area:
            continue
        ys, xs = np.where(selected)
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        cx = float(xs.mean()) / float(width)
        cy = float(ys.mean()) / float(height)
        object_mask_path = object_dir / f"{color_key}_remote_object_{index}_mask.png"
        cv2.imwrite(str(object_mask_path), selected.astype(np.uint8) * 255)
        objects.append({
            "object_id": index,
            "color_key": color_key,
            "color_label": color_labels.get(color_key, color_key),
            "color_rgb": list(palette_rgb[color_key]),
            "bbox": [
                round(x1 / width, 6),
                round(y1 / height, 6),
                round((x2 - x1) / width, 6),
                round((y2 - y1) / height, 6),
            ],
            "bbox_xywh": [
                round(x1 / width, 6),
                round(y1 / height, 6),
                round((x2 - x1) / width, 6),
                round((y2 - y1) / height, 6),
            ],
            "bbox_mode": "xywh",
            "center_point": [round(cx, 6), round(cy, 6)],
            "area_ratio": round(area / float(max(1, width * height)), 5),
            "score": 0.0,
            "mask_path": str(object_mask_path),
            "crop_path": "",
        })
    if not objects:
        raise ValueError("remote_colored_mask_has_no_palette_objects")
    return objects


def _build_storyboard_mode2_remote_role_mask_candidates(
    *,
    root: Path,
    assets: list[dict[str, Any]],
    asset_id: str,
    video_path: str,
    time_seconds: float,
    original_frame: Path,
    cache_dir: Path,
    candidate_id: str,
) -> dict[str, Any]:
    from spvideo.ffmpeg_tools import probe_video
    from spvideo.scail2_client import Scail2Client

    role_pairs = _storyboard_mode2_role_pairs_for_remote_mask(
        assets,
        asset_id=asset_id,
        time_seconds=time_seconds,
    )
    if not role_pairs:
        raise ValueError("remote_sam3_requires_role_target_images")
    meta = probe_video(video_path)
    source_fps = max(1, int(round(float(meta.fps or 24.0))))
    source_frame_idx = max(0, int(round(float(time_seconds or 0.0) * source_fps)))
    clip_path, prompt_frame, clip_start_frame, clip_frames, track_fps = _make_sam3_window_clip(
        video_path,
        frame_idx=source_frame_idx,
        max_frames=1,
        job_id=f"{candidate_id}_remote",
        max_side=720,
    )
    client = Scail2Client()
    result = client.inspect_masks(
        video_path=str(clip_path),
        ref_images=[str(pair["ref_image"]) for pair in role_pairs],
        role_names=[str(pair["name"]) for pair in role_pairs],
        video_window={
            "force_rate": max(1, int(round(track_fps or source_fps))),
            "frame_load_cap": 1,
            "skip_first_frames": 0,
            "select_every_nth": 1,
        },
        sampler_preset="fast",
        normalize_size=True,
        sam_text="person",
        strict_track_preflight=False,
        source_identity_points=[
            pair.get("source_point") if isinstance(pair.get("source_point"), (list, tuple)) else None
            for pair in role_pairs
        ],
        source_identity_shapes=[
            pair.get("source_shape") if isinstance(pair.get("source_shape"), dict) else None
            for pair in role_pairs
        ],
    )
    output_path = str(result.get("output_path") or "").strip()
    if not output_path or not Path(output_path).exists():
        raise ValueError("remote_sam3_did_not_return_colored_mask_video")
    colored_mask_path = cache_dir / "remote_colored_mask.png"
    _extract_first_video_frame_to_png(output_path, colored_mask_path)
    objects = _storyboard_mode2_objects_from_colored_mask(colored_mask_path, cache_dir)
    return {
        "colored_mask": str(colored_mask_path),
        "overlay_image": str(colored_mask_path),
        "objects": objects,
        "remote_output_path": output_path,
        "remote_result": {
            "prompt_id": result.get("prompt_id"),
            "workflow_path": result.get("workflow_path"),
            "role_names": result.get("role_names"),
            "mask_output_paths": result.get("mask_output_paths"),
            "warnings": result.get("warnings") or [],
        },
        "sam3": {
            "source": "remote_sam3_color",
            "object_count": len(objects),
            "role_count": len(role_pairs),
            "clip_path": str(clip_path),
            "source_frame_idx": source_frame_idx,
            "clip_start_frame": clip_start_frame,
            "prompt_frame": prompt_frame,
            "clip_frames": clip_frames,
            "force_rate": max(1, int(round(track_fps or source_fps))),
        },
    }


def _build_storyboard_mode2_role_mask_candidates(
    *,
    video_path: str,
    time_seconds: float,
    original_frame: Path,
    cache_dir: Path,
    candidate_id: str,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    from spvideo.ffmpeg_tools import probe_video
    from spvideo.sam3_tracker import SAM3Tracker

    frame = cv2.imread(str(original_frame), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"cannot_read_original_frame: {original_frame}")
    height, width = frame.shape[:2]
    meta = probe_video(video_path)
    source_fps = float(meta.fps or 0.0) or 24.0
    source_frame_idx = max(0, int(round(float(time_seconds or 0.0) * source_fps)))
    job_token = f"{candidate_id}_{uuid.uuid4().hex[:6]}"

    with SAM3_TRACK_LOCK:
        clip_path, prompt_frame, start_frame, clip_frames, track_fps = _make_sam3_window_clip(
            video_path,
            frame_idx=source_frame_idx,
            max_frames=1,
            job_id=job_token,
            max_side=720,
        )
        tracker = SAM3Tracker()
        try:
            tracked = tracker.track_text_objects(
                video_path=str(clip_path),
                text_prompt="person",
                frame_idx=prompt_frame,
                max_frames=clip_frames,
                propagation_direction="both",
            )
        finally:
            tracker.close()

    object_ids = [int(value) for value in tracked.get("object_ids") or []]
    if not object_ids:
        raise ValueError("SAM3 did not find any person objects in this frame")

    raw_objects: list[dict[str, Any]] = []
    prompt_index = max(0, min(int(tracked.get("num_frames") or 1) - 1, int(prompt_frame)))
    min_area = max(16, int(width * height * 0.001))
    for obj_id in object_ids:
        object_result = (tracked.get("objects") or {}).get(obj_id) or {}
        masks = object_result.get("masks") or []
        if prompt_index >= len(masks):
            continue
        mask = masks[prompt_index]
        if mask is None:
            continue
        mask_array = np.squeeze(mask).astype(np.uint8)
        if mask_array.ndim != 2 or not np.any(mask_array):
            continue
        if mask_array.shape[:2] != (height, width):
            mask_array = cv2.resize(mask_array, (width, height), interpolation=cv2.INTER_NEAREST)
        selected = mask_array > 0
        area = int(np.count_nonzero(selected))
        if area < min_area:
            continue
        ys, xs = np.where(selected)
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        cx = float(xs.mean()) / float(width)
        cy = float(ys.mean()) / float(height)
        raw_objects.append({
            "object_id": obj_id,
            "mask": selected,
            "area": area,
            "area_ratio": area / float(max(1, width * height)),
            "bbox": [
                round(x1 / width, 6),
                round(y1 / height, 6),
                round((x2 - x1) / width, 6),
                round((y2 - y1) / height, 6),
            ],
            "center_point": [round(cx, 6), round(cy, 6)],
            "score": float((object_result.get("scores") or [0.0])[prompt_index] or 0.0)
            if prompt_index < len(object_result.get("scores") or []) else 0.0,
            "bounds_px": (x1, y1, x2, y2),
        })

    if not raw_objects:
        raise ValueError("SAM3 person objects were too small or empty in this frame")

    raw_objects.sort(key=lambda item: (float(item["center_point"][0]), -float(item["area_ratio"])))
    raw_objects = raw_objects[:len(WAN22_MASK_COLOR_KEYS)]
    palette_rgb: dict[str, tuple[int, int, int]] = {
        "blue": (40, 120, 255),
        "red": (255, 64, 64),
        "green": (52, 211, 153),
        "magenta": (216, 80, 255),
        "cyan": (45, 212, 255),
        "yellow": (250, 204, 21),
    }
    color_labels = {
        "blue": "蓝色",
        "red": "红色",
        "green": "绿色",
        "magenta": "紫色",
        "cyan": "青色",
        "yellow": "黄色",
    }

    colored = np.zeros((height, width, 3), dtype=np.uint8)
    overlay = (frame.astype(np.float32) * 0.58).astype(np.uint8)
    objects: list[dict[str, Any]] = []
    object_dir = cache_dir / "objects"
    object_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(raw_objects):
        color_key = WAN22_MASK_COLOR_KEYS[index]
        color_rgb = palette_rgb[color_key]
        color_bgr = np.array([color_rgb[2], color_rgb[1], color_rgb[0]], dtype=np.uint8)
        selected = item["mask"]
        colored[selected] = color_bgr
        overlay[selected] = (
            frame[selected].astype(np.float32) * 0.36
            + color_bgr.astype(np.float32) * 0.64
        ).astype(np.uint8)

        object_mask_path = object_dir / f"{color_key}_object_{item['object_id']}_mask.png"
        cv2.imwrite(str(object_mask_path), selected.astype(np.uint8) * 255)
        x1, y1, x2, y2 = item["bounds_px"]
        pad_x = max(8, int((x2 - x1) * 0.18))
        pad_y = max(8, int((y2 - y1) * 0.14))
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(width, x2 + pad_x)
        cy2 = min(height, y2 + pad_y)
        crop_path = object_dir / f"{color_key}_object_{item['object_id']}_crop.jpg"
        if cx2 > cx1 and cy2 > cy1:
            cv2.imwrite(str(crop_path), frame[cy1:cy2, cx1:cx2], [cv2.IMWRITE_JPEG_QUALITY, 92])
        objects.append({
            "object_id": int(item["object_id"]),
            "color_key": color_key,
            "color_label": color_labels.get(color_key, color_key),
            "color_rgb": list(color_rgb),
            "bbox": item["bbox"],
            "bbox_xywh": item["bbox"],
            "bbox_mode": "xywh",
            "center_point": item["center_point"],
            "area_ratio": round(float(item["area_ratio"]), 5),
            "score": round(float(item["score"]), 4),
            "mask_path": str(object_mask_path),
            "crop_path": str(crop_path) if crop_path.exists() else "",
        })

    colored_mask_path = cache_dir / "colored_mask.png"
    overlay_path = cache_dir / "overlay.jpg"
    cv2.imwrite(str(colored_mask_path), colored)
    cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])

    return {
        "colored_mask": str(colored_mask_path),
        "overlay_image": str(overlay_path),
        "objects": objects,
        "sam3": {
            "clip_path": str(clip_path),
            "source_frame_idx": source_frame_idx,
            "clip_start_frame": start_frame,
            "clip_frames": clip_frames,
            "prompt_frame": prompt_frame,
            "track_fps": track_fps,
            "object_count": len(objects),
        },
    }


def _update_storyboard_mode2_asset(payload: dict[str, Any]) -> dict[str, Any]:
    project_dir = str(payload.get("project_dir") or "").strip()
    asset_id = str(payload.get("asset_id") or payload.get("id") or "").strip()
    if not project_dir:
        raise ValueError("missing_project_dir")
    if not asset_id:
        raise ValueError("missing_asset_id")

    root = _resolve_storyboard_mode2_project_dir(project_dir)
    store_path = _storyboard_mode2_asset_store_path(root)
    if not store_path.exists():
        raise ValueError(f"mode2 storyboard asset store not found: {store_path}")

    data = json.loads(store_path.read_text(encoding="utf-8-sig"))
    assets = [
        item for item in (data.get("assets") or [])
        if isinstance(item, dict)
    ]
    shots = [
        item for item in (data.get("shots") or [])
        if isinstance(item, dict)
    ]
    target = next((item for item in assets if str(item.get("id") or "") == asset_id), None)
    if target is None:
        raise ValueError(f"asset_not_found: {asset_id}")

    updates = payload.get("updates") if isinstance(payload.get("updates"), dict) else payload
    allowed_string_fields = {
        "target_image",
        "prompt",
        "manual_asset_status",
        "manual_note",
        "source_usage_role",
    }
    for field in allowed_string_fields:
        if field in updates:
            target[field] = str(updates.get(field) or "").strip()

    manual_status = str(target.get("manual_asset_status") or "").strip()
    valid_statuses = {"", "pending", "approved", "needs_replacement", "shot_reference", "ignored"}
    if manual_status not in valid_statuses:
        raise ValueError(f"invalid_manual_asset_status: {manual_status}")
    if manual_status == "approved":
        target["status"] = "ready"
    elif manual_status == "needs_replacement":
        target["status"] = "needs_replacement"
    elif manual_status == "shot_reference":
        target["status"] = "shot_reference"
        target["source_usage_role"] = "shot_reference"
    elif manual_status == "ignored":
        target["status"] = "ignored"
    elif str(target.get("target_image") or "").strip():
        target["status"] = "ready"
    elif str(target.get("status") or "") not in {"needs_identification", "ignored"}:
        target["status"] = "pending"

    target["manual_updated_at"] = time.time()
    data["assets"] = assets
    data["shots"] = shots
    data.setdefault("asset_manual_history", [])
    if isinstance(data["asset_manual_history"], list):
        data["asset_manual_history"].append({
            "asset_id": asset_id,
            "manual_asset_status": manual_status,
            "target_image": str(target.get("target_image") or ""),
            "updated_at": target["manual_updated_at"],
        })
        data["asset_manual_history"] = data["asset_manual_history"][-200:]

    project_config = _normalize_storyboard_project_config(data.get("project_config") or {})
    _compile_storyboard_prompts(assets, shots, project_config=project_config)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    try:
        _audit_storyboard_mode2_assets({"project_dir": str(root)})
    except Exception as exc:  # noqa: BLE001
        logging.warning("Mode2 asset audit skipped after manual update: %s", exc)
    result = _load_mode2_storyboard_result(root, store_path)
    result["updated_asset_id"] = asset_id
    return result


def _storyboard_manual_asset_kind(value: Any) -> str:
    text = str(value or "").strip().lower()
    mapping = {
        "role": "role",
        "character": "role",
        "person": "role",
        "角色": "role",
        "人物": "role",
        "scene": "scene",
        "background": "scene",
        "location": "scene",
        "场景": "scene",
        "背景": "scene",
        "prop": "prop",
        "object": "prop",
        "item": "prop",
        "物品": "prop",
        "道具": "prop",
    }
    if text not in mapping:
        raise ValueError(f"invalid_asset_kind: {value}")
    return mapping[text]


def _next_storyboard_manual_asset_id(assets: list[dict[str, Any]], kind: str) -> str:
    prefix = {"role": "role", "scene": "scene", "prop": "prop"}.get(kind, "asset")
    max_index = 0
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$", re.I)
    for asset in assets:
        asset_id = str(asset.get("id") or "").strip()
        match = pattern.match(asset_id)
        if not match:
            continue
        try:
            max_index = max(max_index, int(match.group(1)))
        except ValueError:
            continue
    return f"{prefix}_{max_index + 1}"


def _unique_manual_asset_path(target_dir: Path, stem: str, suffix: str) -> Path:
    safe_stem = _mode2_safe_asset_slug(stem, fallback="manual_asset")
    clean_suffix = suffix.lower() if suffix else ".png"
    if clean_suffix == ".jpe":
        clean_suffix = ".jpg"
    if clean_suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
        clean_suffix = ".png"
    target = target_dir / f"{safe_stem}{clean_suffix}"
    index = 2
    while target.exists():
        target = target_dir / f"{safe_stem}_{index}{clean_suffix}"
        index += 1
    return target


def _save_manual_asset_data_url(value: Any, target_dir: Path, stem: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    mime = "image/png"
    payload = text
    if text.startswith("data:"):
        if "," not in text:
            raise ValueError("invalid_image_data_url")
        header, payload = text.split(",", 1)
        match = re.match(r"^data:([^;,]+)", header, re.I)
        mime = (match.group(1).lower() if match else mime).strip()
    if not mime.startswith("image/"):
        raise ValueError("manual_asset_image_required")
    raw = base64.b64decode(payload, validate=True)
    if len(raw) > 25 * 1024 * 1024:
        raise ValueError("manual_asset_image_too_large")
    suffix = mimetypes.guess_extension(mime) or ".png"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = _unique_manual_asset_path(target_dir, stem, suffix)
    target.write_bytes(raw)
    return str(target)


def _copy_manual_asset_image(value: Any, target_dir: Path, stem: str) -> str:
    text = str(value or "").strip().strip('"')
    if not text:
        return ""
    source = Path(text)
    if not source.exists() or not source.is_file():
        raise ValueError(f"manual_asset_image_not_found: {text}")
    guessed_type = mimetypes.guess_type(str(source))[0] or ""
    if guessed_type and not guessed_type.startswith("image/"):
        raise ValueError("manual_asset_image_required")
    suffix = source.suffix.lower() or ".png"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = _unique_manual_asset_path(target_dir, stem, suffix)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return str(target)


def _manual_asset_default_store(root: Path, video_path: str) -> dict[str, Any]:
    now = time.time()
    return {
        "version": 2,
        "job_id": "manual_asset_library",
        "project_dir": str(root),
        "reference_project_dir": "",
        "video_path": str(video_path or ""),
        "reference_strategy": "new_only",
        "project_config": copy.deepcopy(DEFAULT_STORYBOARD_PROJECT_CONFIG),
        "created_at": now,
        "understanding": {
            "status": "manual",
            "summary": "手动资产库，不使用智能反推。",
            "characters": [],
            "scenes": [],
            "boundary_hints": [],
            "source": "manual_asset_library",
            "source_path": str(video_path or ""),
            "project_dir": str(root),
            "note": "用户从原视频手动截图并上传参考图。",
            "asset_manifest": {"results": []},
            "analysis_frame_manifest": [],
        },
        "asset_manifest": {"results": []},
        "analysis_frame_manifest": [],
        "raw_visual_segments": [],
        "visual_segments": [],
        "semantic_segments": [],
        "reference_segments": [],
        "auto_director": {},
        "assets": [],
        "shots": [],
        "asset_stage": "manual_asset_library",
        "timeline_stage": "manual_asset_library",
        "storyboard_stage": "manual_asset_library",
        "mode_boundary": "Mode2 only; no Mode1 transfer/render dependency.",
    }


def _create_storyboard_mode2_asset(payload: dict[str, Any]) -> dict[str, Any]:
    project_dir = str(payload.get("project_dir") or "").strip()
    if not project_dir:
        raise ValueError("missing_project_dir")
    root = _resolve_storyboard_mode2_project_dir(project_dir)
    store_path = _storyboard_mode2_asset_store_path(root)
    video_path = str(payload.get("video_path") or "").strip()

    if store_path.exists():
        data = json.loads(store_path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("mode2 storyboard asset store is invalid")
    else:
        data = _manual_asset_default_store(root, video_path)

    if not video_path:
        video_path = str(data.get("video_path") or (data.get("meta") or {}).get("source_path") or "").strip()
    if video_path:
        data["video_path"] = video_path
        if isinstance(data.get("understanding"), dict):
            data["understanding"]["source_path"] = video_path

    kind = _storyboard_manual_asset_kind(payload.get("kind") or "role")
    assets = [item for item in (data.get("assets") or []) if isinstance(item, dict)]
    shots = [item for item in (data.get("shots") or []) if isinstance(item, dict)]
    asset_id = str(payload.get("asset_id") or payload.get("id") or "").strip() or _next_storyboard_manual_asset_id(assets, kind)
    if any(str(item.get("id") or "") == asset_id for item in assets):
        raise ValueError(f"asset_already_exists: {asset_id}")

    now = time.time()
    name = str(payload.get("name") or "").strip() or {
        "role": "新角色",
        "scene": "新场景",
        "prop": "新物品",
    }[kind]
    tag = str(payload.get("tag") or "").strip()
    prompt = str(payload.get("prompt") or "").strip()
    manual_note = str(payload.get("manual_note") or payload.get("note") or "").strip()
    try:
        source_time = max(0.0, float(payload.get("source_time") or payload.get("time") or 0.0))
    except (TypeError, ValueError):
        source_time = 0.0

    manual_dir = root / "assets" / "manual_assets" / asset_id
    source_image = _save_manual_asset_data_url(
        payload.get("source_image_data") or payload.get("source_frame_data") or "",
        manual_dir,
        f"{asset_id}_source",
    )
    if not source_image:
        source_image = _copy_manual_asset_image(
            payload.get("source_image") or payload.get("source_image_path") or "",
            manual_dir,
            f"{asset_id}_source",
        )
    target_image = _save_manual_asset_data_url(
        payload.get("target_image_data") or payload.get("upload_image_data") or "",
        manual_dir,
        f"{asset_id}_target",
    )
    if not target_image:
        target_image = _copy_manual_asset_image(
            payload.get("target_image") or payload.get("target_image_path") or "",
            manual_dir,
            f"{asset_id}_target",
        )

    crop_rect = payload.get("source_rect") if isinstance(payload.get("source_rect"), dict) else {}
    shot_id = str(payload.get("shot_id") or payload.get("segment_id") or "").strip()
    source_segment_ids = [shot_id] if shot_id and bool(payload.get("bind_to_shot")) else []
    keyframes = []
    if source_image:
        keyframes.append({
            "time": round(source_time, 3),
            "path": source_image,
            "crop_path": source_image,
            "reason": "手动截图",
            "source": "manual_capture",
        })

    asset = {
        "id": asset_id,
        "kind": kind,
        "name": name,
        "tag": tag,
        "alias": tag,
        "prompt": prompt,
        "manual_note": manual_note,
        "manual_asset_status": "approved" if target_image else "needs_replacement",
        "manual_asset": True,
        "manual_created_at": now,
        "manual_updated_at": now,
        "source_kind": "manual_capture",
        "source_usage_role": "manual_asset",
        "source_video_path": video_path,
        "source_time": round(source_time, 3),
        "source_image": source_image,
        "source_images": [source_image] if source_image else [],
        "source_rect": crop_rect,
        "source_segment_ids": source_segment_ids,
        "candidate_source_image": source_image,
        "candidate_source_images": [source_image] if source_image else [],
        "representative_image": source_image or target_image,
        "representative_label": "手动截图",
        "representative_status": "manual",
        "representative_is_clean": True,
        "target_image": target_image,
        "target_asset": {
            "active_image": target_image,
            "source": "manual_upload",
            "updated_at": now,
        } if target_image else {},
        "keyframes": keyframes,
        "status": "ready" if target_image else "needs_replacement",
        "created_at": now,
        "updated_at": now,
    }
    if kind == "scene":
        asset["source_visual_status"] = "manual_scene_reference"
        asset["source_trust_level"] = "manual"
    elif kind == "prop":
        asset["source_visual_status"] = "manual_prop_reference"
        asset["source_trust_level"] = "manual"
    else:
        asset["identity_status"] = "manual"
        asset["track_status"] = "manual"

    assets.append(asset)
    data["assets"] = assets
    data["shots"] = shots
    history = data.setdefault("asset_manual_history", [])
    if isinstance(history, list):
        history.append({
            "asset_id": asset_id,
            "action": "create",
            "kind": kind,
            "name": name,
            "source_image": source_image,
            "target_image": target_image,
            "updated_at": now,
        })
        data["asset_manual_history"] = history[-200:]

    project_config = _normalize_storyboard_project_config(data.get("project_config") or {})
    _compile_storyboard_prompts(assets, shots, project_config=project_config)
    _refresh_mode2_structured_fields(root, data)
    _write_storyboard_mode2_store(root, data)
    result = _load_mode2_storyboard_result(root, store_path)
    result["created_asset_id"] = asset_id
    return result


def _load_project_result(project_dir: str | Path) -> dict[str, Any]:
    if not str(project_dir).strip():
        raise ValueError("missing_project_dir")
    root = resolve_auto_director_project_root(project_dir)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"项目目录不存在: {project_dir}")
    mode2_store = root / "assets" / "storyboard_assets.json"
    if mode2_store.exists():
        return _load_mode2_storyboard_result(root, mode2_store)
    if not (
        (root / "manifest.json").exists()
        or (root / "01_分析探针" / "two_pass_result.json").exists()
        or (root / "01_probe" / "two_pass_result.json").exists()
        or (root / "assets" / "storyboard_assets.json").exists()
    ):
        fallback_root = _latest_result_child(root)
        if fallback_root is not None:
            root = fallback_root
            mode2_store = root / "assets" / "storyboard_assets.json"
            if mode2_store.exists():
                return _load_mode2_storyboard_result(root, mode2_store)

    manifest_path = root / "manifest.json"
    probe_dir = root / "01_分析探针"
    clips_dir = root / "02_分镜片段" / "00_all_mp4_clips"
    report_path = probe_dir / "segmentation_report.html"

    visual_merge = None
    visual_merge_path = probe_dir / "visual_merge" / "visual_merge.json"
    if visual_merge_path.exists():
        try:
            visual_merge = json.loads(visual_merge_path.read_text(encoding="utf-8"))
        except Exception:
            visual_merge = None

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        segments = manifest.get("segments", [])
        frames = manifest.get("frames", [])
        meta = manifest.get("meta", {})
    else:
        two_pass_path = probe_dir / "two_pass_result.json"
        if not two_pass_path.exists():
            raise ValueError("未找到 manifest.json 或 01_分析探针/two_pass_result.json")
        two_pass = json.loads(two_pass_path.read_text(encoding="utf-8"))
        raw_segments = two_pass.get("sub_segments", [])
        segments = []
        for idx, item in enumerate(raw_segments, 1):
            seg_id = f"{idx:03d}"
            start = float(item.get("start", 0.0))
            end = float(item.get("end", start))
            clip = next(clips_dir.glob(f"{seg_id}_*.mp4"), None) if clips_dir.exists() else None
            segments.append({
                "segment_id": seg_id,
                "start": start,
                "end": end,
                "duration": max(0.0, end - start),
                "person_count": item.get("person_count", -1),
                "start_sources": item.get("start_sources", []),
                "end_sources": item.get("end_sources", []),
                "output_path": str(clip) if clip else "",
            })
        meta = {}
        frames = []

    background_count = sum(1 for seg in segments if seg.get("person_count") == 0)
    auto_director_plan = load_auto_director(root)
    return {
        "project_dir": str(root),
        "clips_dir": str(clips_dir),
        "report_path": str(report_path) if report_path.exists() else "",
        "segment_count": len(segments),
        "background_count": background_count,
        "visual_merge": visual_merge or {},
        "sam3_finalize": None,
        "segments": segments,
        "frames": frames,
        "meta": meta,
        "auto_director": auto_director_plan,
    }

def _edit_segment(
    project_dir: str,
    segment_id: str,
    action: str,
) -> dict[str, Any]:
    """合并或删除片段,更新 manifest 和 clips."""
    root = Path(project_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("manifest.json 不存在")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    segments: list[dict[str, Any]] = list(manifest.get("segments", []))

    idx = None
    for i, seg in enumerate(segments):
        if str(seg.get("segment_id") or "") == segment_id:
            idx = i
            break
    if idx is None:
        raise ValueError(f"未找到片段: {segment_id}")

    source_pointer = root / "00_原始视频" / "source_pointer.json"
    if not source_pointer.exists():
        raise ValueError("找不到原视频引用")
    source_info = json.loads(source_pointer.read_text(encoding="utf-8"))
    video_path = str(source_info.get("source_path") or "")
    if not video_path or not Path(video_path).exists():
        raise ValueError("原视频文件不存在")

    clips_dir = root / "02_分镜片段" / "00_all_mp4_clips"

    if action == "delete":
        target = segments[idx]
        out_path = target.get("output_path") or ""
        if out_path and Path(out_path).exists():
            Path(out_path).unlink()
        segments.pop(idx)
        manifest["segments"] = segments
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return _load_project_result(project_dir)

    def _build_merged_seg(prev: dict[str, Any], nxt: dict[str, Any]) -> dict[str, Any]:
        new_start = float(prev.get("start", 0))
        new_end = float(nxt.get("end", 0))
        new_id = f"{prev.get('segment_id')}_{nxt.get('segment_id')}"
        return {
            "segment_id": new_id,
            "start": new_start,
            "end": new_end,
            "segment_type": "with_human" if (prev.get("segment_type") == "with_human" or nxt.get("segment_type") == "with_human") else "without_human",
            "confidence": round(min(float(prev.get("confidence", 0.5)), float(nxt.get("confidence", 0.5))), 3),
            "output_path": "",
            "representative_frame": None,
            "detected": list({str(v) for v in (prev.get("detected", []) + nxt.get("detected", []))}),
            "recommended_tech": "",
            "needs_ai_driver": bool(prev.get("needs_ai_driver")) or bool(nxt.get("needs_ai_driver")),
            "needs_manual_check": bool(prev.get("needs_manual_check")) or bool(nxt.get("needs_manual_check")),
            "notes": list({str(v) for v in (prev.get("notes", []) + nxt.get("notes", []) + ["已合并片段"])}),
            "person_count": max(int(prev.get("person_count", -1)), int(nxt.get("person_count", -1))),
            "transient_multi_person": bool(prev.get("transient_multi_person")) or bool(nxt.get("transient_multi_person")),
            "start_sources": list(prev.get("start_sources", [])),
            "end_sources": list(nxt.get("end_sources", [])),
            "edit_action": "",
        }

    if action == "merge_prev":
        if idx == 0:
            raise ValueError("已经是第一个片段,无法向上合并")
        prev = segments[idx - 1]
        curr = segments[idx]
        new_seg = _build_merged_seg(prev, curr)
        from spvideo.ffmpeg_tools import cut_segment
        new_clip = clips_dir / f"{new_seg['segment_id']}_{new_seg['segment_type']}_{new_seg['start']:.2f}_{new_seg['end']:.2f}.mp4"
        cut_segment(video_path, new_seg["start"], new_seg["end"], new_clip)
        new_seg["output_path"] = str(new_clip)
        for old in (prev, curr):
            old_path = old.get("output_path") or ""
            if old_path and Path(old_path).exists():
                Path(old_path).unlink()
        segments[idx - 1:idx + 1] = [new_seg]
        manifest["segments"] = segments
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return _load_project_result(project_dir)

    if action == "merge_next":
        if idx >= len(segments) - 1:
            raise ValueError("已经是最后一个片段,无法向下合并")
        curr = segments[idx]
        nxt = segments[idx + 1]
        new_seg = _build_merged_seg(curr, nxt)
        from spvideo.ffmpeg_tools import cut_segment
        new_clip = clips_dir / f"{new_seg['segment_id']}_{new_seg['segment_type']}_{new_seg['start']:.2f}_{new_seg['end']:.2f}.mp4"
        cut_segment(video_path, new_seg["start"], new_seg["end"], new_clip)
        new_seg["output_path"] = str(new_clip)
        for old in (curr, nxt):
            old_path = old.get("output_path") or ""
            if old_path and Path(old_path).exists():
                Path(old_path).unlink()
        segments[idx:idx + 2] = [new_seg]
        manifest["segments"] = segments
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return _load_project_result(project_dir)

    raise ValueError(f"不支持的操作: {action}")


def _resize_ref_image(image_path: str, max_size_mb: int = 4, max_pixels: int = 2048) -> str:
    """压缩参考图到 Wan2.2 可接受的大小，返回路径（可能原地覆盖或生成新文件）"""
    if image_path.startswith("http"):
        return image_path
    p = Path(image_path)
    if not p.exists():
        return image_path
    import cv2
    size_mb = os.path.getsize(image_path) / (1024 * 1024)
    if size_mb <= max_size_mb:
        return image_path
    img = cv2.imread(image_path)
    if img is None:
        return image_path
    h, w = img.shape[:2]
    if max(h, w) > max_pixels:
        scale = max_pixels / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    out_path = str(p.parent / f"{p.stem}_resized.jpg")
    cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out_path


def _file_sha256_short(path: str, length: int = 12) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:length]
    except Exception:
        return "unknown"


def _log_role_reference_debug(role_pairs: list[dict[str, Any]], add_log) -> None:
    refs_by_hash: dict[str, list[str]] = {}
    for pair in role_pairs:
        name = str(pair.get("name") or "")
        ref_image = str(pair.get("ref_image") or "")
        ref_path = Path(ref_image)
        digest = _file_sha256_short(ref_image)
        size = ref_path.stat().st_size if ref_path.exists() and ref_path.is_file() else 0
        add_log(f"  参考图: {name} path={ref_image} sha={digest} size={size}")
        extra_refs = _string_list(pair.get("extra_ref_images"))
        add_log(f"  SCAIL-2 补充图数量: {name} {len(extra_refs)}/6")
        for index, extra_ref in enumerate(extra_refs, 1):
            extra_path = Path(extra_ref)
            extra_digest = _file_sha256_short(extra_ref)
            extra_size = extra_path.stat().st_size if extra_path.exists() and extra_path.is_file() else 0
            add_log(
                f"  SCAIL-2 补充图: {name} #{index} "
                f"path={extra_ref} sha={extra_digest} size={extra_size}"
            )
        if digest != "unknown":
            refs_by_hash.setdefault(digest, []).append(name)
    for digest, names in refs_by_hash.items():
        if len(names) > 1:
            add_log(
                "> 警告: 多个角色正在使用同一张参考图 "
                f"sha={digest}: {'、'.join(names)}；这通常会导致换人不明显或换错人"
            )


def _default_scail2_sam_text(role_count: int) -> str:
    return "a single human person, full body"


def _resolve_scail2_sam_text(value: str, role_count: int) -> str:
    text = str(value or "").strip()
    if not text:
        return _default_scail2_sam_text(role_count)
    if int(role_count) > 1 and text.lower() in {"people", "persons", "humans", "human beings"}:
        return _default_scail2_sam_text(role_count)
    return text


def _default_scail2_positive_prompt(role_pairs: list[dict[str, Any]]) -> str:
    count = len(role_pairs)
    appearance_lock = (
        "preserve each reference subject's apparent age, face shape, facial proportions, body build, "
        "shoulder width, waist and hip shape, height and weight impression; do not beautify, slim, "
        "de-age, reshape, or turn mature/stocky parent characters into young thin model bodies"
    )
    scene_lock = (
        "the source video background and camera space are locked temporary guides; ignore any background "
        "seen in reference images; do not add a theater stage, black curtain, studio backdrop, runway, "
        "dance floor, showroom, or blank/default background unless it is already present in the source video"
    )
    motion_lock = (
        "strictly follow the source video motion frame by frame; keep the exact standing positions, body angles, "
        "hand positions, arm grabs, touch points, eye lines, and contact/occlusion relationships; if the source video "
        "is not a dance scene, do not reinterpret it as dancing, choreography, performance, posing, or a music-video "
        "movement; if the source video is a dance scene, preserve the exact original choreography and timing without "
        "inventing new dance moves"
    )
    role_lock = _scail2_role_appearance_constraints(role_pairs)
    if count > 1:
        return (
            f"{count} people in the original scene, replace each tracked person with the matching reference subject, "
            "keep the original plot moment, action, body pose, camera movement, occlusion order, lighting, background, "
            "and scene composition; keep each person's gender and position; do not invent a new scene type; "
            f"{motion_lock}; {appearance_lock}; {scene_lock}; {role_lock}"
        )
    return (
        "one person in the original scene, replace the tracked person with the reference subject, "
        "keep the original plot moment, action, body pose, camera movement, lighting, background, and scene composition; "
        f"do not invent a new scene type; {motion_lock}; {appearance_lock}; {scene_lock}; {role_lock}"
    )


def _resolve_scail2_positive_prompt(value: str, role_pairs: list[dict[str, Any]]) -> str:
    base = str(value or "").strip()
    constraints = _default_scail2_positive_prompt(role_pairs)
    if not base:
        return constraints
    lower = base.lower()
    required_markers = (
        "do not invent a new scene type",
        "strictly follow the source video motion frame by frame",
        "preserve each reference subject's apparent age",
        "do not add a theater stage",
    )
    if all(marker in lower for marker in required_markers):
        return base
    return base.rstrip(" .") + ". " + constraints


def _scail2_role_appearance_constraints(role_pairs: list[dict[str, Any]]) -> str:
    constraints: list[str] = []
    for pair in role_pairs:
        name = str(pair.get("name") or "").strip()
        if not name:
            continue
        if any(token in name for token in ("养父", "父亲", "爸爸", "爹")):
            constraints.append(
                f"{name}: mature father, keep an older middle-aged male face, natural wrinkles, "
                "broader solid body, not a young skinny man"
            )
        elif any(token in name for token in ("养母", "母亲", "妈妈", "婆婆", "阿姨")):
            constraints.append(
                f"{name}: mature mother, keep an older middle-aged female face and natural fuller body, "
                "not a young thin model"
            )
        elif any(token in name for token in ("千金", "女儿", "女主", "女孩")):
            constraints.append(f"{name}: young adult woman, keep the reference face and body proportions")
    if not constraints:
        return "keep every role's reference-specific age, face shape, body type, and clothing"
    return "; ".join(constraints)


def _resolve_scail2_role_pairs(
    *,
    char_names: list[str],
    characters: dict[str, Any],
    originals: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    video_path: str,
    director_plan: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Resolve target images and align their order with SCAIL-2 mask colors."""
    names: list[str] = []
    for raw_name in char_names:
        name = str(raw_name or "").strip()
        if name and name not in names:
            names.append(name)
    if not names:
        raise ValueError("请先选择要替换的角色")
    if len(names) > len(SCAIL2_COLOR_NAMES):
        raise ValueError(f"一次最多替换 {len(SCAIL2_COLOR_NAMES)} 个角色")

    character_by_name = {
        str(item.get("name") or "").strip(): item
        for item in characters.values()
        if str(item.get("name") or "").strip()
    }
    pairs: list[dict[str, Any]] = []
    missing_characters: list[str] = []
    missing_refs: list[str] = []
    director_roles = {
        str(item.get("name") or "").strip(): item
        for item in (director_plan or {}).get("roles", [])
        if str(item.get("name") or "").strip()
    }

    for name in names:
        character = character_by_name.get(name)
        if character is None:
            missing_characters.append(name)
            continue
        ref_image = str(character.get("ref_image") or "").strip()
        if not ref_image or not Path(ref_image).exists():
            missing_refs.append(name)
            continue
        extra_ref_images = [
            path
            for path in _string_list(character.get("extra_ref_images"))
            if Path(path).exists()
        ]

        source_items = [
            item
            for item in originals
            if str(item.get("label_name") or "").strip() == name
        ]
        exact_sources = [
            item for item in source_items if _same_video_path(item.get("video_path"), video_path)
        ]
        source_item = min(exact_sources or source_items, key=_mapping_item_time, default=None)

        director_role = director_roles.get(name, {})
        director_annotation_id = str(director_role.get("annotation_id") or "")
        position_items = [
            item
            for item in annotations
            if item.get("type") == "person"
            and str(item.get("label_name") or "").strip() == name
            and _same_video_path(item.get("video_path"), video_path)
        ]
        position_item = _pick_role_position_item(
            position_items,
            preferred_annotation_id=director_annotation_id,
        )
        if position_item is None and exact_sources:
            position_item = min(exact_sources, key=_mapping_item_time)

        pairs.append({
            "name": name,
            "ref_image": ref_image,
            "extra_ref_images": extra_ref_images,
            "source_image": (
                str((source_item or {}).get("cutout_path") or (source_item or {}).get("crop_path") or "")
            ),
            "source_x": _mapping_item_x(position_item),
            "source_time": _mapping_item_time(position_item) if position_item is not None else None,
            "annotation_id": str((position_item or {}).get("id") or ""),
            "track_dir": str((position_item or {}).get("track_dir") or ""),
        })

    if missing_characters:
        raise ValueError("角色不存在: " + "、".join(missing_characters))
    if missing_refs:
        raise ValueError(
            "以下角色还没有新人物参考图，请先点配对里的「换图」: " + "、".join(missing_refs)
        )

    mapping_warning = ""
    if len(pairs) > 1:
        positions = [pair.get("source_x") for pair in pairs]
        positions_known = all(isinstance(value, (int, float)) for value in positions)
        positions_distinct = positions_known and all(
            abs(float(left) - float(right)) >= 0.02
            for index, left in enumerate(positions)
            for right in positions[index + 1:]
        )
        if not positions_distinct:
            unresolved_roles = [
                pair["name"]
                for pair in pairs
                if not isinstance(pair.get("source_x"), (int, float))
            ]
            detail = "、".join(unresolved_roles or [pair["name"] for pair in pairs])
            raise ValueError(
                "多人转绘已阻止：当前片段开头无法区分这些角色的位置: "
                f"{detail}。请切成单镜头片段，并只勾选该镜头实际出现的角色。"
            )
        if not director_plan:
            late_pairs = [
                pair
                for pair in pairs
                if not isinstance(pair.get("source_time"), (int, float))
                or float(pair["source_time"]) > 1.0
            ]
            if late_pairs:
                detail = "、".join(
                    f"{pair['name']}({float(pair['source_time']):.2f}s)"
                    if isinstance(pair.get("source_time"), (int, float))
                    else f"{pair['name']}(无开头标注)"
                    for pair in late_pairs
                )
                raise ValueError(
                    "多人转绘已阻止：所有目标人物必须同时出现在片段第一帧附近；"
                    f"{detail} 不满足条件。请按镜头分别转绘，每次只勾选当前人物。"
                )

        # SCAIL2 assigns subject_1...subject_n to its masks from left to right.
        # Director role order is only a UI/annotation order, never a mask order.
        pairs.sort(key=lambda pair: float(pair["source_x"]))

    return pairs, mapping_warning


def _sync_auto_director_roles_to_assets(
    project_dir: str | Path,
    plan: dict[str, Any],
) -> list[str]:
    """Create global character stubs from confirmed auto-director answers."""
    ignored = {
        "",
        "unknown",
        "keep_original",
        "no_person",
        "manual_director",
        "passerby",
    }
    names: list[str] = []

    for entity in plan.get("entities", []):
        name = str(entity.get("resolved_as") or "").strip()
        if name and name not in ignored and name not in names:
            names.append(name)

    for decision in (plan.get("segment_decisions") or {}).values():
        for raw in decision.get("suggested_roles") or []:
            name = str(raw or "").strip()
            if name and name not in ignored and name not in names:
                names.append(name)

    # Also sync from answered question answers
    for question in plan.get("questions", []):
        answer = question.get("answer")
        if not answer:
            continue
        values = answer if isinstance(answer, list) else [answer]
        for raw in values:
            name = str(raw or "").strip()
            if name and name not in ignored and name not in names:
                names.append(name)

    # Also sync from director plans associated with questions
    for question in plan.get("questions", []):
        for seg_id in (question.get("segment_ids") or []):
            dp = get_director_plan(project_dir, segment_id=str(seg_id), video_path="")
            for role in (dp or {}).get("roles", []):
                name = str(role.get("name") or "").strip()
                if name and name not in ignored and name not in names:
                    names.append(name)

    synced: list[str] = []
    for name in names:
        upsert_character(project_dir, name=name)
        synced.append(name)
    return synced


def _sync_director_roles_to_characters(
    project_dir: str | Path,
    plan: dict[str, Any],
) -> None:
    """Ensure every role name in a director plan has a character entry."""
    for role in (plan or {}).get("roles", []):
        name = str(role.get("name") or "").strip()
        if name:
            upsert_character(project_dir, name=name)



def _mapping_item_time(item: dict[str, Any] | None) -> float:
    try:
        return max(0.0, float((item or {}).get("time") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _annotation_has_ready_track(item: dict[str, Any] | None) -> bool:
    if not item:
        return False
    if str(item.get("track_status") or "") != "ready":
        return False
    track_dir = str(item.get("track_dir") or "").strip()
    if not track_dir:
        return False
    path = Path(track_dir)
    if not path.exists():
        return False
    if int(item.get("tracked_frames") or 0) > 0:
        return True
    return any(path.glob("mask_*.png"))


def _mapping_item_point(item: dict[str, Any] | None) -> tuple[float, float] | None:
    if not item:
        return None
    point = item.get("point")
    if isinstance(point, list) and len(point) == 2:
        try:
            x = float(point[0])
            y = float(point[1])
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                return x, y
        except (TypeError, ValueError):
            pass
    box = item.get("box") or item.get("mask_box")
    if isinstance(box, list) and len(box) >= 4:
        try:
            x = float(box[0]) + float(box[2]) / 2.0
            y = float(box[1]) + float(box[3]) / 2.0
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                return x, y
        except (TypeError, ValueError):
            pass
    return None


def _mapping_item_distance(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> float | None:
    left_point = _mapping_item_point(left)
    right_point = _mapping_item_point(right)
    if left_point is None or right_point is None:
        return None
    return math.hypot(left_point[0] - right_point[0], left_point[1] - right_point[1])


def _mode2_reference_mask_role_pairs_from_store(
    project_dir: str,
    segment_id: str,
    fallback_pairs: list[dict[str, Any]],
    *,
    prefer_current_shot_roles: bool = True,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if not prefer_current_shot_roles or not project_dir or not segment_id:
        return fallback_pairs, warnings
    try:
        root = _resolve_storyboard_mode2_project_dir(project_dir)
        store_path = _storyboard_mode2_asset_store_path(root)
        data = json.loads(store_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"未能读取 Mode2 当前分镜角色绑定，沿用前端参考图: {exc}")
        return fallback_pairs, warnings

    assets = {
        str(asset.get("id") or ""): asset
        for asset in (data.get("assets") or [])
        if isinstance(asset, dict)
    }
    shot = next(
        (
            item for item in (data.get("shots") or [])
            if isinstance(item, dict) and str(item.get("segment_id") or "").strip() == segment_id
        ),
        None,
    )
    if not shot:
        warnings.append(f"未找到当前分镜 {segment_id}，沿用前端参考图")
        return fallback_pairs, warnings

    annotations = [
        item for item in (data.get("identity_annotations") or [])
        if isinstance(item, dict)
    ]

    role_ids = _string_list(shot.get("role_ids") or shot.get("role_asset_ids"))
    if not role_ids:
        role_ids = [
            asset_id for asset_id in _string_list(shot.get("asset_ids"))
            if str((assets.get(asset_id) or {}).get("kind") or "") == "role"
        ]
    role_pairs: list[dict[str, Any]] = []
    missing_targets: list[str] = []
    seen_paths: set[str] = set()
    for role_id in role_ids:
        asset = assets.get(role_id)
        if not asset or str(asset.get("kind") or "") != "role":
            continue
        ref_image = str(asset.get("target_image") or asset.get("seedance_reference_image") or "").strip()
        name = str(asset.get("name") or asset.get("tag") or role_id).strip() or role_id
        if not ref_image:
            missing_targets.append(name)
            continue
        key = str(Path(ref_image)).lower()
        if key in seen_paths:
            continue
        seen_paths.add(key)
        candidates = [
            item for item in annotations
            if str(item.get("role_id") or "") == role_id
        ]
        preferred = next(
            (
                item for item in candidates
                if str(item.get("shot_id") or "") == segment_id
                or str(item.get("source_segment_id") or "") == segment_id
            ),
            None,
        )
        if preferred is None and candidates:
            preferred = max(candidates, key=lambda item: float(item.get("updated_at") or item.get("time") or 0.0))
        point = preferred.get("point") if isinstance(preferred, dict) else None
        source_time = preferred.get("time") if isinstance(preferred, dict) else None
        source_shape = _normalize_identity_shape(
            preferred.get("source_shape") or preferred.get("shape")
        ) if isinstance(preferred, dict) else None
        role_pairs.append({
            "name": name,
            "ref_image": ref_image,
            "asset_id": role_id,
            "source_point": point,
            "source_shape": source_shape,
            "source_time": source_time,
            "annotation_id": str(preferred.get("id") or "") if isinstance(preferred, dict) else "",
        })

    if missing_targets:
        warnings.append("这些当前分镜角色没有目标参考图，未参与彩色蒙版: " + "、".join(missing_targets))
    if not role_pairs:
        warnings.append("当前分镜没有可用角色目标图，沿用前端参考图")
        return fallback_pairs, warnings
    if len(role_pairs) > 6:
        warnings.append("当前分镜角色目标图超过 6 张，只取前 6 张")
        role_pairs = role_pairs[:6]
    fallback_keys = {str(Path(str(pair.get("ref_image") or ""))).lower() for pair in fallback_pairs}
    role_keys = {str(Path(str(pair.get("ref_image") or ""))).lower() for pair in role_pairs}
    if fallback_keys and fallback_keys != role_keys:
        warnings.append("已忽略 Seedance 草稿里的非当前分镜角色参考图，改用当前分镜角色绑定")
    return role_pairs, warnings


def _mode2_reference_video_hard_cuts(video_path: str) -> tuple[list[tuple[float, float]], str]:
    try:
        from spvideo.scene_detector import detect_hard_cuts

        scenes = detect_hard_cuts(video_path, min_scene_duration=0.3, threshold=27.0)
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)
    clean = [
        (round(max(0.0, float(start)), 3), round(max(0.0, float(end)), 3))
        for start, end in scenes
        if float(end) > float(start) + 0.05
    ]
    return clean, ""


def _mode2_reference_mask_subclip_dir(video_path: str) -> Path:
    source = Path(video_path)
    return source.parent / "subshots" / source.stem


def _mode2_create_reference_mask_subclips(
    video_path: str,
    hard_cuts: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    from spvideo.ffmpeg_tools import cut_segment

    source = Path(video_path)
    output_dir = _mode2_reference_mask_subclip_dir(video_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    subclips: list[dict[str, Any]] = []
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in source.stem)[:60] or "clip"
    for index, (start, end) in enumerate(hard_cuts, 1):
        duration = max(0.01, end - start)
        start_ms = max(0, int(round(start * 1000)))
        end_ms = max(start_ms + 10, int(round(end * 1000)))
        target = output_dir / f"{safe_stem}_sub{index:02d}_{start_ms:08d}_{end_ms:08d}.mp4"
        if not (target.exists() and target.stat().st_size > 0):
            cut_segment(source, start, end, target)
        subclips.append({
            "index": index,
            "label": f"子镜头{index}",
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(duration, 3),
            "path": str(target),
            "source_video_path": str(source),
        })
    return subclips


def _run_scail2_mask_check_job(
    job_id: str,
    video_path: str,
    role_pairs: list[dict[str, Any]],
    sampler_preset: str,
    video_window: dict[str, Any] | None,
    normalize_size: bool,
    sam_text: str = "",
    strict_track_preflight: bool = False,
    require_single_shot: bool = False,
) -> None:
    def add_log(message: str) -> None:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job.setdefault("logs", []).append(message)

    def fail(message: str) -> None:
        snapshot = None
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job["status"] = "failed"
                job["error"] = message
                job.setdefault("logs", []).append(f"> 失败: {message}")
                snapshot = dict(job)
        if snapshot is not None:
            _write_storyboard_job_snapshot(snapshot)

    def split_required(message: str, subclips: list[dict[str, Any]]) -> None:
        snapshot = None
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job["status"] = "split_required"
                job["error"] = message
                job["result"] = {
                    "status": "split_required",
                    "reason": "reference_video_has_internal_cuts",
                    "output_path": "",
                    "subclips": subclips,
                    "message": message,
                }
                job.setdefault("logs", []).append(f"> 需要按子镜头生成彩色蒙版: {message}")
                for item in subclips:
                    job.setdefault("logs", []).append(
                        f"> {item.get('label')}: {item.get('start'):.2f}-{item.get('end'):.2f}s -> {item.get('path')}"
                    )
                snapshot = dict(job)
        if snapshot is not None:
            _write_storyboard_job_snapshot(snapshot)

    if not Path(video_path).exists():
        fail(f"video_not_found: {video_path}")
        return
    ref_images = [str(pair.get("ref_image") or "") for pair in role_pairs]
    if not ref_images:
        fail("ref_image_not_found: 没有参考图")
        return
    if require_single_shot:
        hard_cuts, cut_error = _mode2_reference_video_hard_cuts(video_path)
        if cut_error:
            add_log(f"> 单镜头校验跳过: {cut_error}")
        elif len(hard_cuts) > 1:
            ranges = "、".join(f"{start:.2f}-{end:.2f}s" for start, end in hard_cuts[:8])
            message = (
                "reference_video_has_internal_cuts: 这个参考片段内部包含 "
                f"{len(hard_cuts)} 个硬切镜头（{ranges}）。"
                "彩色蒙版/SAM3 分轨必须按单个真实镜头分别生成；"
                "已自动切出子镜头，请选择子镜头再生成彩色蒙版。"
            )
            try:
                subclips = _mode2_create_reference_mask_subclips(video_path, hard_cuts)
            except Exception as exc:  # noqa: BLE001
                fail(f"{message} 自动切子镜头失败: {exc}")
                return
            split_required(message, subclips)
            return
        elif hard_cuts:
            add_log(f"> 单镜头校验通过: {hard_cuts[0][0]:.2f}-{hard_cuts[0][1]:.2f}s")

    try:
        from spvideo.scail2_client import Scail2Client

        client = Scail2Client()
        source_identity_points = [
            pair.get("source_point") if isinstance(pair.get("source_point"), (list, tuple)) else None
            for pair in role_pairs
        ]
        source_identity_shapes = [
            pair.get("source_shape") if isinstance(pair.get("source_shape"), dict) else None
            for pair in role_pairs
        ]
        result = client.inspect_masks(
            video_path=video_path,
            ref_images=ref_images,
            role_names=[str(pair.get("name") or "") for pair in role_pairs],
            video_window=video_window,
            sampler_preset=sampler_preset,
            normalize_size=normalize_size,
            sam_text=_resolve_scail2_sam_text(sam_text, len(ref_images)),
            strict_track_preflight=strict_track_preflight,
            source_identity_points=source_identity_points,
            source_identity_shapes=source_identity_shapes,
            on_progress=add_log,
        )
        for warning in result.get("warnings") or []:
            add_log(f"> 蒙版质量提醒: {warning}")
        for label, paths in (result.get("mask_output_paths") or {}).items():
            for path in paths:
                add_log(f"> {label} 蒙版文件: {path}")
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job["status"] = "done"
                job["result"] = result
                job.setdefault("logs", []).append(
                    f"> 远程 SAM3 蒙版检查完成: {result.get('output_path') or ''}"
                )
                snapshot = dict(job)
            else:
                snapshot = None
        if snapshot is not None:
            _write_storyboard_job_snapshot(snapshot)
    except Exception as exc:  # noqa: BLE001
        error_message = str(exc) or type(exc).__name__
        trace_tail = traceback.format_exc().strip().splitlines()[-8:]
        for line in trace_tail:
            add_log("> traceback: " + line)
        fail(error_message)


def _run_scail2_white_mask_job(
    job_id: str,
    video_path: str,
    model: str,
    video_window: dict[str, Any] | None,
    normalize_size: bool,
    input_size: int,
    max_res: int,
    precision: str,
    output_kind: str = "white",
) -> None:
    def add_log(message: str) -> None:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job.setdefault("logs", []).append(message)

    def finish(status: str, *, result: dict[str, Any] | None = None, error: str | None = None) -> None:
        snapshot = None
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job["status"] = status
                job["result"] = result
                job["error"] = error
                if error:
                    job.setdefault("logs", []).append(f"> failed: {error}")
                snapshot = dict(job)
        if snapshot is not None:
            _write_storyboard_job_snapshot(snapshot)

    if not Path(video_path).exists():
        finish("failed", error=f"video_not_found: {video_path}")
        return

    try:
        from spvideo.scail2_client import Scail2Client

        client = Scail2Client()
        output_kind = str(output_kind or "white").strip().lower() or "white"
        if output_kind in {"identity_gray_relief", "gray_relief", "light_gray_control"}:
            result = client.inspect_identity_gray_relief(
                video_path=video_path,
                video_window=video_window,
                normalize_size=normalize_size,
                on_progress=add_log,
            )
            add_log(f"> 光影灰白控制主输出: {result.get('output_path') or ''}")
            add_log(f"> 光影灰白控制完成: {result.get('output_path') or ''}")
            finish("done", result=result)
            return

        result = client.inspect_white_mask(
            video_path=video_path,
            model=model,
            video_window=video_window,
            normalize_size=normalize_size,
            input_size=input_size,
            max_res=max_res,
            precision=precision,
            on_progress=add_log,
        )
        if output_kind in {"background_gray", "bg_gray", "depth", "scene_depth"}:
            depth_path = str(result.get("depth_path") or ((result.get("mask_output_paths") or {}).get("depth") or [""])[0] or "")
            if depth_path:
                result["background_gray_path"] = depth_path
                result["output_path"] = depth_path
                result["workflow_mode"] = "remote_vda_background_gray"
                add_log(f"> VDA 背景灰模主输出: {depth_path}")
        elif output_kind in {"normal_lighting", "normal_lit", "light_clay", "strong_clay"}:
            depth_path = Path(str(result.get("depth_path") or ((result.get("mask_output_paths") or {}).get("depth") or [""])[0] or ""))
            if depth_path.exists():
                output_dir = Path(result.get("output_path") or depth_path).parent
                task_id = str(result.get("prompt_id") or uuid.uuid4().hex)
                strong_path = output_dir / f"scail2_vda_strong_clay_{Path(video_path).stem}_{task_id[:8]}.mp4"
                client._make_strong_depth_relief_video(depth_path, strong_path)
                mask_output_paths = dict(result.get("mask_output_paths") or {})
                mask_output_paths["strong_clay"] = [str(strong_path)]
                mask_output_paths["normal_lit_clay"] = [str(strong_path)]
                result["mask_output_paths"] = mask_output_paths
                result["normal_lit_clay_path"] = str(strong_path)
                result["strong_clay_path"] = str(strong_path)
                result["output_path"] = str(strong_path)
                result["workflow_mode"] = "remote_vda_normal_lighting"
                add_log(f"> VDA 强2.5D白模主输出: {strong_path}")
            else:
                add_log("> VDA 强2.5D白模缺少 depth_path，保留普通白膜输出")
        for warning in result.get("warnings") or []:
            add_log(f"> 白膜质量提醒: {warning}")
        for label, paths in (result.get("mask_output_paths") or {}).items():
            for path in paths:
                add_log(f"> {label} 白膜文件: {path}")
        add_log(f"> 远程 VDA 白膜完成: {result.get('output_path') or ''}")
        finish("done", result=result)
    except Exception as exc:  # noqa: BLE001
        error_message = str(exc) or type(exc).__name__
        trace_tail = traceback.format_exc().strip().splitlines()[-8:]
        for line in trace_tail:
            add_log("> traceback: " + line)
        finish("failed", error=error_message)


def _run_scail2_capsule_control_job(
    job_id: str,
    video_path: str,
    video_window: dict[str, Any],
    normalize_size: bool,
    pose_backend: str,
    pose_capsule_strength: str,
) -> None:
    def add_log(message: str) -> None:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job.setdefault("logs", []).append(message)

    def finish(status: str, *, result: dict[str, Any] | None = None, error: str | None = None) -> None:
        snapshot = None
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job["status"] = status
                job["result"] = result
                job["error"] = error
                if error:
                    job.setdefault("logs", []).append(f"> failed: {error}")
                snapshot = dict(job)
        if snapshot is not None:
            _write_storyboard_job_snapshot(snapshot)

    if not Path(video_path).exists():
        finish("failed", error=f"video_not_found: {video_path}")
        return

    try:
        from spvideo.scail2_client import Scail2Client

        client = Scail2Client()
        result = client.inspect_capsule_control(
            video_path=video_path,
            video_window=video_window,
            normalize_size=normalize_size,
            pose_backend=pose_backend,
            pose_capsule_strength=pose_capsule_strength,
            on_progress=add_log,
        )
        for label, paths in (result.get("mask_output_paths") or {}).items():
            for path in paths:
                add_log(f"> {label} control file: {path}")
        add_log(f"> low poly capsule control complete: {result.get('output_path') or ''}")
        finish("done", result=result)
    except Exception as exc:  # noqa: BLE001
        error_message = str(exc) or type(exc).__name__
        trace_tail = traceback.format_exc().strip().splitlines()[-8:]
        for line in trace_tail:
            add_log("> traceback: " + line)
        finish("failed", error=error_message)


def _run_scail2_expression_mask_job(
    job_id: str,
    video_path: str,
    model: str,
    video_window: dict[str, Any],
    normalize_size: bool,
    input_size: int,
    max_res: int,
    precision: str,
    max_faces: int,
    include_mouth: bool = True,
    color_faces: bool = True,
    include_eyes: bool = True,
    include_brows: bool = True,
    include_head_pose: bool = True,
    include_face_outline: bool = True,
    include_soft_face_relief: bool = False,
    strong_depth_relief: bool = False,
    safe_mode: bool = True,
    body_color_mode: str = "none",
    include_body_pose: bool = False,
    pose_backend: str = "server_dwpose",
    pose_render_style: str = "capsule",
    pose_capsule_strength: str = "strong",
) -> None:
    def add_log(message: str) -> None:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job.setdefault("logs", []).append(message)

    def finish(status: str, *, result: dict[str, Any] | None = None, error: str | None = None) -> None:
        snapshot = None
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job["status"] = status
                job["result"] = result
                job["error"] = error
                if error:
                    job.setdefault("logs", []).append(f"> failed: {error}")
                snapshot = dict(job)
        if snapshot is not None:
            _write_storyboard_job_snapshot(snapshot)

    if not Path(video_path).exists():
        finish("failed", error=f"video_not_found: {video_path}")
        return

    try:
        from spvideo.scail2_client import Scail2Client

        client = Scail2Client()
        result = client.inspect_expression_mask(
            video_path=video_path,
            model=model,
            video_window=video_window,
            normalize_size=normalize_size,
            input_size=input_size,
            max_res=max_res,
            precision=precision,
            max_faces=max_faces,
            include_mouth=include_mouth,
            color_faces=color_faces,
            include_eyes=include_eyes,
            include_brows=include_brows,
            include_head_pose=include_head_pose,
            include_face_outline=include_face_outline,
            include_soft_face_relief=include_soft_face_relief,
            strong_depth_relief=strong_depth_relief,
            safe_mode=safe_mode,
            body_color_mode=body_color_mode,
            include_body_pose=include_body_pose,
            pose_backend=pose_backend,
            pose_render_style=pose_render_style,
            pose_capsule_strength=pose_capsule_strength,
            on_progress=add_log,
        )
        for warning in result.get("warnings") or []:
            add_log(f"> 表情白膜质量提醒: {warning}")
        for label, paths in (result.get("mask_output_paths") or {}).items():
            for path in paths:
                add_log(f"> {label} 表情白膜文件: {path}")
        stats = result.get("expression_stats") or {}
        if isinstance(stats, dict) and "body_pose_frames" in stats:
            add_log(f"> body pose overlay frames: {stats.get('body_pose_frames') or 0}")
            add_log(f"> body pose render style: {stats.get('body_pose_render_style') or pose_render_style}")
            add_log(f"> body pose capsule strength: {stats.get('body_pose_capsule_strength') or pose_capsule_strength}")
        if isinstance(stats, dict):
            add_log(f"> face outline: {stats.get('face_outline') or 'unknown'}")
            add_log(f"> soft face relief: {stats.get('soft_face_relief') or 'unknown'}")
            add_log(f"> strong depth relief: {stats.get('strong_depth_relief') or 'unknown'}")
            layers = stats.get("guidance_layers") or []
            if isinstance(layers, list):
                add_log("> guidance layers: " + ", ".join(str(item) for item in layers))
        add_log(f"> 远程 VDA 表情白膜完成: {result.get('output_path') or ''}")
        finish("done", result=result)
    except Exception as exc:  # noqa: BLE001
        error_message = str(exc) or type(exc).__name__
        trace_tail = traceback.format_exc().strip().splitlines()[-8:]
        for line in trace_tail:
            add_log("> traceback: " + line)
        finish("failed", error=error_message)


def _safe_mode2_output_name(value: str, fallback: str) -> str:
    name = Path(str(value or "").strip()).name
    if not name:
        name = fallback
    name = re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("._")
    if not name:
        name = fallback
    if Path(name).suffix.lower() != ".mp4":
        name = f"{Path(name).stem or name}.mp4"
    return name


def _run_storyboard_reference_mask_merge_job(job_id: str, payload: dict[str, Any]) -> None:
    def add_log(message: str) -> None:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job.setdefault("logs", []).append(message)

    def finish(status: str, *, result: dict[str, Any] | None = None, error: str | None = None) -> None:
        snapshot = None
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job["status"] = status
                job["result"] = result
                job["error"] = error
                if error:
                    job.setdefault("logs", []).append(f"> failed: {error}")
                snapshot = dict(job)
        if snapshot is not None:
            _write_storyboard_job_snapshot(snapshot)

    try:
        root = _resolve_storyboard_mode2_project_dir(payload.get("project_dir") or "")
        raw_mask_paths = payload.get("mask_paths") or payload.get("input_mask_paths") or []
        mask_paths = _string_list(raw_mask_paths)
        if not mask_paths:
            raise ValueError("mask_paths_required")
        input_paths: list[Path] = []
        for item in mask_paths:
            path = Path(item)
            if not path.exists() or not path.is_file():
                raise ValueError(f"mask_path_not_found: {item}")
            if path.suffix.lower() != ".mp4":
                raise ValueError(f"mask_path_must_be_mp4: {item}")
            input_paths.append(path)

        source_video_path = str(payload.get("source_video_path") or payload.get("video_path") or "").strip()
        if source_video_path and not Path(source_video_path).exists():
            raise ValueError(f"source_video_not_found: {source_video_path}")
        segment_id = str(payload.get("segment_id") or "").strip()
        source_stem = Path(source_video_path).stem if source_video_path else ""
        output_key = segment_id or source_stem or "reference"
        output_hash = hashlib.sha1("|".join(str(path.resolve()) for path in input_paths).encode("utf-8")).hexdigest()[:8]
        fallback_name = f"scail2_mask_only_colored_full_{output_key}_{output_hash}.mp4"
        output_name = _safe_mode2_output_name(str(payload.get("output_name") or ""), fallback_name)
        output_dir = root / "04_AI输出成片"
        output_path = output_dir / output_name
        if output_path.exists():
            output_path = output_dir / f"{Path(output_name).stem}_{uuid.uuid4().hex[:8]}.mp4"

        add_log(f"> merge output: {output_path}")
        add_log("> ffmpeg concat copy first, reencode fallback enabled")
        merged = concat_videos(input_paths, output_path, reencode_fallback=True)
        mask_label = str(payload.get("mask_label") or "pose").strip() or "pose"
        subclips = payload.get("subclips") if isinstance(payload.get("subclips"), list) else []
        result = {
            "output_path": str(merged),
            "mask_output_paths": {mask_label: [str(merged)]},
            "input_mask_paths": [str(path) for path in input_paths],
            "subclips": subclips,
            "source_video_path": source_video_path,
            "segment_id": segment_id,
            "project_dir": str(root),
        }
        add_log(f"> merge done: {merged}")
        finish("done", result=result)
    except Exception as exc:  # noqa: BLE001
        error_message = str(exc) or type(exc).__name__
        trace_tail = traceback.format_exc().strip().splitlines()[-8:]
        for line in trace_tail:
            add_log("> traceback: " + line)
        finish("failed", error=error_message)


def _pick_role_position_item(
    position_items: list[dict[str, Any]],
    *,
    preferred_annotation_id: str = "",
) -> dict[str, Any] | None:
    """Pick the best identity mark for a role on the current clip.

    Director binding anchors identity. Reuse a ready SAM3 track only when it is
    spatially close to that binding; old same-label tracks may point at another
    person.
    """
    if not position_items:
        return None

    preferred = next(
        (item for item in position_items if str(item.get("id") or "") == preferred_annotation_id),
        None,
    )
    ready_items = [item for item in position_items if _annotation_has_ready_track(item)]
    if preferred is not None and _annotation_has_ready_track(preferred):
        return preferred

    def sort_ready(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            items,
            key=lambda item: (
                -int(item.get("tracked_frames") or 0),
                _mapping_item_time(item),
            ),
        )

    if preferred is not None:
        nearby_ready = [
            item
            for item in ready_items
            if (distance := _mapping_item_distance(preferred, item)) is not None
            and distance <= 0.20
        ]
        if nearby_ready:
            return sort_ready(nearby_ready)[0]
        if str(preferred.get("track_status") or "") != "failed":
            preferred_time = _mapping_item_time(preferred)
            # Near-end director marks are fragile for SAM3; prefer earlier nearby marks when available.
            earlier = [
                item
                for item in position_items
                if _mapping_item_time(item) + 0.05 < preferred_time
                and isinstance(item.get("point"), list)
                and len(item.get("point") or []) == 2
                and (distance := _mapping_item_distance(preferred, item)) is not None
                and distance <= 0.20
            ]
            if earlier and preferred_time > 1.0:
                earlier.sort(key=_mapping_item_time)
                return earlier[0]
        return preferred

    if ready_items:
        return sort_ready(ready_items)[0]

    valid = [
        item
        for item in position_items
        if isinstance(item.get("point"), list) and len(item.get("point") or []) == 2
    ]
    if not valid:
        return preferred or min(position_items, key=_mapping_item_time, default=None)
    valid.sort(key=_mapping_item_time)
    return valid[0]


def _mapping_item_x(item: dict[str, Any] | None) -> float | None:
    if not item:
        return None
    point = item.get("point")
    if isinstance(point, list) and len(point) >= 1:
        try:
            value = float(point[0])
            return value if 0.0 <= value <= 1.0 else None
        except (TypeError, ValueError):
            pass
    box = item.get("box") or item.get("mask_box")
    if isinstance(box, list) and len(box) >= 3:
        try:
            value = float(box[0]) + float(box[2]) / 2.0
            return value if 0.0 <= value <= 1.0 else None
        except (TypeError, ValueError):
            pass
    return None


def _run_transfer_job(
    job_id: str,
    video_path: str,
    role_pairs: list[dict[str, Any]],
    project_dir: str = "",
    protection_annotation_id: str = "",
    segment_id: str = "",
    use_background: bool = False,
    sampler_preset: str = "balanced",
    video_window: dict[str, Any] | None = None,
    normalize_size: bool = True,
    positive_prompt: str = "",
    sam_text: str = "",
    transfer_backend: str | None = None,
) -> None:
    """后台执行单人 SCAIL2 或多人 Wan2.2 换人。"""
    def add_log(message: str) -> None:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job.setdefault("logs", []).append(message)

    def fail(message: str) -> None:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job["status"] = "failed"
                job["error"] = message
                job.setdefault("logs", []).append(f"> 失败: {message}")

    if not Path(video_path).exists():
        fail(f"video_not_found: {video_path}")
        return
    ref_images = [str(pair.get("ref_image") or "") for pair in role_pairs]
    source_identity_points = [
        pair.get("source_point") if isinstance(pair.get("source_point"), (list, tuple)) else None
        for pair in role_pairs
    ]
    source_identity_shapes = [
        pair.get("source_shape") if isinstance(pair.get("source_shape"), dict) else None
        for pair in role_pairs
    ]
    subject_extra_ref_images = [
        _string_list(pair.get("extra_ref_images"))
        for pair in role_pairs
    ]
    if not ref_images:
        fail("ref_image_not_found: 没有参考图")
        return

    use_wan22 = _use_wan22_transfer(len(role_pairs), transfer_backend)
    use_scail2_colored = _use_scail2_colored_transfer(transfer_backend)
    use_scail2_masked = _use_scail2_masked_transfer(transfer_backend)
    use_bernini = _use_bernini_transfer(transfer_backend)
    use_runninghub_bernini = _use_runninghub_bernini_transfer(transfer_backend)
    backend_label = "Wan2.2 云端换人" if use_wan22 else "SCAIL-2 ComfyUI 换人"
    add_log(f"> {backend_label} ({len(ref_images)} 个目标人物)")
    for i, pair in enumerate(role_pairs):
        color = SCAIL2_COLOR_NAMES[i]
        add_log(f"  {color}: {pair['name']} → {Path(pair['ref_image']).name}")
    _log_role_reference_debug(role_pairs, add_log)

    output_dir = Path(video_path).parent.parent.parent / "04_AI输出成片"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if use_wan22:
            if len(role_pairs) > 1:
                add_log("> 正在自动建立每个角色的 SAM3 轨迹")
                _ensure_wan_role_tracks(
                    project_dir=project_dir,
                    video_path=video_path,
                    role_pairs=role_pairs,
                    job_id=job_id,
                    add_log=add_log,
                )
            else:
                add_log("> Wan2.2 单角色模式: 跳过 SAM3，直接上传原始片段")
            result = _run_wan_multi_role_transfer(
                video_path=video_path,
                role_pairs=role_pairs,
                output_dir=output_dir,
                api_key=_wan_api_key(),
                add_log=add_log,
            )
            final_output_path = str(result["output_path"])
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is not None:
                    job["status"] = "done"
                    job["result"] = result
                    job.setdefault("logs", []).append(f"> 完成! {final_output_path}")
            return

        if use_runninghub_bernini:
            from spvideo.runninghub_client import RunningHubClient

            add_log("> RunningHub Bernini path: exported API workflow")
            client = RunningHubClient()
            result = client.transfer_bernini(
                video_path=video_path,
                ref_images=ref_images,
                role_names=[str(pair.get("name") or "") for pair in role_pairs],
                video_window=video_window,
                normalize_size=normalize_size,
                positive_prompt=_resolve_scail2_positive_prompt(positive_prompt, role_pairs),
                on_progress=add_log,
            )
        elif use_bernini:
            from spvideo.scail2_client import Scail2Client

            client = Scail2Client()
            add_log("> Bernini test path: WanAnimatePlus Bernini rv2v")
            result = client.transfer_bernini_test(
                video_path=video_path,
                ref_images=ref_images,
                role_names=[str(pair.get("name") or "") for pair in role_pairs],
                video_window=video_window,
                sampler_preset=sampler_preset,
                normalize_size=normalize_size,
                positive_prompt=_resolve_scail2_positive_prompt(positive_prompt, role_pairs),
                on_progress=add_log,
            )
        elif use_scail2_colored:
            from spvideo.scail2_client import Scail2Client

            client = Scail2Client()
            add_log("> SCAIL-2 colored mask path AUTO_SEG_V2: explicit SCAIL2ColoredMask + WanSCAILToVideo")
            result = client.transfer_colored_mask_test(
                video_path=video_path,
                ref_images=ref_images,
                role_names=[str(pair.get("name") or "") for pair in role_pairs],
                source_positions=[pair.get("source_x") for pair in role_pairs],
                video_window=video_window,
                sampler_preset=sampler_preset,
                normalize_size=normalize_size,
                positive_prompt=_resolve_scail2_positive_prompt(positive_prompt, role_pairs),
                sam_text=_resolve_scail2_sam_text(sam_text, len(ref_images)),
                on_progress=add_log,
            )
        elif use_scail2_masked:
            from spvideo.scail2_client import Scail2Client

            client = Scail2Client()
            add_log("> SCAIL-2 masked test path: WanAnimatePlus + explicit colored masks")
            result = client.transfer_wananimate_masked_test(
                video_path=video_path,
                ref_images=ref_images,
                role_names=[str(pair.get("name") or "") for pair in role_pairs],
                video_window=video_window,
                sampler_preset=sampler_preset,
                normalize_size=normalize_size,
                positive_prompt=_resolve_scail2_positive_prompt(positive_prompt, role_pairs),
                sam_text=_resolve_scail2_sam_text(sam_text, len(ref_images)),
                on_progress=add_log,
            )
        else:
            from spvideo.scail2_client import Scail2Client

            client = Scail2Client()
            result = client.transfer(
                video_path=video_path,
                ref_images=ref_images,
                subject_extra_ref_images=subject_extra_ref_images,
                role_names=[str(pair.get("name") or "") for pair in role_pairs],
                video_window=video_window,
                sampler_preset=sampler_preset,
                normalize_size=normalize_size,
                positive_prompt=_resolve_scail2_positive_prompt(positive_prompt, role_pairs),
                sam_text=_resolve_scail2_sam_text(sam_text, len(ref_images)),
                save_debug_masks=True,
                source_identity_points=source_identity_points,
                source_identity_shapes=source_identity_shapes,
                on_progress=add_log,
            )
            if len(ref_images) > 1 and not (result.get("mask_output_paths") or {}):
                add_log("> SCAIL-2 normal path produced no colored masks; running mask inspection fallback")
                mask_result = client.inspect_masks(
                    video_path=video_path,
                    ref_images=ref_images,
                    role_names=[str(pair.get("name") or "") for pair in role_pairs],
                    video_window=video_window,
                    sampler_preset=sampler_preset,
                    normalize_size=normalize_size,
                    sam_text=_resolve_scail2_sam_text(sam_text, len(ref_images)),
                    on_progress=add_log,
                )
                result["mask_output_paths"] = mask_result.get("mask_output_paths") or {}
        final_output_path = str(result["output_path"])

        protection_track_dir = ""
        if protection_annotation_id:
            store = load_asset_store(project_dir)
            protection = next(
                (item for item in store.get("annotations", []) if item.get("id") == protection_annotation_id),
                None,
            )
            if not protection:
                raise ValueError("protection_annotation_not_found")
            protection_track_dir = str(protection.get("track_dir") or "")
            protected_path = output_dir / f"protected_{Path(final_output_path).name}"
            add_log("> SCAIL-2 换人完成，开始保护区域回贴")
            from spvideo.protection_compositor import composite_protected_region

            composite_protected_region(
                source_video=video_path,
                generated_video=final_output_path,
                track_dir=protection_track_dir,
                output_path=protected_path,
            )
            final_output_path = str(protected_path)
            add_log(f"> 保护区域回贴完成: {final_output_path}")

        background_result = None
        if use_background:
            config = get_background_config(
                project_dir,
                segment_id=segment_id,
                video_path=video_path,
            )
            if not config or config.get("mode") == "keep_original":
                raise ValueError("background_config_required")
            store = load_asset_store(project_dir)
            foreground_tracks = [
                str(item.get("track_dir") or "")
                for item in store.get("annotations", [])
                if item.get("type") == "person"
                and _same_video_path(item.get("video_path"), video_path)
                and item.get("track_status") == "ready"
                and (Path(str(item.get("track_dir") or "")) / "track_summary.json").exists()
            ]
            if protection_track_dir:
                foreground_tracks.append(protection_track_dir)
            if not foreground_tracks:
                raise ValueError("background_foreground_track_not_ready")

            background_path = output_dir / f"background_{Path(final_output_path).name}"
            update_background_config(
                project_dir,
                segment_id=segment_id,
                video_path=video_path,
                status="running",
                error="",
            )
            add_log("> 换人结果已完成，开始独立背景后处理")
            from spvideo.background_compositor import compose_background

            background_result = compose_background(
                foreground_video=final_output_path,
                background_asset=str(config.get("asset_path") or ""),
                track_dirs=foreground_tracks,
                output_path=background_path,
                fit_mode=str(config.get("fit_mode") or "cover"),
                feather_pixels=int(config.get("feather_pixels") or 9),
                dilate_pixels=int(config.get("dilate_pixels") or 6),
            )
            final_output_path = str(background_result["output_path"])
            update_background_config(
                project_dir,
                segment_id=segment_id,
                video_path=video_path,
                status="done",
                output_path=final_output_path,
                error="",
            )
            add_log(f"> 背景后处理完成: {final_output_path}")

        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job["status"] = "done"
                job["result"] = {
                    "output_path": final_output_path,
                    "scail2_output_path": str(result["output_path"]),
                    "prompt_id": result.get("prompt_id"),
                    "workflow_path": result.get("workflow_path"),
                    "role_names": result.get("role_names"),
                    "ref_images": result.get("ref_images"),
                    "background": background_result,
                    "video_meta": result.get("video_meta"),
                    "video_window": result.get("video_window"),
                    "output_size": result.get("output_size"),
                    "sampler_preset": result.get("sampler_preset"),
                    "mask_output_paths": result.get("mask_output_paths"),
                }
                job.setdefault("logs", []).append(f"> 完成! {final_output_path}")
    except Exception as exc:  # noqa: BLE001
        error_message = str(exc) or type(exc).__name__
        trace_tail = traceback.format_exc().strip().splitlines()[-8:]
        if use_background and project_dir:
            update_background_config(
                project_dir,
                segment_id=segment_id,
                video_path=video_path,
                status="failed",
                error=error_message,
            )
        for line in trace_tail:
            add_log("> traceback: " + line)
        fail(error_message)


def _ensure_wan_role_tracks(
    *,
    project_dir: str,
    video_path: str,
    role_pairs: list[dict[str, Any]],
    job_id: str,
    add_log,
) -> None:
    """Reuse or build the per-role SAM3 tracks required by Wan focus videos."""
    _ensure_wan_text_object_tracks(
        project_dir=project_dir,
        video_path=video_path,
        role_pairs=role_pairs,
        job_id=job_id,
        add_log=add_log,
    )
    return


def _ensure_wan_text_object_tracks(
    *,
    project_dir: str,
    video_path: str,
    role_pairs: list[dict[str, Any]],
    job_id: str,
    add_log,
) -> None:
    """Track all people with SAM3 text, then bind object IDs using manual marks."""
    import itertools

    import cv2
    import numpy as np

    store = load_asset_store(project_dir)
    annotations = {
        str(item.get("id") or ""): item
        for item in store.get("annotations", [])
        if isinstance(item, dict)
    }
    role_items: list[tuple[dict[str, Any], dict[str, Any]]] = []
    all_reusable = True
    for pair in role_pairs:
        annotation_id = str(pair.get("annotation_id") or "")
        annotation = annotations.get(annotation_id)
        if not annotation:
            raise ValueError(f"{pair['name']} 缺少片段导演身份点")
        point = annotation.get("point")
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"{pair['name']} 缺少有效身份点")
        role_items.append((pair, annotation))
        track_dir = Path(str(annotation.get("track_dir") or pair.get("track_dir") or ""))
        reusable = (
            str(annotation.get("track_status") or "") == "ready"
            and track_dir.exists()
            and _track_uses_sam3_text_objects(track_dir)
            and len(list(track_dir.glob("mask_*.png"))) > 0
        )
        if reusable:
            pair["track_dir"] = str(track_dir)
        else:
            all_reusable = False
    if all_reusable:
        for pair, _ in role_items:
            add_log(f"> [{pair['name']}] 复用 SAM3 多对象文本轨迹 ({Path(pair['track_dir']).name})")
        return

    from spvideo.ffmpeg_tools import probe_video
    from spvideo.sam3_tracker import SAM3Tracker

    meta = probe_video(video_path)
    source_fps = float(meta.fps or 0.0) or 24.0
    prompt_time = max(0.0, float(meta.duration or 0.0) * 0.5)
    source_prompt_frame = max(0, int(round(prompt_time * source_fps)))
    with SAM3_TRACK_LOCK:
        clip_path, text_prompt_frame, start_frame, clip_frames, track_fps = _make_sam3_window_clip(
            video_path,
            frame_idx=source_prompt_frame,
            max_frames=SAM3_PROTECTION_MAX_FRAMES,
            job_id=f"{job_id}_text_objects",
        )
        tracker = SAM3Tracker()
        try:
            tracked = tracker.track_text_objects(
                video_path=str(clip_path),
                text_prompt="person",
                frame_idx=text_prompt_frame,
                max_frames=clip_frames,
                propagation_direction="both",
            )
        finally:
            tracker.close()

    object_ids = [int(value) for value in tracked.get("object_ids") or []]
    if len(object_ids) < len(role_items):
        raise ValueError(
            f"SAM3 只找到 {len(object_ids)} 个人物，但导演配置了 {len(role_items)} 个角色"
        )
    clip_start_time = start_frame / source_fps

    def point_cost(annotation: dict[str, Any], obj_id: int) -> float:
        point = [float(annotation["point"][0]), float(annotation["point"][1])]
        role_time = float(annotation.get("time") or 0.0)
        frame_index = max(
            0,
            min(clip_frames - 1, int(round((role_time - clip_start_time) * track_fps))),
        )
        mask = tracked["objects"][obj_id]["masks"][frame_index]
        if mask is None:
            return 10.0
        mask_array = np.squeeze(mask).astype(np.uint8)
        height, width = mask_array.shape[:2]
        px = max(0, min(width - 1, int(round(point[0] * width))))
        py = max(0, min(height - 1, int(round(point[1] * height))))
        if mask_array[py, px] > 0:
            return 0.0
        distances = cv2.distanceTransform((1 - mask_array).astype(np.uint8), cv2.DIST_L2, 3)
        return float(distances[py, px]) / float(max(width, height))

    best_assignment = None
    for permutation in itertools.permutations(object_ids, len(role_items)):
        costs = [
            point_cost(annotation, obj_id)
            for (_, annotation), obj_id in zip(role_items, permutation)
        ]
        candidate = (sum(costs), max(costs), permutation, costs)
        if best_assignment is None or candidate[:2] < best_assignment[:2]:
            best_assignment = candidate
    if best_assignment is None:
        raise ValueError("SAM3 多对象身份映射失败")
    _, max_cost, assignment, costs = best_assignment
    if max_cost > 0.05:
        raise ValueError(f"SAM3 人物蒙版距离手工身份点过远: {max_cost:.3f}")

    for (pair, annotation), obj_id, identity_distance in zip(role_items, assignment, costs):
        annotation_id = str(annotation["id"])
        point = [float(annotation["point"][0]), float(annotation["point"][1])]
        role_time = float(annotation.get("time") or 0.0)
        role_prompt_frame = max(
            0,
            min(clip_frames - 1, int(round((role_time - clip_start_time) * track_fps))),
        )
        output_dir = Path(project_dir) / "assets" / "tracks" / f"{annotation_id}_sam3_text"
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale_mask in output_dir.glob("mask_*.png"):
            stale_mask.unlink(missing_ok=True)
        object_result = tracked["objects"][obj_id]
        for frame_index, mask in enumerate(object_result["masks"]):
            if mask is None:
                continue
            mask_array = np.squeeze(mask).astype(np.uint8) * 255
            ok, encoded = cv2.imencode(".png", mask_array)
            if ok:
                (output_dir / f"mask_{frame_index:04d}.png").write_bytes(encoded.tobytes())
        summary = {
            "prompt_source": "sam3_text_person_manual_identity",
            "prompt_mode": "text_objects",
            "annotation_id": annotation_id,
            "sam3_object_id": obj_id,
            "identity_point_distance": identity_distance,
            "video_path": video_path,
            "clip_path": str(clip_path),
            "clip_start_frame": start_frame,
            "clip_start_time": clip_start_time,
            "text_prompt": "person",
            "text_prompt_frame": text_prompt_frame,
            "clip_prompt_frame": role_prompt_frame,
            "source_fps": source_fps,
            "track_fps": track_fps,
            "prompt_point": point,
            "candidate_time": role_time,
            "propagation_direction": "both",
            "num_frames": int(tracked.get("num_frames") or 0),
            "tracked_frames": int(object_result.get("tracked_frames") or 0),
        }
        _validate_sam3_role_track(
            output_dir=output_dir,
            point=point,
            prompt_frame=role_prompt_frame,
            tracked_frames=summary["tracked_frames"],
            total_frames=summary["num_frames"],
            role_name=str(pair["name"]),
        )
        summary_path = output_dir / "track_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        update_annotation(
            project_dir,
            annotation_id,
            track_status="ready",
            track_dir=str(output_dir),
            track_summary=str(summary_path),
            tracked_frames=summary["tracked_frames"],
            track_frames=summary["num_frames"],
            track_error="",
        )
        pair["track_dir"] = str(output_dir)
        add_log(
            f"> [{pair['name']}] SAM3 object={obj_id} "
            f"{summary['tracked_frames']}/{summary['num_frames']} 帧"
        )
    return


def _legacy_ensure_wan_direct_point_tracks(
    *,
    project_dir: str,
    video_path: str,
    role_pairs: list[dict[str, Any]],
    job_id: str,
    add_log,
) -> None:
    """Legacy direct-point tracker retained only for controlled rollback."""
    store = load_asset_store(project_dir)
    annotations_list = [
        item for item in store.get("annotations", [])
        if isinstance(item, dict)
    ]
    annotations = {str(item.get("id") or ""): item for item in annotations_list}
    fps = _video_fps(video_path)

    for index, pair in enumerate(role_pairs, 1):
        annotation_id = str(pair.get("annotation_id") or "")
        annotation = annotations.get(annotation_id)
        if not annotation:
            raise ValueError(f"{pair['name']} 缺少片段导演身份点")
        point = annotation.get("point")
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"{pair['name']} 缺少有效身份点")

        reused = _try_reuse_ready_role_track(
            project_dir=project_dir,
            annotation=annotation,
            pair=pair,
            add_log=add_log,
        )
        if reused is None:
            reused = _try_reuse_same_role_ready_track(
                project_dir=project_dir,
                video_path=video_path,
                role_name=str(pair["name"]),
                annotation=annotation,
                annotation_id=annotation_id,
                annotations=annotations_list,
                pair=pair,
                add_log=add_log,
            )
        if reused is not None:
            continue

        candidates = _role_track_prompt_candidates(
            video_path=video_path,
            role_name=str(pair["name"]),
            annotation=annotation,
            annotations=annotations_list,
        )
        last_error = "未跟踪到人物"
        success = False
        for candidate_index, candidate in enumerate(candidates, 1):
            track_prompt_time = float(candidate["time"])
            track_prompt_point = [float(candidate["point"][0]), float(candidate["point"][1])]
            add_log(
                f"> [{index}/{len(role_pairs)}] 用手工身份点直接生成 {pair['name']} 的 SAM3 轨迹 "
                f"[候选{candidate_index}/{len(candidates)} {candidate['label']} @ {track_prompt_time:.3f}s]"
            )
            update_annotation(project_dir, annotation_id, track_status="tracking", track_error="")
            frame_idx = max(0, int(round(track_prompt_time * fps)))
            output_dir = Path(project_dir) / "assets" / "tracks" / f"{annotation_id}_sam3"
            output_dir.mkdir(parents=True, exist_ok=True)
            for stale_mask in output_dir.glob("mask_*.png"):
                stale_mask.unlink(missing_ok=True)
            (output_dir / "track_summary.json").unlink(missing_ok=True)

            try:
                with SAM3_TRACK_LOCK:
                    clip_path, prompt_frame, start_frame, clip_frames, track_fps = _make_sam3_window_clip(
                        video_path,
                        frame_idx=frame_idx,
                        max_frames=SAM3_PROTECTION_MAX_FRAMES,
                        job_id=f"{job_id}_{index}_c{candidate_index}",
                    )
                    from spvideo.sam3_tracker import SAM3Tracker

                    tracker = SAM3Tracker()
                    try:
                        propagation_direction = (
                            "backward"
                            if clip_frames > 1 and prompt_frame >= int(clip_frames * 0.75)
                            else "both"
                        )
                        if propagation_direction == "backward":
                            add_log(
                                f"> [{pair['name']}] 身份点靠近片尾，SAM3 改用 backward 传播 "
                                f"(prompt={prompt_frame}/{clip_frames})"
                            )
                        tracked = tracker.track_by_point(
                            video_path=str(clip_path),
                            point=track_prompt_point,
                            frame_idx=prompt_frame,
                            max_frames=clip_frames,
                            output_dir=str(output_dir),
                            propagation_direction=propagation_direction,
                        )
                    finally:
                        tracker.close()
            except Exception as error:
                last_error = str(error)
                add_log(
                    f"> [{pair['name']}] 候选 {candidate_index}/{len(candidates)} "
                    f"SAM3 失败: {error}"
                )
                continue

            summary = {
                "prompt_source": "manual_annotation_direct",
                "annotation_id": annotation_id,
                "video_path": video_path,
                "clip_path": str(clip_path),
                "clip_start_frame": start_frame,
                "clip_start_time": start_frame / fps,
                "frame_idx": frame_idx,
                "clip_prompt_frame": prompt_frame,
                "source_fps": fps,
                "track_fps": track_fps,
                "prompt_point": track_prompt_point,
                "candidate_label": candidate["label"],
                "candidate_time": track_prompt_time,
                "propagation_direction": str(tracked.get("propagation_direction") or propagation_direction),
                "prompt_mode": str(tracked.get("prompt_mode") or ""),
                "num_frames": int(tracked.get("num_frames") or 0),
                "tracked_frames": int(tracked.get("tracked_frames") or 0),
            }
            if summary["tracked_frames"] <= 0:
                last_error = (
                    f"未跟踪到人物（prompt={prompt_frame}/{clip_frames}, "
                    f"point={track_prompt_point}）"
                )
                add_log(f"> [{pair['name']}] 候选 {candidate_index}/{len(candidates)} 轨迹为空，继续尝试")
                continue
            try:
                _validate_sam3_role_track(
                    output_dir=output_dir,
                    point=track_prompt_point,
                    prompt_frame=prompt_frame,
                    tracked_frames=summary["tracked_frames"],
                    total_frames=summary["num_frames"],
                    role_name=str(pair["name"]),
                )
            except Exception as error:
                last_error = (
                    f"{error} "
                    f"(tracked={summary['tracked_frames']}/{summary['num_frames']}, "
                    f"mode={summary['prompt_mode']}, direction={summary['propagation_direction']}, "
                    f"prompt={prompt_frame}/{clip_frames}, point={track_prompt_point})"
                )
                add_log(
                    f"> [{pair['name']}] 候选 {candidate_index}/{len(candidates)} "
                    f"轨迹校验失败: {last_error}"
                )
                continue

            (output_dir / "track_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            update_annotation(
                project_dir,
                annotation_id,
                track_status="ready",
                track_dir=str(output_dir),
                track_summary=str(output_dir / "track_summary.json"),
                tracked_frames=summary["tracked_frames"],
                track_frames=summary["num_frames"],
                track_error="",
            )
            pair["track_dir"] = str(output_dir)
            add_log(f"> [{pair['name']}] SAM3 {summary['tracked_frames']}/{summary['num_frames']} 帧")
            success = True
            break

        if not success:
            update_annotation(project_dir, annotation_id, track_status="failed", track_error=last_error)
            raise ValueError(
                f"{pair['name']} 的 SAM3 轨迹为空，请重新调整身份点"
                f"（已尝试 {len(candidates)} 个候选时间；最后错误: {last_error}）"
            )


def _try_reuse_ready_role_track(
    *,
    project_dir: str,
    annotation: dict[str, Any],
    pair: dict[str, Any],
    add_log,
) -> tuple[float, float] | None:
    """Reuse a ready track when the director annotation already has valid masks."""
    track_status = str(annotation.get("track_status") or "")
    track_dir_value = str(annotation.get("track_dir") or pair.get("track_dir") or "").strip()
    if track_status != "ready" or not track_dir_value:
        return None
    track_dir = Path(track_dir_value)
    if not track_dir.exists():
        return None
    mask_count = len(list(track_dir.glob("mask_*.png")))
    tracked_frames = int(annotation.get("tracked_frames") or mask_count or 0)
    if tracked_frames <= 0 or mask_count <= 0:
        return None
    if not _track_uses_direct_manual_prompt(track_dir):
        add_log(f"> [{pair['name']}] 跳过旧的非纯 SAM3 轨迹 ({track_dir.name})")
        return None
    point = annotation.get("point")
    if not isinstance(point, list) or len(point) != 2:
        return None
    pair["track_dir"] = str(track_dir)
    add_log(
        f"> [{pair['name']}] 复用已有 SAM3 轨迹 "
        f"{tracked_frames} 帧 ({track_dir.name})"
    )
    return float(point[0]), float(point[1])


def _try_reuse_same_role_ready_track(
    *,
    project_dir: str,
    video_path: str,
    role_name: str,
    annotation: dict[str, Any],
    annotation_id: str,
    annotations: list[dict[str, Any]],
    pair: dict[str, Any],
    add_log,
) -> tuple[float, float] | None:
    """Reuse another ready track of the same role on the same clip when possible."""
    anchor = annotation.get("point")
    anchor_segment = str(annotation.get("segment_id") or "").strip()
    ready_items = []
    for item in annotations:
        if (
            item.get("type") != "person"
            or str(item.get("label_name") or "").strip() != role_name
            or str(item.get("id") or "") == annotation_id
            or not _same_video_path(item.get("video_path"), video_path)
            or not str(item.get("track_dir") or "").strip()
            or not (
                str(item.get("track_status") or "") == "ready"
                or _annotation_has_ready_track(item)
            )
        ):
            continue
        item_segment = str(item.get("segment_id") or "").strip()
        if anchor_segment and item_segment and item_segment != anchor_segment:
            continue
        item_point = item.get("point")
        if (
            isinstance(anchor, list)
            and len(anchor) == 2
            and isinstance(item_point, list)
            and len(item_point) == 2
            and math.hypot(float(anchor[0]) - float(item_point[0]), float(anchor[1]) - float(item_point[1])) > 0.20
        ):
            add_log(f"> [{role_name}] 跳过距离身份点过远的旧轨迹 ({str(item.get('id') or '')[:12]})")
            continue
        ready_items.append(item)
    if not ready_items:
        return None
    ready_items.sort(
        key=lambda item: (
            int(item.get("tracked_frames") or 0),
            1 if str(item.get("track_status") or "") == "ready" else 0,
        ),
        reverse=True,
    )
    for item in ready_items:
        track_dir = Path(str(item.get("track_dir") or ""))
        if not track_dir.exists():
            continue
        if not _track_uses_direct_manual_prompt(track_dir):
            continue
        mask_count = len(list(track_dir.glob("mask_*.png")))
        tracked_frames = int(item.get("tracked_frames") or mask_count or 0)
        if tracked_frames <= 0 or mask_count <= 0:
            continue
        point = item.get("point") or pair.get("point") or [0.5, 0.5]
        if not isinstance(point, list) or len(point) != 2:
            point = [0.5, 0.5]
        pair["track_dir"] = str(track_dir)
        # Mirror ready state onto the director annotation so later retries skip work.
        if annotation_id:
            update_annotation(
                project_dir,
                annotation_id,
                track_status="ready",
                track_dir=str(track_dir),
                track_summary=str(item.get("track_summary") or ""),
                tracked_frames=tracked_frames,
                track_frames=int(item.get("track_frames") or tracked_frames),
                track_error="",
            )
        add_log(
            f"> [{role_name}] 复用同角色已有轨迹 "
            f"{tracked_frames} 帧 ({track_dir.name})"
        )
        return float(point[0]), float(point[1])
    return None


def _role_track_prompt_candidates(
    *,
    video_path: str,
    role_name: str,
    annotation: dict[str, Any],
    annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build ordered SAM3 prompt candidates for a role from real role marks only."""
    point = annotation.get("point")
    if not isinstance(point, list) or len(point) != 2:
        raise ValueError(f"{role_name} 缺少有效身份点")

    raw_time = float(annotation.get("time") or 0.0)
    prompt_time = _clamp_video_time(video_path, raw_time)
    candidates: list[dict[str, Any]] = []
    def add_candidate(label: str, time_seconds: float, prompt_point: list[float]) -> None:
        clamped = _clamp_video_time(video_path, time_seconds)
        key = (
            round(clamped, 3),
            round(float(prompt_point[0]), 4),
            round(float(prompt_point[1]), 4),
        )
        if any(
            abs(key[0] - round(float(item["time"]), 3)) < 0.001
            and abs(key[1] - round(float(item["point"][0]), 4)) < 0.0001
            and abs(key[2] - round(float(item["point"][1]), 4)) < 0.0001
            for item in candidates
        ):
            return
        candidates.append(
            {
                "label": label,
                "time": clamped,
                "point": [float(prompt_point[0]), float(prompt_point[1])],
            }
        )

    # The director's exact mark is the authoritative SAM3 prompt. Other manually
    # placed marks are fallback candidates; no detector is allowed to move them.
    add_candidate("导演身份点", prompt_time, [float(point[0]), float(point[1])])

    current_segment = str(annotation.get("segment_id") or "").strip()
    current_id = str(annotation.get("id") or "")
    same_role = []
    for item in annotations:
        item_point = item.get("point")
        item_segment = str(item.get("segment_id") or "").strip()
        if (
            item.get("type") != "person"
            or str(item.get("label_name") or "").strip() != role_name
            or str(item.get("id") or "") == current_id
            or not _same_video_path(item.get("video_path"), video_path)
            or not isinstance(item_point, list)
            or len(item_point) != 2
        ):
            continue
        if current_segment and item_segment and item_segment != current_segment:
            continue
        if math.hypot(float(point[0]) - float(item_point[0]), float(point[1]) - float(item_point[1])) > 0.20:
            continue
        same_role.append(item)
    same_role.sort(key=_mapping_item_time)
    for item in same_role:
        item_time = _clamp_video_time(video_path, _mapping_item_time(item))
        add_candidate(
            f"同角色手工标注 {str(item.get('id') or '')[:12]}",
            item_time,
            [float(item["point"][0]), float(item["point"][1])],
        )

    return candidates


def _track_uses_sam3_text_objects(track_dir: Path) -> bool:
    summary_path = track_dir / "track_summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        str(summary.get("prompt_source") or "") == "sam3_text_person_manual_identity"
        and str(summary.get("prompt_mode") or "") == "text_objects"
        and summary.get("sam3_object_id") is not None
    )


def _track_uses_direct_manual_prompt(track_dir: Path) -> bool:
    summary_path = track_dir / "track_summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        str(summary.get("prompt_source") or "") == "manual_annotation_direct"
        and str(summary.get("prompt_mode") or "") == "points"
    )


def _snap_role_point_to_person(
    *,
    video_path: str,
    time_seconds: float,
    point: list[float],
    role_name: str,
    job_id: str,
    max_distance: float = 0.08,
    return_box: bool = False,
) -> Any:
    """Snap a rough role mark to the torso of the detected person it identifies."""
    import cv2
    import subprocess
    from spvideo.ffmpeg_tools import ffmpeg_path, subprocess_no_window_kwargs

    ffmpeg = ffmpeg_path()
    frame_ext = ".jpg"
    probe_dir = ROOT.parent / ".sam3_tmp"
    probe_dir.mkdir(parents=True, exist_ok=True)
    frame_path = probe_dir / f"{job_id}_role_probe{frame_ext}"
    frame_path.unlink(missing_ok=True)

    duration = _video_duration(video_path)
    requested_time = max(0.0, float(time_seconds or 0.0))
    candidate_times: list[float] = []
    for value in [requested_time, _clamp_video_time(video_path, requested_time)]:
        if not any(abs(value - existing) < 0.001 for existing in candidate_times):
            candidate_times.append(value)

    last_error = "unknown"
    for candidate_time in candidate_times:
        for accurate_seek in (True, False):
            frame_path.unlink(missing_ok=True)
            if accurate_seek:
                args = [
                    ffmpeg, "-y", "-loglevel", "error",
                    "-i", str(video_path),
                    "-ss", f"{candidate_time:.3f}",
                    "-frames:v", "1", "-c:v", "mjpeg", "-q:v", "2", "-update", "1",
                    str(frame_path),
                ]
            else:
                args = [
                    ffmpeg, "-y", "-loglevel", "error",
                    "-ss", f"{candidate_time:.3f}",
                    "-i", str(video_path),
                    "-frames:v", "1", "-c:v", "mjpeg", "-q:v", "2", "-update", "1",
                    str(frame_path),
                ]
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                **subprocess_no_window_kwargs(),
            )
            if result.returncode == 0 and frame_path.exists() and frame_path.stat().st_size > 0:
                break
            detail = (result.stderr.strip() or result.stdout.strip() or "ffmpeg未生成帧")
            last_error = (
                f"{detail[-240:]} "
                f"(returncode={result.returncode}, time={candidate_time:.3f}s, duration={duration:.3f}s)"
            )
        else:
            continue
        break
    else:
        raise ValueError(f"无法读取 {role_name} 的身份标注帧: {last_error}")

    frame = cv2.imread(str(frame_path))
    if frame is None:
        frame_path.unlink(missing_ok=True)
        raise ValueError(f"无法读取 {role_name} 的身份标注帧")
    height, width = frame.shape[:2]

    from spvideo.scene_detector import detect_persons_in_frame

    persons = detect_persons_in_frame(frame_path, conf_threshold=0.25, device="cpu")
    if not persons:
        frame_path.unlink(missing_ok=True)
        raise ValueError(f"{role_name} 的标注帧未检测到人物")
    pixel_x = point[0] * width
    pixel_y = point[1] * height

    def distance_to_box(person: dict[str, Any]) -> float:
        x1, y1, x2, y2 = [float(value) for value in person["bbox"]]
        dx = max(x1 - pixel_x, 0.0, pixel_x - x2)
        dy = max(y1 - pixel_y, 0.0, pixel_y - y2)
        return math.hypot(dx / width, dy / height)

    containing = [
        person for person in persons
        if float(person["bbox"][0]) <= pixel_x <= float(person["bbox"][2])
        and float(person["bbox"][1]) <= pixel_y <= float(person["bbox"][3])
    ]
    person = min(containing, key=lambda item: float(item.get("area_ratio") or 1.0), default=None)
    if person is None:
        person = min(persons, key=distance_to_box)
        if distance_to_box(person) > max_distance:
            frame_path.unlink(missing_ok=True)
            raise ValueError(f"{role_name} 的身份点距离任何人物过远，请重新标注")
    x1, y1, x2, y2 = [float(value) for value in person["bbox"]]
    snapped = [round(((x1 + x2) / 2.0) / width, 6), round((y1 + (y2 - y1) * 0.38) / height, 6)]
    frame_path.unlink(missing_ok=True)
    if return_box:
        return {
            "point": snapped,
            "box_xywh": [
                round(x1 / width, 6),
                round(y1 / height, 6),
                round((x2 - x1) / width, 6),
                round((y2 - y1) / height, 6),
            ],
        }
    return snapped


def _validate_sam3_role_track(
    *,
    output_dir: Path,
    point: list[float],
    source_shape: dict[str, Any] | None = None,
    prompt_frame: int,
    tracked_frames: int,
    total_frames: int,
    role_name: str,
) -> None:
    """Reject empty, tiny, oversized, or discontinuous role tracks before Wan runs."""
    import cv2
    import numpy as np

    coverage = tracked_frames / max(1, total_frames)
    if coverage < 0.75:
        raise ValueError(f"{role_name} 的 SAM3 轨迹覆盖率过低: {coverage:.0%}")
    mask_path = output_dir / f"mask_{prompt_frame:04d}.png"
    from spvideo.foreground_compositor import _read_grayscale_image

    mask = _read_grayscale_image(mask_path)
    if mask is None:
        raise ValueError(f"{role_name} 的标注帧没有生成有效 mask")
    height, width = mask.shape[:2]
    area_ratio = float(np.count_nonzero(mask > 127)) / float(width * height)
    if not 0.01 <= area_ratio <= 0.65:
        raise ValueError(f"{role_name} 的人物 mask 面积异常: {area_ratio:.1%}")
    px = max(0, min(width - 1, int(round(point[0] * width))))
    py = max(0, min(height - 1, int(round(point[1] * height))))
    radius = max(2, min(width, height) // 100)
    neighborhood = mask[max(0, py - radius): py + radius + 1, max(0, px - radius): px + radius + 1]
    if neighborhood.size == 0 or not np.any(neighborhood > 127):
        raise ValueError(f"{role_name} 的 SAM3 mask 没有覆盖手工身份点")
    shape_overlap = _identity_shape_mask_overlap(mask, source_shape)
    if shape_overlap is not None and shape_overlap < SAM3_SHAPE_MIN_OVERLAP:
        raise ValueError(f"{role_name} 的 SAM3 mask 没有覆盖手绘身份区域: {shape_overlap:.0%}")


def _run_wan_multi_role_transfer(
    *,
    video_path: str,
    role_pairs: list[dict[str, Any]],
    output_dir: Path,
    api_key: str,
    add_log,
) -> dict[str, Any]:
    """Generate each role from a role-focused guide video and return Wan's raw output."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from spvideo.foreground_compositor import (
        build_focus_video,
        build_focus_video_from_color_mask_video,
        masked_mean_absdiff,
    )
    from spvideo.wan22_client import Wan22MixClient

    if not api_key:
        raise ValueError("wan22_api_key_missing")

    role_limit = _wan_multi_role_limit()
    focus_settings = _wan_focus_settings()
    active_role_pairs = role_pairs if role_limit <= 0 else role_pairs[:role_limit]
    direct_single_role = len(role_pairs) == 1
    if not direct_single_role and not _wan22_allow_experimental_multi_focus():
        raise RuntimeError(
            "Wan2.2 云端 animate-mix 接口不支持 SAM3/color mask 条件输入，只能上传 image_url + video_url。"
            "多人片段直接跑会把其他人当成普通视频内容，暗背景 guide 也会破坏遮挡。"
            "请改用 SCAIL2/ComfyUI 蒙版工作流；若只是调试旧实验方案，可设置 "
            "WAN22_ALLOW_EXPERIMENTAL_MULTI_FOCUS=1。"
        )
    if direct_single_role:
        add_log(f"> Wan2.2 单角色任务: {active_role_pairs[0]['name']}；直接上传原始片段")
    elif len(active_role_pairs) < len(role_pairs):
        add_log(
            f"> Wan2.2 首轮试跑模式: 共 {len(role_pairs)} 人，本次只提交第 1 个角色 "
            f"({active_role_pairs[0]['name']})；只上传 1 个聚焦视频"
        )
    else:
        add_log(f"> Wan2.2 多人任务: {len(role_pairs)} 人；先为每个角色生成聚焦视频再并发提交")
    focus_items: list[dict[str, Any]] = []
    remote_mask_result: dict[str, Any] | None = None
    remote_mask_pose_path = ""
    remote_mask_error = ""
    if not direct_single_role:
        try:
            from spvideo.scail2_client import Scail2Client

            add_log("> Wan2.2 多人模式: 调用远程 SAM3 生成彩色身份蒙版")
            remote_client = Scail2Client()
            remote_mask_result = remote_client.inspect_masks(
                video_path=video_path,
                ref_images=[str(pair.get("ref_image") or "") for pair in active_role_pairs],
                role_names=[str(pair.get("name") or "") for pair in active_role_pairs],
                sam_text=_default_scail2_sam_text(len(active_role_pairs)),
                normalize_size=True,
                on_progress=lambda message: add_log(f"> [Wan2.2 remote SAM3] {message}"),
            )
            remote_mask_pose_path = str(
                ((remote_mask_result.get("mask_output_paths") or {}).get("pose") or [""])[0]
            )
            if remote_mask_pose_path:
                add_log(f"> Wan2.2 使用远程 SAM3 彩色蒙版: {remote_mask_pose_path}")
        except Exception as exc:  # noqa: BLE001
            remote_mask_result = None
            remote_mask_pose_path = ""
            remote_mask_error = str(exc) or type(exc).__name__
            add_log(f"> 警告: Wan2.2 远程 SAM3 彩色蒙版不可用: {exc}")
        if not remote_mask_pose_path and not _wan22_allow_local_mask_fallback():
            message = remote_mask_error or "remote SAM3 did not return a pose mask video"
            raise RuntimeError(
                "Wan2.2 多人模式必须拿到远程 SAM3 彩色蒙版；"
                f"本地旧轨迹回退已关闭，避免继续生成脏结果。{message}"
            )
        if not remote_mask_pose_path:
            add_log("> Wan2.2 已启用本地旧轨迹回退；这只适合调试，不建议用于正式多人换人")
    if direct_single_role:
        pair = active_role_pairs[0]
        add_log(f"> [{pair['name']}] 使用原始片段作为 Wan2.2 驱动视频: {video_path}")
        focus_items.append({
            "index": 0,
            "pair": pair,
            "focus_path": str(video_path),
            "focus": None,
            "input_mode": "original_video",
        })
    else:
        for index, pair in enumerate(active_role_pairs):
            name = str(pair["name"])
            focus_path = output_dir / f"wan22_focus_{index + 1:02d}_{_safe_output_name(name)}_{Path(video_path).stem}.mp4"
            add_log(f"> [{index + 1}/{len(active_role_pairs)}] 生成 {name} 的目标聚焦视频")
            mask_color = WAN22_MASK_COLOR_KEYS[index]
            if remote_mask_pose_path:
                focus = build_focus_video_from_color_mask_video(
                    base_video=video_path,
                    mask_video=remote_mask_pose_path,
                    output_path=focus_path,
                    color=mask_color,
                    feather_pixels=focus_settings["feather_pixels"],
                    erode_pixels=focus_settings["erode_pixels"],
                    dilate_pixels=focus_settings["dilate_pixels"],
                    background_dim=focus_settings["background_dim"],
                    background_blur_pixels=focus_settings["background_blur_pixels"],
                )
                input_mode = "remote_sam3_color_mask"
            else:
                focus = build_focus_video(
                    base_video=video_path,
                    track_dir=str(pair["track_dir"]),
                    output_path=focus_path,
                    feather_pixels=focus_settings["feather_pixels"],
                    dilate_pixels=focus_settings["dilate_pixels"],
                    background_dim=focus_settings["background_dim"],
                    background_blur_pixels=focus_settings["background_blur_pixels"],
                )
                input_mode = "local_sam3_track"
            add_log(
                f"> [{name}] 聚焦视频完成: {focus.get('focused_frames')}/{focus.get('total_frames')} 帧 "
                f"(feather={focus.get('feather_pixels')}, erode={focus.get('erode_pixels')}, "
                f"dilate={focus.get('dilate_pixels')}, "
                f"bg_dim={focus.get('background_dim')}, bg_blur={focus.get('background_blur_pixels')}, "
                f"input={input_mode}, color={mask_color}, "
                f"mask_area={float(focus.get('mean_mask_ratio') or 0.0):.1%})"
            )
            add_log(f"> [{name}] Wan2.2 guide video: {focus_path}")
            focus_items.append({
                "index": index,
                "pair": pair,
                "focus_path": str(focus_path),
                "focus": focus,
                "input_mode": input_mode,
                "mask_color": mask_color,
            })

    def run_role(item: dict[str, Any]) -> dict[str, Any]:
        index = int(item["index"])
        pair = item["pair"]
        name = str(pair["name"])
        client = Wan22MixClient(api_key)
        input_label = "原始片段" if item.get("input_mode") == "original_video" else "聚焦视频"
        add_log(f"> [{index + 1}/{len(active_role_pairs)}] 上传 {name} 目标图 + {input_label}并提交 Wan2.2")
        image_url = client.upload(str(pair["ref_image"]))
        video_url = client.upload(str(item["focus_path"]))
        task_id = client.create_task(image_url, video_url)
        add_log(f"> [{name}] Wan2.2 task: {task_id}")
        state_cache = ""

        def on_status(status: str, elapsed: float) -> None:
            nonlocal state_cache
            if status != state_cache:
                state_cache = status
                add_log(f"> [{name}] {status} ({elapsed:.0f}s)")

        result = client.poll_task(task_id, interval=10, timeout=1200, on_status=on_status)
        result_url = result["output"]["results"]["video_url"]
        generated_path = output_dir / f"wan22_{index + 1:02d}_{_safe_output_name(name)}_{Path(video_path).stem}.mp4"
        client.download_video(result_url, str(generated_path))
        add_log(f"> [{name}] Wan2.2 回传视频已保存: {generated_path}")
        diff: dict[str, object] = {}
        if item.get("input_mode") == "focused_video":
            try:
                diff = masked_mean_absdiff(video_path, generated_path, str(pair["track_dir"]))
                mean_absdiff = float(diff.get("mean_absdiff") or 0.0)
                add_log(f"> [{name}] 目标区域变化量: {mean_absdiff:.1f}")
                if int(diff.get("samples") or 0) > 0 and mean_absdiff < 12.0:
                    add_log(f"> 警告: {name} 的目标区域变化偏低，Wan2.2 可能仍未按参考图替换")
            except Exception as exc:  # noqa: BLE001
                add_log(f"> 警告: {name} 的目标区域变化量检测失败: {exc}")
        return {
            "name": name,
            "task_id": task_id,
            "generated_path": str(generated_path),
            "output_path": str(generated_path),
            "track_dir": str(pair.get("track_dir") or ""),
            "focus_path": str(item["focus_path"]),
            "focus": item.get("focus") or {},
            "input_mode": str(item.get("input_mode") or "focused_video"),
            "mask_color": str(item.get("mask_color") or ""),
            "diff": diff,
        }

    generated: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=len(active_role_pairs), thread_name_prefix="wan22-role") as executor:
        futures = {executor.submit(run_role, item): item for item in focus_items}
        for future in as_completed(futures):
            item = futures[future]
            name = str(item["pair"]["name"])
            try:
                generated.append(future.result())
            except Exception as exc:  # noqa: BLE001
                message = str(exc) or type(exc).__name__
                failed.append({"name": name, "error": message})
                add_log(f"> [{name}] Wan2.2 失败: {message}")

    if failed:
        success_names = [str(item["name"]) for item in generated]
        failed_text = "；".join(f"{item['name']}: {item['error']}" for item in failed)
        success_text = "、".join(success_names) if success_names else "无"
        raise RuntimeError(f"Wan2.2 部分角色失败: {failed_text}；已成功: {success_text}")

    generated_by_name = {item["name"]: item for item in generated}
    ordered_generated = [generated_by_name[str(pair["name"])] for pair in active_role_pairs]
    output_path = str(ordered_generated[0]["generated_path"])
    for item in ordered_generated:
        add_log(f"> [{item['name']}] Wan2.2 output: {item['generated_path']}")
    add_log(f"> 跳过本地遮罩合成，直接使用 Wan2.2 回传视频: {output_path}")

    return {
        "output_path": output_path,
        "primary_output_path": output_path,
        "output_paths": [str(item["generated_path"]) for item in ordered_generated],
        "role_output_paths": [
            {
                "name": str(item["name"]),
                "output_path": str(item["generated_path"]),
                "task_id": str(item.get("task_id") or ""),
                "input_mode": str(item.get("input_mode") or "focused_video"),
                "mask_color": str(item.get("mask_color") or ""),
                "focus_path": str(item.get("focus_path") or ""),
                "focus": item.get("focus") or {},
            }
            for item in ordered_generated
        ],
        "wan22_mask_output_paths": (remote_mask_result or {}).get("mask_output_paths") or {},
        "backend": "wan22_raw",
        "role_tasks": ordered_generated,
        "role_count": len(role_pairs),
        "active_role_count": len(active_role_pairs),
    }


def _safe_output_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value).strip("_") or "role"


def _wan_api_key() -> str:
    return (
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("ALIYUN_DASHSCOPE_API_KEY")
        or os.environ.get("WAN22_API_KEY")
        or DEFAULT_WAN22_API_KEY
        or ""
    ).strip()


def _plain_secret(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.startswith("***"):
        return ""
    return text


def _windows_user_env(name: str) -> str:
    if os.name != "nt":
        return ""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
        return str(value or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _runtime_or_user_env(name: str) -> str:
    return str(os.environ.get(name) or _windows_user_env(name) or "").strip()


def _seedance_a_api_key() -> str:
    return (
        _plain_secret(_runtime_or_user_env("SEEDANCE_A_API_KEY"))
        or _plain_secret(_runtime_or_user_env("TIANYUE_A_API_KEY"))
        or _plain_secret(_runtime_or_user_env("TIANYUE_API_KEY"))
        or _plain_secret(_runtime_or_user_env("SEEDANCE_API_KEY"))
        or _plain_secret(_runtime_or_user_env("NEWAPI_API_KEY"))
        or _plain_secret(_runtime_or_user_env("LINDONG_API_KEY"))
        or _plain_secret(_runtime_or_user_env("ZHENZHEN_API_KEY"))
        or _plain_secret(_runtime_or_user_env("YUANQI_API_KEY"))
        or _plain_secret(_runtime_or_user_env("YUANQI_UPLOAD_API_KEY"))
        or _plain_secret(_runtime_or_user_env("WAN22_API_KEY"))
    ).strip()


def _seedance_a_base_url() -> str:
    return (
        _runtime_or_user_env("SEEDANCE_A_BASE_URL")
        or _runtime_or_user_env("TIANYUE_A_BASE_URL")
        or _runtime_or_user_env("NEWAPI_BASE_URL")
        or _runtime_or_user_env("ZHENZHEN_BASE_URL")
        or DEFAULT_SEEDANCE_A_BASE_URL
    ).strip().rstrip("/")


def _seedance_a_key_source_hint() -> str:
    env_keys = (
        "SEEDANCE_A_API_KEY",
        "TIANYUE_A_API_KEY",
        "TIANYUE_API_KEY",
        "SEEDANCE_API_KEY",
        "NEWAPI_API_KEY",
        "LINDONG_API_KEY",
        "ZHENZHEN_API_KEY",
        "YUANQI_API_KEY",
        "YUANQI_UPLOAD_API_KEY",
        "WAN22_API_KEY",
    )
    for key in env_keys:
        if _plain_secret(_runtime_or_user_env(key)):
            return f"environment variable {key}"
    return "Mode2 04 视频生成填写的通道A API Key 或环境变量 SEEDANCE_A_API_KEY"


def _seedance_a_friendly_error(
    message: str,
    status_code: int | None = None,
    key_source_hint: str = "",
) -> str:
    text = str(message or "").strip()
    if status_code in {401, 403} or "invalid token" in text.lower():
        hint = str(key_source_hint or _seedance_a_key_source_hint()).strip()
        return (
            f"Seedance 通道A鉴权失败：{text or 'Invalid token'}。"
            f"请检查{hint}，"
            "保存有效值后刷新本页再试。"
        )
    return text


def _seedance_a_config_sources(payload: dict[str, Any], request: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[int] = set()

    def add_source(source: Any) -> None:
        if not isinstance(source, dict):
            return
        marker = id(source)
        if marker in seen:
            return
        seen.add(marker)
        sources.append(source)

    for source in (request, payload):
        add_source(source)
        if not isinstance(source, dict):
            continue
        for key in ("relay", "relay_config", "relayConfig", "seedance_relay", "seedanceRelay", "channel_a", "channelA"):
            add_source(source.get(key))
    return sources


def _seedance_a_request_config_value(
    payload: dict[str, Any],
    request: dict[str, Any],
    keys: tuple[str, ...],
) -> Any:
    sources = _seedance_a_config_sources(payload, request)
    for key in keys:
        for source in sources:
            value = source.get(key)
            if value is not None:
                return value
    return ""


def _seedance_a_request_api_key(payload: dict[str, Any], request: dict[str, Any]) -> tuple[str, str]:
    api_key = _plain_secret(
        _seedance_a_request_config_value(
            payload,
            request,
            (
                "relay_api_key",
                "relayApiKey",
                "seedance_api_key",
                "seedanceApiKey",
                "api_key",
                "apiKey",
            ),
        )
    )
    if api_key:
        return api_key, "Mode2 04 视频生成填写的通道A API Key"
    return _seedance_a_api_key(), _seedance_a_key_source_hint()


def _seedance_a_request_base_url(payload: dict[str, Any], request: dict[str, Any]) -> str:
    base_url = _plain_secret(
        _seedance_a_request_config_value(
            payload,
            request,
            (
                "relay_base_url",
                "relayBaseUrl",
                "seedance_base_url",
                "seedanceBaseUrl",
                "base_url",
                "baseUrl",
            ),
        )
    )
    return (base_url or _seedance_a_base_url()).strip().rstrip("/")


def _seedance_a_upload_base_url() -> str:
    return (
        os.environ.get("SEEDANCE_A_UPLOAD_BASE_URL")
        or os.environ.get("YUANQI_UPLOAD_BASE_URL")
        or os.environ.get("PUBLIC_UPLOAD_URL")
        or DEFAULT_SEEDANCE_A_UPLOAD_BASE_URL
    ).strip().rstrip("/")


def _seedance_a_ref_to_data_url(ref: Any) -> str:
    text = str(ref or "").strip()
    if not text:
        return ""
    if text.startswith("data:"):
        return text
    try:
        if text.startswith(("http://", "https://")):
            response = requests.get(text, timeout=60)
            if not response.ok:
                return ""
            mime = response.headers.get("content-type") or "application/octet-stream"
            import base64

            return f"data:{mime};base64,{base64.b64encode(response.content).decode('ascii')}"
        local_path = Path(_decode_media_ref_path(text))
        if not local_path.exists() or not local_path.is_file():
            return ""
        mime = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
        import base64

        return f"data:{mime};base64,{base64.b64encode(local_path.read_bytes()).decode('ascii')}"
    except Exception:  # noqa: BLE001
        return ""


def _decode_media_ref_path(ref: str) -> str:
    text = str(ref or "").strip()
    if not text:
        return ""
    if text.startswith("/media"):
        parsed = urlparse(text)
        query = parse_qs(parsed.query)
        return unquote((query.get("path") or [""])[0])
    if text.lower().startswith("file://"):
        return unquote(urlparse(text).path).lstrip("/") if os.name == "nt" else unquote(urlparse(text).path)
    return text


def _seedance_a_response_error_message(data: dict[str, Any], fallback: str, status_code: int) -> str:
    error = data.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or fallback or f"天悦A HTTP {status_code}")
    return str(error or data.get("message") or fallback or f"天悦A HTTP {status_code}")


def _seedance_a_is_sensitive_image_error(data: dict[str, Any], text: str) -> bool:
    error = data.get("error") if isinstance(data, dict) else {}
    code = str(error.get("code") if isinstance(error, dict) else "").strip()
    message = str(error.get("message") if isinstance(error, dict) else "").strip()
    haystack = " ".join([code, message, str(text or "")]).lower()
    return (
        "inputimagesensitivecontentdetected" in haystack
        or "input image may contain real person" in haystack
        or "privacyinformation" in haystack
    )


def _seedance_a_upload_ref(ref: Any, api_key: str) -> str:
    text = str(ref or "").strip()
    if not text:
        return ""
    if text.startswith(("http://", "https://")):
        return text

    from spvideo.yuanqi_upload import upload_file_for_url

    upload_base = _seedance_a_upload_base_url()
    if text.startswith("data:"):
        import base64

        header, _, encoded = text.partition(",")
        if not encoded:
            return ""
        content_type = header.split(";", 1)[0].replace("data:", "") or "application/octet-stream"
        ext = mimetypes.guess_extension(content_type) or ".bin"
        temp_dir = ROOT.parent / ".seedance_upload_cache"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"seedance_ref_{uuid.uuid4().hex}{ext}"
        temp_path.write_bytes(base64.b64decode(encoded))
        try:
            return str(upload_file_for_url(temp_path, api_key=api_key, base_url=upload_base).get("url") or "")
        finally:
            temp_path.unlink(missing_ok=True)

    local_path = Path(_decode_media_ref_path(text))
    if not local_path.exists() or not local_path.is_file():
        return ""
    return str(upload_file_for_url(local_path, api_key=api_key, base_url=upload_base).get("url") or "")


def _seedance_a_output_dir(project_dir: str, source_video_path: str) -> Path:
    project_text = str(project_dir or "").strip()
    if project_text:
        base = Path(project_text)
    else:
        source = Path(str(source_video_path or "")).expanduser()
        base = source.parent if source.name else ROOT.parent
    output_dir = base / "04_AI输出成片"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _normalize_seedance_a_model(model: str) -> str:
    raw = str(model or "").strip()
    if raw.startswith("doubao-seedance-"):
        return raw
    lower = raw.lower()
    if "mini" in lower:
        return "doubao-seedance-2-0-mini-260615"
    if "fast" in lower:
        return "doubao-seedance-2-0-fast-260128"
    return "doubao-seedance-2-0-260128"


def _seedance_a_build_content(
    prompt: str,
    *,
    first_frame: str = "",
    last_frame: str = "",
    ref_images: list[str] | None = None,
    videos: list[str] | None = None,
    audios: list[str] | None = None,
    api_key: str,
    logic_variant: str = "",
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": str(prompt)}]
    has_first = bool(str(first_frame or "").strip())
    has_last = bool(str(last_frame or "").strip())
    use_data_urls = str(logic_variant or "").strip() == "video-v2"

    if has_first:
        url = _seedance_a_ref_to_data_url(first_frame) if use_data_urls else _seedance_a_upload_ref(first_frame, api_key)
        if not url:
            raise RuntimeError("Seedance first_frame 处理失败")
        entry: dict[str, Any] = {"type": "image_url", "image_url": {"url": url}}
        if has_last:
            entry["role"] = "first_frame"
        content.append(entry)

    if has_first and has_last:
        url = _seedance_a_ref_to_data_url(last_frame) if use_data_urls else _seedance_a_upload_ref(last_frame, api_key)
        if not url:
            raise RuntimeError("Seedance last_frame 处理失败")
        content.append({"type": "image_url", "image_url": {"url": url}, "role": "last_frame"})

    for ref in _string_list(ref_images or []):
        url = _seedance_a_ref_to_data_url(ref) if use_data_urls else _seedance_a_upload_ref(ref, api_key)
        if url:
            content.append({"type": "image_url", "image_url": {"url": url}, "role": "reference_image"})

    for index, ref in enumerate(_string_list(videos or []), 1):
        if not ref:
            continue
        url = _seedance_a_upload_ref(ref, api_key)
        if not url:
            raise RuntimeError(f"Seedance reference_video {index} 处理失败")
        content.append({"type": "video_url", "video_url": {"url": url}, "role": "reference_video"})

    for index, ref in enumerate(_string_list(audios or []), 1):
        if not ref:
            continue
        url = _seedance_a_upload_ref(ref, api_key)
        if not url:
            raise RuntimeError(f"Seedance reference_audio {index} 处理失败")
        content.append({"type": "audio_url", "audio_url": {"url": url}, "role": "reference_audio"})

    return content


def _submit_seedance_a_task(payload: dict[str, Any]) -> dict[str, Any]:
    request = payload.get("payload") if isinstance(payload.get("payload"), dict) else payload
    model = _normalize_seedance_a_model(str(request.get("model") or "").strip())
    prompt = str(request.get("prompt") or "").strip()
    logic_variant = str(request.get("__logicVariant") or request.get("logicVariant") or "").strip()
    if not model:
        raise ValueError("model 必填")
    if not prompt:
        raise ValueError("prompt 不得为空")

    api_key, api_key_hint = _seedance_a_request_api_key(payload, request)
    if not api_key:
        raise ValueError("缺少 Seedance 通道A API Key，请在 Mode2 04 视频生成填写通道A API Key，或设置本项目环境变量 SEEDANCE_A_API_KEY。")

    warnings: list[str] = []
    duration = max(1, int(float(request.get("seconds") or request.get("duration") or 5)))
    ratio = str(request.get("ratio") or request.get("size") or "9:16").strip() or "9:16"
    resolution = str(request.get("resolution") or ("720p" if "720p" in model else "480p")).strip() or "480p"
    try:
        seed = int(request.get("seed") or 0)
    except (TypeError, ValueError):
        seed = 0
    generate_audio = request.get("generate_audio", request.get("generateAudio", True))
    return_last_frame = request.get("return_last_frame", request.get("returnLastFrame", False))
    watermark = request.get("watermark", False)
    web_search = request.get("web_search", request.get("webSearch", False))
    first_frame = str(request.get("first_frame") or request.get("firstFrame") or "").strip()
    last_frame = str(request.get("last_frame") or request.get("lastFrame") or "").strip()
    ref_images = _string_list(request.get("images") or request.get("refImages"))
    videos = _string_list(request.get("videos"))
    audios = _string_list(request.get("audios"))
    project_dir = str(payload.get("project_dir") or payload.get("projectDir") or "").strip()
    image_deny_keys = _mode2_seedance_generation_image_deny_keys(project_dir)
    first_candidates = _mode2_filter_seedance_generation_images(
        [first_frame] if first_frame else [],
        warnings,
        field="first_frame",
        deny_keys=image_deny_keys,
    )
    last_candidates = _mode2_filter_seedance_generation_images(
        [last_frame] if last_frame else [],
        warnings,
        field="last_frame",
        deny_keys=image_deny_keys,
    )
    first_frame = first_candidates[0] if first_candidates else ""
    last_frame = last_candidates[0] if last_candidates else ""
    ref_images = _mode2_filter_seedance_generation_images(
        ref_images,
        warnings,
        field="images/refImages",
        deny_keys=image_deny_keys,
    )
    ignored_mask_fields = [
        key for key in ("mask", "mask_path", "maskPath", "mask_image", "maskImage", "candidate_mask_path")
        if str(request.get(key) or "").strip()
    ]
    if ignored_mask_fields:
        warnings.append(
            "Mode2 safety: ignored mask field(s) "
            + ", ".join(ignored_mask_fields)
            + "; Seedance receives no masks."
        )

    content = _seedance_a_build_content(
        prompt,
        first_frame=first_frame,
        last_frame=last_frame,
        ref_images=ref_images,
        videos=videos,
        audios=audios,
        api_key=api_key,
    )
    non_text_content = [item for item in content if item.get("type") != "text"]
    metadata: dict[str, Any] = {
        "content": non_text_content,
        "ratio": ratio,
        "resolution": resolution,
        "generate_audio": generate_audio is not False,
        "return_last_frame": return_last_frame is True,
        "watermark": watermark is True,
    }
    if web_search is True:
        metadata["tools"] = [{"type": "web_search"}]
    if seed > 0:
        metadata["seed"] = seed

    upstream_payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "duration": duration,
        "metadata": metadata,
    }

    base_url = _seedance_a_request_base_url(payload, request)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def post_generation() -> tuple[requests.Response, str, dict[str, Any]]:
        posted = requests.post(
            f"{base_url}/v1/video/generations",
            headers=headers,
            json=upstream_payload,
            timeout=600,
        )
        posted_text = posted.text
        try:
            posted_data = posted.json()
        except ValueError:
            posted_data = {"_raw": posted_text}
        return posted, posted_text, posted_data

    response, text, data = post_generation()
    image_content = [item for item in non_text_content if item.get("type") == "image_url"]
    if (
        not response.ok
        and image_content
        and _seedance_a_is_sensitive_image_error(data, text)
        and request.get("auto_drop_sensitive_images", request.get("autoDropSensitiveImages", True)) is not False
    ):
        image_count = len(image_content)
        content = [item for item in content if item.get("type") != "image_url"]
        non_text_content = [item for item in content if item.get("type") != "text"]
        metadata["content"] = non_text_content
        upstream_payload["metadata"] = metadata
        warnings.append(
            f"上游拒绝 {image_count} 张参考图：疑似真人隐私图片。已自动移除图片，仅用视频/音频/提示词重试。"
        )
        response, text, data = post_generation()
    if not response.ok:
        message = _seedance_a_response_error_message(data, text, response.status_code)
        raise RuntimeError(_seedance_a_friendly_error(str(message)[:800], response.status_code, api_key_hint))

    task_id = str(
        ((data.get("data") or {}).get("task_id") if isinstance(data.get("data"), dict) else "")
        or data.get("task_id")
        or data.get("id")
        or ""
    ).strip()
    if not task_id:
        raise RuntimeError(f"天悦A未返回 task_id: {str(data)[:500]}")

    task_name = str(payload.get("taskName") or payload.get("task_name") or "shot_retake").strip()
    with SEEDANCE_A_TASKS_LOCK:
        SEEDANCE_A_TASKS[task_id] = {
            "api_key": api_key,
            "base_url": base_url,
            "project_dir": str(payload.get("project_dir") or payload.get("projectDir") or "").strip(),
            "source_video_path": str(payload.get("source_video_path") or payload.get("video_path") or "").strip(),
            "task_name": task_name,
            "segment_id": str(payload.get("segment_id") or "").strip(),
            "submitted_at": time.time(),
            "output_path": "",
            "raw_submit": data,
            "model": model,
        }

    return {
        "success": True,
        "task_id": task_id,
        "taskId": task_id,
        "status": "SUBMITTED",
        "model": model,
        "uploaded": {
            "images": [item.get("image_url", {}).get("url") for item in non_text_content if item.get("type") == "image_url"],
            "videos": [item.get("video_url", {}).get("url") for item in non_text_content if item.get("type") == "video_url"],
            "audios": [item.get("audio_url", {}).get("url") for item in non_text_content if item.get("type") == "audio_url"],
        },
        "warnings": warnings,
        "raw": data,
    }


def _query_seedance_a_task(payload: dict[str, Any]) -> dict[str, Any]:
    task_id = str(payload.get("task_id") or payload.get("taskId") or "").strip()
    if not task_id:
        raise ValueError("缺少 task_id")
    with SEEDANCE_A_TASKS_LOCK:
        task_meta = dict(SEEDANCE_A_TASKS.get(task_id) or {})
    api_key = str(task_meta.get("api_key") or _seedance_a_api_key()).strip()
    if not api_key:
        raise ValueError("缺少天悦A/Seedance API Key")
    base_url = str(task_meta.get("base_url") or _seedance_a_base_url()).rstrip("/")

    response = requests.get(
        f"{base_url}/v1/video/generations/{task_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    text = response.text
    try:
        data = response.json()
    except ValueError:
        data = {"_raw": text}
    if not response.ok:
        message = (
            (data.get("error") or {}).get("message")
            if isinstance(data.get("error"), dict)
            else data.get("error")
        ) or data.get("message") or f"天悦A HTTP {response.status_code}"
        raise RuntimeError(_seedance_a_friendly_error(str(message)[:800], response.status_code))

    status, progress, fail_reason = _parse_seedance_a_status(data)
    result_url = _extract_seedance_a_result_url(data, base_url)
    output_path = str(task_meta.get("output_path") or "")
    if status == "SUCCESS" and result_url and not (output_path and Path(output_path).exists()):
        output_path = _download_seedance_a_result(result_url, task_id, task_meta, api_key)
        with SEEDANCE_A_TASKS_LOCK:
            SEEDANCE_A_TASKS.setdefault(task_id, {}).update({"output_path": output_path, "finished_at": time.time()})

    return {
        "success": status != "FAILED",
        "task_id": task_id,
        "taskId": task_id,
        "status": status,
        "progress": progress,
        "video_url": output_path or result_url,
        "videoUrl": output_path or result_url,
        "output_path": output_path,
        "fail_reason": fail_reason,
        "raw": data,
    }


def _parse_seedance_a_status(data: dict[str, Any]) -> tuple[str, str, str]:
    outer = data.get("data") if isinstance(data.get("data"), dict) else data
    inner = outer.get("data") if isinstance(outer.get("data"), dict) else {}
    candidates = [
        inner.get("status"),
        outer.get("status"),
        data.get("status"),
        inner.get("task_status"),
        outer.get("task_status"),
        data.get("task_status"),
        inner.get("state"),
        outer.get("state"),
        data.get("state"),
    ]
    raw_status = ""
    for item in candidates:
        if isinstance(item, str) and item.strip():
            raw_status = item.strip().lower()
            break
    progress = str(inner.get("progress") or outer.get("progress") or data.get("progress") or "")
    fail_reason = str(
        inner.get("fail_reason")
        or outer.get("fail_reason")
        or inner.get("error")
        or outer.get("error")
        or data.get("error")
        or ""
    )
    if raw_status in {"success", "succeeded", "completed", "done", "finished"}:
        return "SUCCESS", progress or "100%", ""
    if raw_status in {"failed", "failure", "error", "cancelled", "canceled"}:
        return "FAILED", progress or "100%", fail_reason or "任务失败"
    if raw_status in {"running", "processing", "in_progress", "submitted"}:
        return "RUNNING", progress, ""
    if raw_status in {"pending", "queued", "created", "waiting", ""}:
        if _extract_seedance_a_result_url(data, _seedance_a_base_url()):
            return "SUCCESS", progress or "100%", ""
        return "PENDING", progress, ""
    if _extract_seedance_a_result_url(data, _seedance_a_base_url()):
        return "SUCCESS", progress or "100%", ""
    return "RUNNING", progress, ""


def _extract_seedance_a_result_url(data: dict[str, Any], base_url: str) -> str:
    outer = data.get("data") if isinstance(data.get("data"), dict) else data
    inner = outer.get("data") if isinstance(outer.get("data"), dict) else {}
    metadata = outer.get("metadata") if isinstance(outer.get("metadata"), dict) else {}
    candidates = [
        metadata.get("url"),
        metadata.get("video_url"),
        metadata.get("videoUrl"),
        metadata.get("result_url"),
        metadata.get("resultUrl"),
        inner.get("content_url"),
        inner.get("media_url"),
        inner.get("video_url"),
        inner.get("url"),
        outer.get("result_url"),
        outer.get("video_url"),
        outer.get("content_url"),
        data.get("result_url"),
        data.get("video_url"),
        data.get("url"),
    ]
    for value in candidates:
        text = str(value or "").strip()
        if not text:
            continue
        if text.startswith("/"):
            return f"{base_url.rstrip('/')}{text}"
        if text.startswith(("http://", "https://")):
            return text
    return ""


def _download_seedance_a_result(
    result_url: str,
    task_id: str,
    task_meta: dict[str, Any],
    api_key: str,
) -> str:
    output_dir = _seedance_a_output_dir(
        str(task_meta.get("project_dir") or ""),
        str(task_meta.get("source_video_path") or ""),
    )
    task_name = _safe_output_name(str(task_meta.get("task_name") or "shot_retake"))
    segment_id = _safe_output_name(str(task_meta.get("segment_id") or "")) if task_meta.get("segment_id") else ""
    stem = "_".join(part for part in ["seedance_a", task_name, segment_id, task_id[:8]] if part)
    output_path = output_dir / f"{stem}.mp4"
    if output_path.exists() and output_path.stat().st_size > 0:
        return str(output_path)

    def fetch(headers: dict[str, str] | None = None) -> requests.Response:
        return requests.get(result_url, headers=headers or {}, stream=True, timeout=300)

    response = fetch()
    if response.status_code in {401, 403}:
        response.close()
        response = fetch({"Authorization": f"Bearer {api_key}"})
    response.raise_for_status()
    try:
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    finally:
        response.close()
    return str(output_path)


def _single_role_transfer_backend() -> str:
    backend = os.environ.get(
        "SINGLE_ROLE_TRANSFER_BACKEND",
        DEFAULT_SINGLE_ROLE_TRANSFER_BACKEND,
    ).strip().lower()
    return backend if backend in TRANSFER_BACKENDS else DEFAULT_SINGLE_ROLE_TRANSFER_BACKEND


def _multi_role_transfer_backend() -> str:
    backend = os.environ.get(
        "MULTI_ROLE_TRANSFER_BACKEND",
        DEFAULT_MULTI_ROLE_TRANSFER_BACKEND,
    ).strip().lower()
    return backend if backend in TRANSFER_BACKENDS else DEFAULT_MULTI_ROLE_TRANSFER_BACKEND


def _normalize_transfer_backend(value: Any) -> str | None:
    backend = str(value or "").strip().lower()
    backend = backend.replace("-", "_")
    if "colored" in backend or "彩色" in backend:
        return "scail2_colored"
    return backend if backend in TRANSFER_BACKENDS else None


def _transfer_backend_for_role_count(role_count: int, requested_backend: Any = None) -> str:
    backend = _normalize_transfer_backend(requested_backend)
    if backend:
        return backend
    return (
        _multi_role_transfer_backend()
        if int(role_count) > 1
        else _single_role_transfer_backend()
    )


def _use_wan22_transfer(role_count: int, requested_backend: Any = None) -> bool:
    return _transfer_backend_for_role_count(role_count, requested_backend) == "wan22"


def _use_scail2_masked_transfer(requested_backend: Any = None) -> bool:
    return _normalize_transfer_backend(requested_backend) == "scail2_masked"


def _use_scail2_colored_transfer(requested_backend: Any = None) -> bool:
    return _normalize_transfer_backend(requested_backend) == "scail2_colored"


def _use_bernini_transfer(requested_backend: Any = None) -> bool:
    return _normalize_transfer_backend(requested_backend) == "bernini"


def _use_runninghub_bernini_transfer(requested_backend: Any = None) -> bool:
    return _normalize_transfer_backend(requested_backend) == "runninghub_bernini"


def _wan_multi_role_limit() -> int:
    raw = os.environ.get("WAN22_MULTI_ROLE_LIMIT", "").strip()
    if not raw:
        return DEFAULT_WAN22_MULTI_ROLE_LIMIT
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_WAN22_MULTI_ROLE_LIMIT


def _env_int(name: str, default: int, *, min_value: int = 0, max_value: int = 100) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(min_value, min(max_value, int(raw)))
    except ValueError:
        return default


def _env_float(name: str, default: float, *, min_value: float = 0.0, max_value: float = 1.0) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(min_value, min(max_value, float(raw)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _wan_focus_settings() -> dict[str, int | float]:
    return {
        "feather_pixels": _env_int(
            "WAN22_FOCUS_FEATHER_PIXELS",
            DEFAULT_WAN22_FOCUS_FEATHER_PIXELS,
            min_value=1,
            max_value=51,
        ),
        "erode_pixels": _env_int(
            "WAN22_FOCUS_ERODE_PIXELS",
            DEFAULT_WAN22_FOCUS_ERODE_PIXELS,
            min_value=0,
            max_value=20,
        ),
        "dilate_pixels": _env_int(
            "WAN22_FOCUS_DILATE_PIXELS",
            DEFAULT_WAN22_FOCUS_DILATE_PIXELS,
            min_value=0,
            max_value=30,
        ),
        "background_dim": _env_float(
            "WAN22_FOCUS_BACKGROUND_DIM",
            DEFAULT_WAN22_FOCUS_BACKGROUND_DIM,
            min_value=0.0,
            max_value=1.0,
        ),
        "background_blur_pixels": _env_int(
            "WAN22_FOCUS_BACKGROUND_BLUR_PIXELS",
            DEFAULT_WAN22_FOCUS_BACKGROUND_BLUR_PIXELS,
            min_value=0,
            max_value=51,
        ),
    }


def _wan22_allow_local_mask_fallback() -> bool:
    return _env_bool("WAN22_ALLOW_LOCAL_MASK_FALLBACK", DEFAULT_WAN22_ALLOW_LOCAL_MASK_FALLBACK)


def _wan22_allow_experimental_multi_focus() -> bool:
    return _env_bool(
        "WAN22_ALLOW_EXPERIMENTAL_MULTI_FOCUS",
        DEFAULT_WAN22_ALLOW_EXPERIMENTAL_MULTI_FOCUS,
    )


def _build_storyboard_assets(
    video_path: str,
    reference_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    max_people = 0
    for item in reference_segments:
        try:
            max_people = max(max_people, int(item.get("person_count") or 0))
        except (TypeError, ValueError):
            pass
    role_count = max(1, min(max_people or 2, 6))
    first_time = 0.1
    if reference_segments:
        try:
            first_time = max(0.1, float(reference_segments[0].get("start") or 0.0) + 0.2)
        except (TypeError, ValueError):
            first_time = 0.1
    segment_start_by_id: dict[str, float] = {}
    for item in reference_segments:
        segment_id = str(item.get("segment_id") or "").strip()
        if not segment_id:
            continue
        try:
            segment_start_by_id[segment_id] = float(item.get("start") or 0.0)
        except (TypeError, ValueError):
            pass

    def first_source_time(segment_ids: list[str], fallback: float) -> float:
        values = [
            segment_start_by_id[segment_id]
            for segment_id in segment_ids
            if segment_id in segment_start_by_id
        ]
        return max(0.1, min(values) + 0.2) if values else fallback

    assets: list[dict[str, Any]] = []
    for index in range(role_count):
        assets.append({
            "id": f"role_{index + 1}",
            "kind": "role",
            "name": f"角色{index + 1}",
            "tag": "待命名",
            "source_video_path": video_path,
            "source_time": round(first_time + index * 0.4, 3),
            "target_image": "",
            "prompt": (
                "写实短剧角色设定图，正面清晰，年龄、脸型、发型、身材比例、"
                "服装轮廓稳定，白底或干净背景，便于后续视频生成保持一致。"
            ),
            "status": "pending",
        })

    assets.append({
        "id": "scene_1",
        "kind": "scene",
        "name": "主场景",
        "tag": "待命名",
        "source_video_path": video_path,
        "source_time": first_time,
        "target_image": "",
        "prompt": "写实短剧场景参考图，保持原视频空间关系、光影气氛和镜头可用性。",
        "status": "pending",
    })

    joined_text = " ".join(str(item.get("description") or "") for item in reference_segments)
    prop_keywords = [
        ("粗铁链", ("铁链", "锁链", "链条")),
        ("车辆", ("车", "车内", "道路")),
        ("床", ("床", "卧室")),
    ]
    for prop_name, keywords in prop_keywords:
        if any(keyword in joined_text for keyword in keywords):
            assets.append({
                "id": f"prop_{len([item for item in assets if item['kind'] == 'prop']) + 1}",
                "kind": "prop",
                "name": prop_name,
                "tag": "物品",
                "source_video_path": video_path,
                "source_time": first_time,
                "source_quality_status": "prop_needs_visual_check",
                "source_usage_role": "object_evidence_candidate",
                "source_selection_method": "semantic_keyword_candidate",
                "track_status": "needs_manual_review",
                "target_image": "",
                "prompt": f"写实{prop_name}道具参考图，材质清晰，适合短剧镜头内复用。",
                "status": "pending",
            })

    return assets


def _build_storyboard_assets_v2(
    video_path: str,
    reference_segments: list[dict[str, Any]],
    auto_director_plan: dict[str, Any] | None = None,
    understanding: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    max_people = 0
    for item in reference_segments:
        try:
            max_people = max(max_people, int(item.get("person_count") or 0))
        except (TypeError, ValueError):
            pass
    role_count = max(1, min(max_people or 2, 6))
    first_time = 0.1
    if reference_segments:
        try:
            first_time = max(0.1, float(reference_segments[0].get("start") or 0.0) + 0.2)
        except (TypeError, ValueError):
            first_time = 0.1
    semantic_asset_segments = _mode2_semantic_scene_segments_from_sources(
        reference_segments,
        auto_director_plan=auto_director_plan,
        understanding=understanding,
    )
    asset_reference_segments = [item for item in reference_segments if isinstance(item, dict)]
    seen_asset_segment_ids = {
        str(item.get("segment_id") or "").strip()
        for item in asset_reference_segments
        if str(item.get("segment_id") or "").strip()
    }
    for item in semantic_asset_segments:
        segment_id = str(item.get("segment_id") or "").strip()
        if segment_id and segment_id in seen_asset_segment_ids:
            continue
        asset_reference_segments.append(item)
        if segment_id:
            seen_asset_segment_ids.add(segment_id)
    segment_start_by_id: dict[str, float] = {}
    for item in asset_reference_segments:
        segment_id = str(item.get("segment_id") or "").strip()
        if not segment_id:
            continue
        try:
            segment_start_by_id[segment_id] = float(item.get("start") or 0.0)
        except (TypeError, ValueError):
            pass

    def first_source_time(segment_ids: list[str], fallback: float) -> float:
        values = [
            segment_start_by_id[segment_id]
            for segment_id in segment_ids
            if segment_id in segment_start_by_id
        ]
        return max(0.1, min(values) + 0.2) if values else fallback

    story = (auto_director_plan or {}).get("story") if isinstance(auto_director_plan, dict) else {}
    story_characters = [
        item for item in ((story or {}).get("characters") or [])
        if isinstance(item, dict)
    ]
    def character_first_segment(item: dict[str, Any]) -> int:
        values = [
            str(value or "").strip()
            for value in (item.get("segment_ids") or [])
            if str(value or "").strip()
        ]
        try:
            return min(int(value) for value in values if value.isdigit())
        except ValueError:
            return 999999

    story_characters = sorted(
        story_characters,
        key=lambda item: (character_first_segment(item), -len(item.get("segment_ids") or [])),
    )[:8]

    role_specs: list[dict[str, Any]] = []
    for index, character in enumerate(story_characters, 1):
        candidate_names = [
            str(character.get("role_name") or "").strip(),
            str(character.get("visual_label") or "").strip(),
            *[str(value or "").strip() for value in (character.get("role_candidates") or [])],
        ]
        name = next((value for value in candidate_names if value), f"角色{index}")
        role_specs.append({
            "name": name,
            "tag": str(character.get("visual_label") or "").strip() or "待命名",
            "description": str(character.get("description") or "").strip(),
            "source_segment_ids": [
                str(value or "").strip()
                for value in (character.get("segment_ids") or [])
                if str(value or "").strip()
            ],
        })
    if not role_specs:
        role_specs.append({
            "name": "待识别角色",
            "tag": "占位",
            "description": "全片理解未完成，暂未识别真实角色。请开启/重跑全片理解后再绑定替换图。",
            "source_segment_ids": [],
            "status": "needs_identification",
            "placeholder": True,
        })

    assets: list[dict[str, Any]] = []
    for index, role in enumerate(role_specs):
        assets.append({
            "id": f"role_{index + 1}",
            "kind": "role",
            "name": role["name"],
            "tag": role["tag"],
            "source_video_path": video_path,
            "source_time": round(first_source_time(role["source_segment_ids"], first_time + index * 0.4), 3),
            "source_segment_ids": role["source_segment_ids"],
            "target_image": "",
            "prompt": (
                "写实短剧角色设定图，正面清晰，年龄、脸型、发型、身材比例、服装轮廓稳定，"
                "白底或干净背景，便于后续视频生成保持一致。"
                + (f" 原片识别描述：{role['description']}" if role["description"] else "")
            ),
            "status": str(role.get("status") or "pending"),
            "placeholder": bool(role.get("placeholder")),
            "selection_reason": str(role.get("description") or ""),
        })

    scene_groups: dict[str, dict[str, Any]] = {}
    for item in asset_reference_segments:
        description = str(item.get("description") or "").strip()
        if not description:
            continue
        key = description[:80]
        group = scene_groups.setdefault(key, {
            "description": description,
            "segments": [],
            "start": float(item.get("start") or 0.0),
        })
        segment_id = str(item.get("segment_id") or "").strip()
        if segment_id and segment_id not in group["segments"]:
            group["segments"].append(segment_id)
        try:
            group["start"] = min(float(group["start"]), float(item.get("start") or group["start"]))
        except (TypeError, ValueError):
            pass

    for index, group in enumerate(sorted(scene_groups.values(), key=lambda value: value.get("start", 0.0))[:8], 1):
        description = str(group.get("description") or "").strip()
        assets.append({
            "id": f"scene_{index}",
            "kind": "scene",
            "name": f"场景{index}",
            "tag": "语义场景",
            "source_video_path": video_path,
            "source_time": max(0.1, float(group.get("start") or 0.0) + 0.2),
            "source_segment_ids": [value for value in group.get("segments", []) if value],
            "target_image": "",
            "prompt": (
                "写实短剧场景参考图，只描述环境、空间关系、光影气氛和镜头可用性；"
                "不要把人物脸部特写或剧情动作当成场景资产。"
                + (f" 原片场景：{description}" if description else "")
            ),
            "status": "pending",
        })

    joined_text = " ".join(str(item.get("description") or "") for item in asset_reference_segments)
    for prop_name, keywords in MODE2_STORYBOARD_PROP_KEYWORDS:
        matched_segment_ids: list[str] = []
        for segment in asset_reference_segments:
            segment_text = " ".join([
                str(segment.get("description") or ""),
                str(segment.get("key_action") or ""),
            ])
            if not any(keyword in segment_text for keyword in keywords):
                continue
            segment_id = str(segment.get("segment_id") or "").strip()
            if segment_id and segment_id not in matched_segment_ids:
                matched_segment_ids.append(segment_id)
        if matched_segment_ids or any(keyword in joined_text for keyword in keywords):
            assets.append({
                "id": f"prop_{len([item for item in assets if item['kind'] == 'prop']) + 1}",
                "kind": "prop",
                "name": prop_name,
                "tag": "物品",
                "source_video_path": video_path,
                "source_time": first_source_time(matched_segment_ids, first_time),
                "source_segment_ids": matched_segment_ids,
                "source_quality_status": "prop_needs_visual_check",
                "source_usage_role": "object_evidence_candidate",
                "source_selection_method": "semantic_keyword_candidate",
                "track_status": "needs_manual_review",
                "target_image": "",
                "prompt": f"写实{prop_name}道具参考图，材质清晰，适合短剧镜头内复用。",
                "status": "pending",
                "selection_reason": (
                    f"从包含“{'/'.join(keywords[:3])}”的语义段抽取候选帧；仍需人工确认道具本体清晰可见。"
                    if matched_segment_ids
                    else "只从文本中识别到道具关键词，未定位到具体语义段，需人工补图。"
                ),
            })

    return assets


def _storyboard_asset_cache_dir(video_path: str) -> Path:
    video_key = hashlib.md5(str(Path(video_path)).encode("utf-8", errors="ignore")).hexdigest()[:12]
    return ROOT.parent / ".storyboard_asset_frames" / video_key


def _storyboard_asset_frame_path(video_path: str, asset: dict[str, Any], time_seconds: float | None = None) -> Path:
    cache_dir = _storyboard_asset_cache_dir(video_path)
    asset_id = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(asset.get("id") or "asset"))
    try:
        source_time = float(asset.get("source_time") or 0.1) if time_seconds is None else float(time_seconds)
        ms = int(round(source_time * 1000))
    except (TypeError, ValueError):
        ms = 100
    return cache_dir / f"{asset_id}_{ms:08d}ms.jpg"


def _storyboard_asset_sheet_path(video_path: str, asset: dict[str, Any], sample_times: list[float]) -> Path:
    cache_dir = _storyboard_asset_cache_dir(video_path)
    asset_id = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(asset.get("id") or "asset"))
    key = "_".join(f"{int(round(time_value * 1000)):08d}" for time_value in sample_times) or "single"
    return cache_dir / f"{asset_id}_sheet_{key}.jpg"


def _storyboard_reference_maps(
    reference_segments: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, float], dict[str, float]]:
    segment_by_id: dict[str, dict[str, Any]] = {}
    start_by_id: dict[str, float] = {}
    end_by_id: dict[str, float] = {}
    for item in reference_segments:
        segment_id = str(item.get("segment_id") or "").strip()
        if not segment_id:
            continue
        segment_by_id[segment_id] = item
        try:
            start_by_id[segment_id] = float(item.get("start") or 0.0)
        except (TypeError, ValueError):
            start_by_id[segment_id] = 0.0
        try:
            end_by_id[segment_id] = float(item.get("end") or item.get("start") or 0.0)
        except (TypeError, ValueError):
            end_by_id[segment_id] = start_by_id[segment_id]
    return segment_by_id, start_by_id, end_by_id


def _storyboard_asset_time_span(
    asset: dict[str, Any],
    *,
    start_by_id: dict[str, float],
    end_by_id: dict[str, float],
    shots: list[dict[str, Any]],
) -> dict[str, Any]:
    asset_segment_ids = [
        str(value or "").strip()
        for value in (asset.get("source_segment_ids") or [])
        if str(value or "").strip()
    ]
    shot_ids = [
        str(value or "").strip()
        for value in (asset.get("used_shots") or [])
        if str(value or "").strip()
    ]

    starts: list[float] = []
    ends: list[float] = []
    windows: list[dict[str, Any]] = []
    if asset_segment_ids:
        for segment_id in asset_segment_ids:
            if segment_id in start_by_id:
                start_value = start_by_id[segment_id]
                starts.append(start_value)
            else:
                start_value = None
            if segment_id in end_by_id:
                end_value = end_by_id[segment_id]
                ends.append(end_value)
            else:
                end_value = None
            if start_value is not None and end_value is not None and end_value > start_value:
                windows.append({
                    "id": segment_id,
                    "start": round(start_value, 3),
                    "end": round(end_value, 3),
                })

    if not starts and shot_ids:
        shot_by_id = {
            str(shot.get("segment_id") or "").strip(): shot
            for shot in shots
            if str(shot.get("segment_id") or "").strip()
        }
        for shot_id in shot_ids:
            shot = shot_by_id.get(shot_id)
            if not shot:
                continue
            try:
                starts.append(float(shot.get("start") or 0.0))
                shot_end = float(shot.get("end") or shot.get("start") or 0.0)
                ends.append(shot_end)
                shot_start = float(shot.get("start") or 0.0)
                if shot_end > shot_start:
                    windows.append({
                        "id": shot_id,
                        "start": round(shot_start, 3),
                        "end": round(shot_end, 3),
                    })
            except (TypeError, ValueError):
                continue

    source_time = max(0.05, float(asset.get("source_time") or 0.1))
    if not starts or not ends:
        start = max(0.05, source_time - 0.9)
        end = max(start + 0.1, source_time + 0.9)
    else:
        start = max(0.05, min(starts))
        end = max(start + 0.1, max(ends))

    duration = max(0.1, end - start)
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(duration, 3),
        "mid": round(start + duration * 0.5, 3),
        "label": f"{start:.2f}s-{end:.2f}s",
        "source_segment_ids": asset_segment_ids,
        "shot_ids": shot_ids,
        "windows": windows[:12],
    }


def _storyboard_candidate_frames(
    video_path: str,
    asset: dict[str, Any],
    *,
    time_span: dict[str, Any],
    reference_frames: list[dict[str, Any]],
    shots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    start = float(time_span.get("start") or 0.0)
    end = float(time_span.get("end") or start)
    mid = float(time_span.get("mid") or (start + end) / 2.0)
    duration = max(0.1, end - start)
    asset_id = str(asset.get("id") or "asset")
    kind = str(asset.get("kind") or "")
    raw_windows = [
        item for item in (time_span.get("windows") or [])
        if isinstance(item, dict)
    ]
    cache_dir = _storyboard_asset_cache_dir(video_path) / "evidence" / asset_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    candidates: dict[int, dict[str, Any]] = {}

    def in_allowed_window(time_seconds: float) -> bool:
        edge_padding = 0.08 if kind in {"scene", "prop"} else 0.02
        windows = raw_windows or [{"start": start, "end": end}]
        for window in windows:
            try:
                window_start = float(window.get("start") or 0.0)
                window_end = float(window.get("end") or window_start)
            except (TypeError, ValueError):
                continue
            if window_end <= window_start:
                continue
            padding = edge_padding if window_end - window_start > edge_padding * 3 else 0.0
            if window_start + padding <= time_seconds <= window_end - padding:
                return True
        return False

    def add_record(raw: dict[str, Any], *, source: str) -> None:
        try:
            time_seconds = float(raw.get("time") or 0.0)
        except (TypeError, ValueError):
            return
        if not in_allowed_window(time_seconds):
            return
        key = int(round(time_seconds * 20))
        record = {
            "time": round(time_seconds, 3),
            "path": str(raw.get("path") or "").strip(),
            "diff_prev": float(raw.get("diff_prev") or 0.0),
            "blue_ratio": float(raw.get("blue_ratio") or 0.0),
            "skin_ratio": float(raw.get("skin_ratio") or 0.0),
            "center_skin_ratio": float(raw.get("center_skin_ratio") or 0.0),
            "white_ratio": float(raw.get("white_ratio") or 0.0),
            "dark_ratio": float(raw.get("dark_ratio") or 0.0),
            "edge_score": float(raw.get("edge_score") or 0.0),
            "source": source,
        }
        candidates.setdefault(key, record)

    for frame in reference_frames:
        if not isinstance(frame, dict):
            continue
        add_record(frame, source="manifest_frame")

    if len(candidates) < 3:
        sample_times = (
            _storyboard_window_sample_times(raw_windows, start=start, end=end, limit=5)
            if kind in {"scene", "prop"}
            else _storyboard_asset_sample_times(asset, shots, limit=5)
        )
        if not sample_times:
            sample_times = [
                round(max(0.05, start + duration * ratio), 3)
                for ratio in (0.15, 0.5, 0.85)
            ]
        for time_seconds in sample_times:
            frame_path = _storyboard_asset_frame_path(video_path, asset, time_seconds)
            if not frame_path.exists() or frame_path.stat().st_size <= 0:
                try:
                    extract_frame(video_path, time_seconds, frame_path)
                except Exception:  # noqa: BLE001
                    continue
            if not frame_path.exists() or frame_path.stat().st_size <= 0:
                continue
            try:
                frame_record, _pixels = analyze_frame(frame_path, time_seconds)
            except Exception:  # noqa: BLE001
                continue
            add_record(frame_record.to_dict(), source="sampled_frame")

    if not candidates:
        fallback_time = max(0.05, mid)
        frame_path = _storyboard_asset_frame_path(video_path, asset, fallback_time)
        if not frame_path.exists() or frame_path.stat().st_size <= 0:
            try:
                extract_frame(video_path, fallback_time, frame_path)
            except Exception:  # noqa: BLE001
                return []
        if frame_path.exists() and frame_path.stat().st_size > 0:
            try:
                frame_record, _pixels = analyze_frame(frame_path, fallback_time)
                add_record(frame_record.to_dict(), source="fallback_frame")
            except Exception:  # noqa: BLE001
                pass

    records = list(candidates.values())
    if not records:
        return []

    for record in records:
        time_seconds = float(record.get("time") or mid)
        center_bias = max(0.0, 1.0 - min(1.0, abs(time_seconds - mid) / max(duration * 0.5, 0.3)))
        diff_prev = float(record.get("diff_prev") or 0.0)
        edge = float(record.get("edge_score") or 0.0)
        skin = float(record.get("skin_ratio") or 0.0)
        center_skin = float(record.get("center_skin_ratio") or 0.0)
        white = float(record.get("white_ratio") or 0.0)
        dark = float(record.get("dark_ratio") or 0.0)
        blue = float(record.get("blue_ratio") or 0.0)

        if kind == "role":
            raw_score = (
                center_skin * 2.2
                + skin * 0.9
                + edge * 1.0
                + center_bias * 0.4
                - white * 0.7
                - dark * 0.5
                - blue * 0.1
                - diff_prev * 0.25
            )
            reason_bits = []
            if center_skin >= 0.12:
                reason_bits.append("脸部更居中")
            if skin >= 0.10:
                reason_bits.append("人物占位足")
            if edge >= 0.08:
                reason_bits.append("画面更清晰")
            if white <= 0.22:
                reason_bits.append("曝光更稳")
            if diff_prev <= 0.10:
                reason_bits.append("动作更稳定")
        elif kind == "scene":
            raw_score = (
                edge * 1.6
                + white * 0.5
                + center_bias * 0.25
                - skin * 0.9
                - center_skin * 1.4
                - dark * 0.4
                + min(diff_prev, 0.2) * 0.4
            )
            reason_bits = []
            if edge >= 0.08:
                reason_bits.append("空间结构更清楚")
            if white >= 0.16:
                reason_bits.append("场景边界更明显")
            if skin <= 0.12:
                reason_bits.append("人物干扰更少")
        else:
            raw_score = (
                edge * 1.2
                + white * 0.6
                + center_bias * 0.3
                - skin * 0.3
                - center_skin * 0.3
                - dark * 0.2
            )
            reason_bits = []
            if edge >= 0.08:
                reason_bits.append("纹理更清楚")
            if white >= 0.16:
                reason_bits.append("画面更稳定")
        record["score"] = round(raw_score, 4)
        record["reason"] = "、".join(reason_bits) if reason_bits else "通用关键帧"
        record["metrics"] = {
            "center_bias": round(center_bias, 3),
            "diff_prev": round(diff_prev, 3),
            "edge_score": round(edge, 3),
            "skin_ratio": round(skin, 3),
            "center_skin_ratio": round(center_skin, 3),
            "white_ratio": round(white, 3),
            "dark_ratio": round(dark, 3),
            "blue_ratio": round(blue, 3),
        }

    records.sort(key=lambda item: (-float(item.get("score") or 0.0), float(item.get("time") or 0.0)))
    ranked = records[:8]
    ranked.sort(key=lambda item: float(item.get("time") or 0.0))
    if len(ranked) > 4:
        ranked = _evenly_pick(ranked, 4)
        ranked.sort(key=lambda item: float(item.get("time") or 0.0))
    return ranked


def _storyboard_role_face_crops(
    video_path: str,
    asset: dict[str, Any],
    keyframes: list[dict[str, Any]],
) -> list[Path]:
    try:
        import cv2
    except Exception:  # noqa: BLE001
        return []

    face_xml = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(str(face_xml))
    if face_cascade.empty():
        return []

    crop_paths: list[Path] = []
    cache_dir = _storyboard_asset_cache_dir(video_path) / "face_crops" / str(asset.get("id") or "asset")
    cache_dir.mkdir(parents=True, exist_ok=True)
    for item in keyframes:
        frame_path = Path(str(item.get("path") or ""))
        if not frame_path.exists() or frame_path.stat().st_size <= 0:
            continue
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(36, 36),
        )
        if len(rects) > 0:
            best = max(rects, key=lambda rect: int(rect[2]) * int(rect[3]))
            crop = _storyboard_crop_face_portrait(image, tuple(map(int, best)))
            if crop.size > 0:
                crop_path = cache_dir / f"face_{int(round(float(item.get('time') or 0.0) * 1000)):08d}.jpg"
                cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if crop_path.exists() and crop_path.stat().st_size > 0:
                    crop_paths.append(crop_path)
                    item["crop_path"] = str(crop_path)
                    item["face_box"] = [int(best[0]), int(best[1]), int(best[2]), int(best[3])]
                    item["face_found"] = True
                    continue
        item["face_found"] = False
    return crop_paths


def _storyboard_time_span_text(time_span: dict[str, Any] | None) -> str:
    if not time_span:
        return ""
    start = float(time_span.get("start") or 0.0)
    end = float(time_span.get("end") or start)
    if end <= start:
        return ""
    return f"{start:.2f}s - {end:.2f}s"


def _evenly_pick(items: list[Any], limit: int) -> list[Any]:
    if len(items) <= limit:
        return items
    if limit <= 1:
        return [items[0]]
    picked: list[Any] = []
    for index in range(limit):
        source_index = round(index * (len(items) - 1) / (limit - 1))
        picked.append(items[source_index])
    return picked


def _storyboard_window_sample_times(
    windows: list[dict[str, Any]],
    *,
    start: float,
    end: float,
    limit: int,
) -> list[float]:
    valid_windows: list[tuple[float, float]] = []
    for item in windows:
        if not isinstance(item, dict):
            continue
        try:
            window_start = max(0.05, float(item.get("start") or 0.0))
            window_end = float(item.get("end") or window_start)
        except (TypeError, ValueError):
            continue
        if window_end > window_start:
            valid_windows.append((window_start, window_end))
    if not valid_windows and end > start:
        valid_windows.append((max(0.05, start), end))
    if not valid_windows:
        return []

    limit = max(1, int(limit or 1))
    if len(valid_windows) >= limit:
        selected = _evenly_pick(valid_windows, limit)
        return [
            round(window_start + (window_end - window_start) * 0.5, 3)
            for window_start, window_end in selected
        ]

    times: list[float] = []
    remaining = limit
    for index, (window_start, window_end) in enumerate(valid_windows):
        window_duration = max(0.1, window_end - window_start)
        windows_left = len(valid_windows) - index
        take = max(1, remaining // windows_left)
        if take == 1 or window_duration < 1.2:
            ratios = (0.5,)
        elif take == 2:
            ratios = (0.32, 0.68)
        elif take == 3:
            ratios = (0.22, 0.5, 0.78)
        else:
            ratios = tuple((i + 1) / (take + 1) for i in range(take))
        for ratio in ratios:
            padding = min(0.12, window_duration * 0.15)
            inner_start = window_start + padding
            inner_end = max(inner_start, window_end - padding)
            times.append(round(inner_start + (inner_end - inner_start) * ratio, 3))
            remaining -= 1
            if remaining <= 0:
                break
        if remaining <= 0:
            break

    deduped: list[float] = []
    seen: set[int] = set()
    for value in times:
        key = int(round(value * 20))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped[:limit]


def _storyboard_asset_sample_times(
    asset: dict[str, Any],
    shots: list[dict[str, Any]],
    *,
    limit: int = 4,
) -> list[float]:
    used = {
        str(shot_id)
        for shot_id in (asset.get("used_shots") or [])
        if str(shot_id).strip()
    }
    times: list[float] = []
    for shot in shots:
        shot_id = str(shot.get("segment_id") or "")
        if used and shot_id not in used:
            continue
        try:
            start = float(shot.get("start") or 0.0)
            end = float(shot.get("end") or start)
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        times.append(max(0.05, start + (end - start) * 0.5))

    if not times:
        try:
            times = [max(0.05, float(asset.get("source_time") or 0.1))]
        except (TypeError, ValueError):
            times = [0.1]

    deduped: list[float] = []
    seen: set[int] = set()
    for value in times:
        key = int(round(value * 10))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return [round(value, 3) for value in _evenly_pick(deduped, limit)]


def _make_storyboard_asset_contact_sheet(frame_paths: list[Path], output_path: Path) -> bool:
    valid = [path for path in frame_paths if path.exists() and path.stat().st_size > 0]
    if not valid:
        return False
    if len(valid) == 1:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(valid[0].read_bytes())
        return True

    try:
        from PIL import Image, ImageDraw, ImageOps
    except Exception:  # noqa: BLE001
        return False

    cell_w = 360
    cell_h = 640
    columns = 2
    rows = math.ceil(len(valid) / columns)
    sheet = Image.new("RGB", (cell_w * columns, cell_h * rows), (5, 7, 8))
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(valid):
        try:
            image = Image.open(path).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        image = ImageOps.fit(image, (cell_w, cell_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        x = (index % columns) * cell_w
        y = (index // columns) * cell_h
        sheet.paste(image, (x, y))
        draw.rectangle((x + 8, y + 8, x + 72, y + 34), fill=(18, 26, 31))
        draw.text((x + 18, y + 15), f"F{index + 1}", fill=(230, 238, 242))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=88)
    return output_path.exists() and output_path.stat().st_size > 0


def _storyboard_rect_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _dedupe_storyboard_rects(rects: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    ordered = sorted(rects, key=lambda item: item[2] * item[3], reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for rect in ordered:
        if all(_storyboard_rect_iou(rect, existing) < 0.35 for existing in kept):
            kept.append(rect)
    return kept


def _storyboard_crop_face_portrait(image: Any, rect: tuple[int, int, int, int]) -> Any:
    height, width = image.shape[:2]
    x, y, w, h = rect
    cx = x + w / 2
    cy = y + h / 2
    crop_w = max(w * 3.0, h * 2.0)
    crop_h = max(h * 4.2, crop_w * 1.35)
    x1 = max(0, int(round(cx - crop_w / 2)))
    y1 = max(0, int(round(cy - crop_h * 0.36)))
    x2 = min(width, int(round(cx + crop_w / 2)))
    y2 = min(height, int(round(y1 + crop_h)))
    if x2 <= x1 or y2 <= y1:
        return image
    return image[y1:y2, x1:x2]


def _build_storyboard_role_face_sources(
    video_path: str,
    role_assets: list[dict[str, Any]],
    shots: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not role_assets:
        return {}
    try:
        import cv2
        import numpy as np
    except Exception:  # noqa: BLE001
        return {}

    sample_times = _storyboard_asset_sample_times({"used_shots": [shot.get("segment_id") for shot in shots]}, shots, limit=10)
    if not sample_times:
        return {}

    face_xml = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    profile_xml = Path(cv2.data.haarcascades) / "haarcascade_profileface.xml"
    face_cascade = cv2.CascadeClassifier(str(face_xml))
    profile_cascade = cv2.CascadeClassifier(str(profile_xml))
    if face_cascade.empty() and profile_cascade.empty():
        return {}

    cache_dir = _storyboard_asset_cache_dir(video_path) / "face_crops"
    frame_asset = {"id": "role_cluster_frame", "source_time": 0.1}
    detections: list[dict[str, Any]] = []
    for time_seconds in sample_times:
        frame_path = _storyboard_asset_frame_path(video_path, frame_asset, time_seconds)
        if not frame_path.exists() or frame_path.stat().st_size <= 0:
            try:
                from spvideo.ffmpeg_tools import extract_frame
                extract_frame(video_path, time_seconds, frame_path)
            except Exception:  # noqa: BLE001
                continue
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects: list[tuple[int, int, int, int]] = []
        if not face_cascade.empty():
            rects.extend([tuple(map(int, rect)) for rect in face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(36, 36),
            )])
        if False and not profile_cascade.empty():
            rects.extend([tuple(map(int, rect)) for rect in profile_cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(36, 36),
            )])
            flipped = cv2.flip(gray, 1)
            flipped_rects = profile_cascade.detectMultiScale(
                flipped,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(36, 36),
            )
            width = gray.shape[1]
            rects.extend([(int(width - x - w), int(y), int(w), int(h)) for x, y, w, h in flipped_rects])
        rects = _dedupe_storyboard_rects(rects)
        for rect_index, rect in enumerate(rects[:4]):
            crop = _storyboard_crop_face_portrait(image, rect)
            if crop.size == 0:
                continue
            crop_path = cache_dir / f"face_{int(round(time_seconds * 1000)):08d}_{rect_index}.jpg"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            resized = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            feature = hsv.reshape(-1, 3).mean(axis=0).astype("float32")
            x, y, w, h = rect
            detections.append({
                "path": crop_path,
                "frame_path": frame_path,
                "feature": feature,
                "time": time_seconds,
                "x": x + w / 2,
                "face_box": [int(x), int(y), int(w), int(h)],
                "area": w * h,
            })

    if len(detections) < len(role_assets):
        return {}

    k = min(len(role_assets), len(detections))
    features = np.array([item["feature"] for item in detections], dtype=np.float32)
    if k <= 1:
        labels = np.zeros((len(detections),), dtype=np.int32)
    else:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
        _compactness, raw_labels, _centers = cv2.kmeans(
            features,
            k,
            None,
            criteria,
            5,
            cv2.KMEANS_PP_CENTERS,
        )
        labels = raw_labels.reshape(-1)

    clusters: list[list[dict[str, Any]]] = [[] for _ in range(k)]
    for item, label in zip(detections, labels):
        clusters[int(label)].append(item)
    clusters = [cluster for cluster in clusters if cluster]
    clusters.sort(key=lambda cluster: min(item["time"] for item in cluster))

    outputs: dict[str, dict[str, Any]] = {}
    for asset, cluster in zip(role_assets, clusters):
        cluster.sort(key=lambda item: (-float(item["area"]), float(item["time"])))
        picked_items = _evenly_pick(cluster[:8], 4)
        crop_paths = [Path(item["path"]) for item in picked_items]
        sheet_path = _storyboard_asset_cache_dir(video_path) / f"{asset.get('id')}_faces_sheet.jpg"
        if _make_storyboard_asset_contact_sheet(crop_paths, sheet_path):
            outputs[str(asset.get("id") or "")] = {
                "sheet_path": sheet_path,
                "crop_paths": crop_paths,
                "items": picked_items,
            }
    return outputs


def _attach_storyboard_asset_source_images(
    video_path: str,
    assets: list[dict[str, Any]],
    shots: list[dict[str, Any]],
    reference_segments: list[dict[str, Any]] | None = None,
    reference_frames: list[dict[str, Any]] | None = None,
) -> None:
    if not video_path or not Path(video_path).exists():
        return

    reference_segments = [item for item in (reference_segments or []) if isinstance(item, dict)]
    reference_frames = [item for item in (reference_frames or []) if isinstance(item, dict)]
    _, start_by_id, end_by_id = _storyboard_reference_maps(reference_segments)
    role_assets = [
        asset for asset in assets
        if str(asset.get("kind") or "") == "role"
        and not bool(asset.get("placeholder"))
        and str(asset.get("status") or "") != "needs_identification"
    ]
    role_face_candidates = _build_storyboard_role_face_sources(video_path, role_assets, shots)

    for asset in assets:
        kind = str(asset.get("kind") or "")
        time_span = _storyboard_asset_time_span(
            asset,
            start_by_id=start_by_id,
            end_by_id=end_by_id,
            shots=shots,
        )
        asset["time_span"] = time_span
        asset["track_path"] = str(asset.get("track_path") or "")
        asset["mask_path"] = str(asset.get("mask_path") or "")
        asset["track_source"] = str(asset.get("track_source") or "")
        asset["track_status"] = str(asset.get("track_status") or "pending")
        if bool(asset.get("placeholder")) or str(asset.get("status") or "") == "needs_identification":
            asset["source_kind"] = "unidentified_placeholder"
            asset["track_source"] = "not_run"
            asset["track_status"] = "needs_identification"
            continue

        if kind == "role":
            candidate = role_face_candidates.get(str(asset.get("id") or ""))
            if candidate:
                crop_paths = [
                    Path(path) for path in (candidate.get("crop_paths") or [])
                    if isinstance(path, Path) and path.exists() and path.stat().st_size > 0
                ]
                items = [
                    item for item in (candidate.get("items") or [])
                    if isinstance(item, dict)
                ]
                sheet_path = candidate.get("sheet_path")
                if isinstance(sheet_path, Path) and sheet_path.exists() and sheet_path.stat().st_size > 0:
                    asset["candidate_source_image"] = str(sheet_path)
                    asset["candidate_source_images"] = [str(path) for path in crop_paths]
                    asset["candidate_source_kind"] = "role_face_cluster_candidate"
                    asset["candidate_source_note"] = (
                        "自动聚类的人脸候选，只能用来找身份点；"
                        "未经过 SAM3 身份分轨前不能当可信角色源图。"
                    )
                    asset["candidate_keyframes"] = [
                        {
                            "time": round(float(item.get("time") or 0.0), 3),
                            "path": str(item.get("frame_path") or ""),
                            "crop_path": str(item.get("path") or ""),
                            "face_box": item.get("face_box") or [],
                            "reason": "候选人脸簇；请在原始帧上点这个角色本人",
                        }
                        for item in items
                        if str(item.get("frame_path") or "").strip()
                    ]

        keyframes = _storyboard_candidate_frames(
            video_path,
            asset,
            time_span=time_span,
            reference_frames=reference_frames,
            shots=shots,
        )
        if not keyframes:
            continue

        source_paths: list[Path]
        source_kind = f"{kind or 'asset'}_keyframe_bundle"
        if kind == "role":
            crop_paths = _storyboard_role_face_crops(video_path, asset, keyframes)
            if crop_paths:
                source_paths = crop_paths
                source_kind = "role_face_bundle"
                asset["track_source"] = "face_detect"
                asset["track_status"] = "ready"
            else:
                source_paths = [Path(str(item.get("path") or "")) for item in keyframes if str(item.get("path") or "").strip()]
                asset["track_source"] = "face_detect"
                asset["track_status"] = "face_not_found"
        else:
            source_paths = [Path(str(item.get("path") or "")) for item in keyframes if str(item.get("path") or "").strip()]
            asset["track_source"] = "keyframe_score"
            asset["track_status"] = "ready"

        source_paths = [path for path in source_paths if path.exists() and path.stat().st_size > 0]
        if not source_paths:
            continue

        sample_times = [float(item.get("time") or time_span.get("mid") or 0.1) for item in keyframes]
        target = _storyboard_asset_sheet_path(video_path, asset, sample_times)
        if not target.exists() or target.stat().st_size <= 0:
            try:
                made = _make_storyboard_asset_contact_sheet(source_paths, target)
            except Exception:  # noqa: BLE001
                made = False
            if not made:
                continue

        asset["source_image"] = str(target)
        asset["source_images"] = [str(path) for path in source_paths]
        asset["source_frame_count"] = len(source_paths)
        asset["source_kind"] = source_kind
        if kind == "prop" and not _mode2_asset_is_manually_approved(asset):
            asset["source_quality_status"] = str(asset.get("source_quality_status") or "prop_needs_visual_check")
            asset["source_usage_role"] = "object_evidence_candidate"
            asset["visual_confirmed"] = False
            asset["source_visual_status"] = "unconfirmed_semantic_prop"
            asset["source_trust_level"] = "semantic_candidate_only"
            asset["source_selection_method"] = str(
                asset.get("source_selection_method") or "semantic_keyword_candidate"
            )
            if str(asset.get("track_status") or "") in {"", "pending", "ready"}:
                asset["track_status"] = "needs_manual_review"
        elif kind == "scene":
            asset["source_usage_role"] = str(asset.get("source_usage_role") or "environment_reference")
            asset["source_selection_method"] = str(
                asset.get("source_selection_method") or "timeline_keyframe_candidate"
            )
        asset["keyframes"] = [
            {
                "time": float(item.get("time") or 0.0),
                "path": str(item.get("path") or ""),
                "crop_path": str(item.get("crop_path") or ""),
                "face_box": item.get("face_box"),
                "face_found": bool(item.get("face_found")),
                "score": round(float(item.get("score") or 0.0), 4),
                "reason": str(item.get("reason") or ""),
                "metrics": item.get("metrics") or {},
                "source": str(item.get("source") or ""),
            }
            for item in keyframes
        ]
        top_scores = [float(item.get("score") or 0.0) for item in keyframes]
        divisor = 2.2 if kind == "role" else 1.8 if kind == "scene" else 1.6
        confidence = min(0.99, max(0.05, (sum(top_scores) / len(top_scores)) / divisor))
        asset["confidence"] = round(confidence, 3)
        primary_reason = str(keyframes[0].get("reason") or "")
        asset["selection_reason"] = (
            f"{_storyboard_time_span_text(time_span)} 关键帧包：{primary_reason}"
            if primary_reason
            else f"{_storyboard_time_span_text(time_span)} 关键帧包"
        )


def _flag_storyboard_role_source_collisions(assets: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for asset in assets:
        if str(asset.get("kind") or "") != "role":
            continue
        if bool(asset.get("placeholder")) or str(asset.get("status") or "") == "needs_identification":
            continue
        source_images = [
            str(value or "").strip()
            for value in (asset.get("source_images") or [])
            if str(value or "").strip()
        ]
        if not source_images and str(asset.get("source_image") or "").strip():
            source_images = [str(asset.get("source_image") or "").strip()]
        if not source_images:
            continue
        key = tuple(source_images)
        groups.setdefault(key, []).append(asset)

    for group in groups.values():
        if len(group) < 2:
            continue
        names = [str(item.get("name") or item.get("id") or "").strip() for item in group]
        for asset in group:
            asset["source_collision"] = True
            asset["source_identity_status"] = "ambiguous_same_frames"
            asset["source_collision_with"] = [
                name for name in names
                if name and name != str(asset.get("name") or "").strip()
            ]
            if str(asset.get("track_status") or "") in ("", "pending", "face_not_found", "ready"):
                asset["track_status"] = "identity_ambiguous"
            reason = str(asset.get("selection_reason") or "").strip()
            warning = "与其他角色使用了同一批源帧，说明当前还没有按人物身份分轨。"
            asset["selection_reason"] = f"{reason}；{warning}" if reason and warning not in reason else warning


def _flag_storyboard_asset_source_quality(assets: list[dict[str, Any]]) -> None:
    for asset in assets:
        kind = str(asset.get("kind") or "")
        if kind not in {"scene", "prop"}:
            continue
        asset["source_usage_role"] = ""
        keyframes = [
            item for item in (asset.get("keyframes") or [])
            if isinstance(item, dict)
        ]
        warning = ""
        status = ""
        if kind == "prop":
            status = "prop_needs_visual_check"
            asset["source_usage_role"] = "object_evidence_candidate"
            asset["source_selection_method"] = str(
                asset.get("source_selection_method") or "semantic_keyword_candidate"
            )
            if not _mode2_asset_is_manually_approved(asset):
                asset["visual_confirmed"] = False
                asset["source_visual_status"] = "unconfirmed_semantic_prop"
                asset["source_trust_level"] = "semantic_candidate_only"
            warning = "物品源图只是按关键词时间窗抽出来的候选证据，还没有目标检测确认本体；床、粗链条这类道具请优先人工核对，或直接换成清晰的物体图/裁剪图。"
        elif kind == "scene":
            _mode2_update_scene_visual_groups(asset)
            if _mode2_scene_is_mixed_visual_bundle(asset):
                status = "scene_mixed_visual_groups"
                asset["source_usage_role"] = "mixed_reference_bundle"
                warning = str(asset.get("source_quality_warning") or "这组场景混有人物近景和环境远景，请拆成不同镜头参考或只保留环境候选。")
            elif not keyframes:
                status = "scene_no_source_frames"
                asset["source_usage_role"] = "environment_reference"
                warning = "场景资产没有抽到可核对源帧，请人工补充空镜或远景环境图。"
            else:
                person_heavy = 0
                for frame in keyframes:
                    metrics = frame.get("metrics") if isinstance(frame.get("metrics"), dict) else {}
                    try:
                        skin = float(metrics.get("skin_ratio") or 0.0)
                        center_skin = float(metrics.get("center_skin_ratio") or 0.0)
                    except (TypeError, ValueError):
                        skin = center_skin = 0.0
                    if skin >= 0.16 or center_skin >= 0.18:
                        person_heavy += 1
                if person_heavy >= max(1, math.ceil(len(keyframes) * 0.5)):
                    status = "scene_person_heavy"
                    asset["source_usage_role"] = "shot_reference"
                    warning = "这组图更适合当镜头/动作/构图参考，不是干净背景；如果后续要做纯背景/环填参考，请再换空镜或远景。"
                else:
                    asset["source_usage_role"] = "environment_reference"
        if not warning:
            continue
        asset["source_quality_status"] = status
        asset["source_quality_warning"] = warning
        if status != "scene_person_heavy" and str(asset.get("track_status") or "") in {"", "pending", "ready"}:
            asset["track_status"] = "needs_manual_review"
        reason = str(asset.get("selection_reason") or "").strip()
        asset["selection_reason"] = f"{reason}；{warning}" if reason and warning not in reason else warning


def _mark_storyboard_mode2_candidate_asset_stage(assets: list[dict[str, Any]]) -> None:
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        kind = str(asset.get("kind") or "")
        asset["asset_stage"] = "candidate_after_timeline"
        asset["asset_stage_note"] = (
            "Built after Mode2 visual/Seedance timeline. Source images are evidence or candidates; "
            "target_image is the generation reference."
        )
        asset.setdefault("generation_reference_kind", "target_image")
        asset["generation_reference_image"] = str(asset.get("target_image") or "")
        asset["seedance_reference_image"] = str(asset.get("target_image") or "")
        asset["seedance_ready"] = bool(str(asset.get("target_image") or "").strip())
        asset["source_image_is_generation_source"] = False
        if kind == "role":
            source_image = str(asset.get("source_image") or "").strip()
            if source_image:
                asset.setdefault("identity_evidence_image", source_image)
                asset.setdefault("candidate_contact_sheet", source_image)
            asset["identity_evidence_only"] = True
            asset["source_usage_role"] = str(asset.get("source_usage_role") or "identity_evidence_only")
            asset["seedance_reference_role"] = "target_image_only"
        elif kind == "scene":
            asset["seedance_reference_role"] = "optional_scene_context_or_target_image"
        elif kind == "prop":
            asset["seedance_reference_role"] = "optional_prop_context_or_target_image"


def _mode2_safe_asset_slug(value: Any, fallback: str = "asset") -> str:
    text = str(value or "").strip() or fallback
    slug = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in text).strip("_")
    return (slug or fallback)[:56]


def _storyboard_existing_paths(values: list[Any]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for value in values:
        path_text = str(value or "").strip()
        if not path_text:
            continue
        path = Path(path_text)
        if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return paths


def _storyboard_asset_primary_source_path(asset: dict[str, Any], *, prefer_crop: bool = False) -> Path | None:
    candidates: list[Any] = []
    keyframes = [
        item for item in (asset.get("keyframes") or [])
        if isinstance(item, dict)
    ]
    keyframes.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    for frame in keyframes:
        if prefer_crop:
            candidates.extend([frame.get("crop_path"), frame.get("path")])
        else:
            candidates.extend([frame.get("path"), frame.get("crop_path")])
    candidates.extend(asset.get("source_images") or [])
    candidates.append(asset.get("source_image"))
    paths = _storyboard_existing_paths(candidates)
    return paths[0] if paths else None


def _storyboard_asset_refinement_dir(root: Path, asset: dict[str, Any]) -> Path:
    asset_id = _mode2_safe_asset_slug(asset.get("id") or asset.get("name") or "asset")
    target = root / "assets" / "refined" / asset_id
    target.mkdir(parents=True, exist_ok=True)
    return target


def _storyboard_refinement_prefix(job_id: str, asset: dict[str, Any], suffix: str) -> str:
    asset_id = _mode2_safe_asset_slug(asset.get("id") or "asset")
    return f"mode2_{job_id[:8]}_{asset_id}_{suffix}"


def _storyboard_prop_grounding_prompt(asset: dict[str, Any]) -> str:
    name = str(asset.get("name") or asset.get("tag") or "").strip()
    text = name.lower()
    aliases: list[str] = []
    if "床" in name or "bed" in text:
        aliases = ["bed", "mattress"]
    elif "链" in name or "chain" in text:
        aliases = ["chain", "iron chain", "metal chain"]
    elif "车" in name or "car" in text or "vehicle" in text:
        aliases = ["car", "vehicle"]
    elif "手机" in name or "phone" in text:
        aliases = ["phone", "mobile phone"]
    elif "刀" in name or "knife" in text:
        aliases = ["knife"]
    elif "包" in name or "bag" in text:
        aliases = ["bag", "handbag"]
    else:
        ascii_name = "".join(ch for ch in name if ord(ch) < 128).strip()
        aliases = [ascii_name or "object"]
    return " . ".join(alias for alias in aliases if alias).strip() or "object"


def _mode2_grounded_sam_workflow(
    image_name: str,
    *,
    prompt: str,
    filename_prefix: str,
    threshold: float = 0.3,
    sam_model: str = "sam_vit_b (375MB)",
    dino_model: str = "GroundingDINO_SwinT_OGC (694MB)",
    crop: bool = True,
) -> tuple[dict[str, Any], dict[str, str]]:
    workflow: dict[str, Any] = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {
            "class_type": "SAMModelLoader (segment anything)",
            "inputs": {"model_name": sam_model},
        },
        "3": {
            "class_type": "GroundingDinoModelLoader (segment anything)",
            "inputs": {"model_name": dino_model},
        },
        "4": {
            "class_type": "GroundingDinoSAMSegment (segment anything)",
            "inputs": {
                "sam_model": ["2", 0],
                "grounding_dino_model": ["3", 0],
                "image": ["1", 0],
                "prompt": prompt,
                "threshold": max(0.05, min(0.95, float(threshold or 0.3))),
            },
        },
        "5": {"class_type": "MaskToImage", "inputs": {"mask": ["4", 1]}},
        "6": {
            "class_type": "SaveImage",
            "inputs": {"images": ["5", 0], "filename_prefix": f"{filename_prefix}_mask"},
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {"images": ["4", 0], "filename_prefix": f"{filename_prefix}_segment"},
        },
    }
    output_nodes = {"mask": "6", "segment": "7"}
    if crop:
        workflow["8"] = {
            "class_type": "ImageCropByMaskAndResize",
            "inputs": {
                "image": ["1", 0],
                "mask": ["4", 1],
                "base_resolution": 768,
                "padding": 48,
                "min_crop_resolution": 192,
                "max_crop_resolution": 1024,
            },
        }
        workflow["9"] = {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": f"{filename_prefix}_crop"},
        }
        output_nodes["crop"] = "9"
    return workflow, output_nodes


def _mode2_lama_scene_workflow(
    image_name: str,
    *,
    filename_prefix: str,
    threshold: float = 0.28,
    sam_model: str = "sam_vit_b (375MB)",
    dino_model: str = "GroundingDINO_SwinT_OGC (694MB)",
) -> tuple[dict[str, Any], dict[str, str]]:
    workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {
            "class_type": "SAMModelLoader (segment anything)",
            "inputs": {"model_name": sam_model},
        },
        "3": {
            "class_type": "GroundingDinoModelLoader (segment anything)",
            "inputs": {"model_name": dino_model},
        },
        "4": {
            "class_type": "GroundingDinoSAMSegment (segment anything)",
            "inputs": {
                "sam_model": ["2", 0],
                "grounding_dino_model": ["3", 0],
                "image": ["1", 0],
                "prompt": "person . human . people",
                "threshold": max(0.05, min(0.95, float(threshold or 0.28))),
            },
        },
        "5": {
            "class_type": "INPAINT_ExpandMask",
            "inputs": {"mask": ["4", 1], "grow": 24, "blur": 7, "blur_type": "gaussian"},
        },
        "6": {"class_type": "INPAINT_LoadInpaintModel", "inputs": {"model_name": "big-lama.pt"}},
        "7": {
            "class_type": "INPAINT_InpaintWithModel",
            "inputs": {"inpaint_model": ["6", 0], "image": ["1", 0], "mask": ["5", 0], "seed": 0},
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {"images": ["7", 0], "filename_prefix": f"{filename_prefix}_clean_bg"},
        },
        "9": {"class_type": "MaskToImage", "inputs": {"mask": ["5", 0]}},
        "10": {
            "class_type": "SaveImage",
            "inputs": {"images": ["9", 0], "filename_prefix": f"{filename_prefix}_person_mask"},
        },
        "11": {
            "class_type": "SaveImage",
            "inputs": {"images": ["4", 0], "filename_prefix": f"{filename_prefix}_person_segment"},
        },
    }
    return workflow, {"clean_bg": "8", "mask": "10", "segment": "11"}


def _mode2_run_image_workflow(
    client: ComfyClient,
    workflow: dict[str, Any],
    output_nodes: dict[str, str],
    *,
    output_dir: Path,
    local_prefix: str,
    add_log,
) -> dict[str, Any]:
    prompt_id, history = client.run_workflow(workflow, log=add_log)
    outputs = (history.get(prompt_id) or {}).get("outputs") or {}
    result: dict[str, Any] = {"prompt_id": prompt_id}
    for key, node_id in output_nodes.items():
        asset = client.first_output_asset(outputs.get(str(node_id)) or {})
        if not asset:
            continue
        suffix = Path(str(asset.get("filename") or "")).suffix.lower() or ".png"
        target = output_dir / f"{local_prefix}_{key}{suffix}"
        url = client.download_output_asset(asset, target)
        result[key] = str(target)
        result[f"{key}_url"] = url
    return result


def _mode2_mask_quality(mask_path: str | Path) -> dict[str, Any]:
    path = Path(str(mask_path or ""))
    if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
        return {}
    try:
        from PIL import Image
        import numpy as np
    except Exception:  # noqa: BLE001
        return {}
    try:
        image = Image.open(path).convert("L")
    except Exception:  # noqa: BLE001
        return {}
    arr = np.asarray(image)
    selected = arr > 16
    height, width = selected.shape[:2]
    total = max(1, int(width * height))
    area = int(selected.sum())
    if area <= 0:
        return {"width": width, "height": height, "area_ratio": 0.0, "bbox_ratio": 0.0, "component_count": 0}
    ys, xs = np.where(selected)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    component_count = 0
    try:
        import cv2

        _count, labels, stats, _centroids = cv2.connectedComponentsWithStats(selected.astype("uint8"), 8)
        min_component_area = max(24, total * 0.0015)
        component_count = int(sum(1 for item in stats[1:] if int(item[cv2.CC_STAT_AREA]) >= min_component_area))
    except Exception:  # noqa: BLE001
        component_count = 1
    return {
        "width": width,
        "height": height,
        "area_ratio": round(area / total, 4),
        "bbox_ratio": round(bbox_area / total, 4),
        "component_count": component_count,
        "bbox": [x1, y1, x2 - x1, y2 - y1],
    }


def _set_storyboard_asset_refinement_failure(
    asset: dict[str, Any],
    *,
    job_id: str,
    error: str,
    status: str = "failed",
) -> None:
    asset["refinement_status"] = status
    asset["refinement_error"] = error
    asset["refinement_job_id"] = job_id
    asset["refined_at"] = time.time()


def _refine_storyboard_role_asset(
    root: Path,
    asset: dict[str, Any],
    *,
    job_id: str,
    add_log,
) -> dict[str, Any]:
    output_dir = _storyboard_asset_refinement_dir(root, asset)
    source_paths = _storyboard_existing_paths(asset.get("source_images") or [])
    if not source_paths:
        primary = _storyboard_asset_primary_source_path(asset, prefer_crop=True)
        source_paths = [primary] if primary else []
    if not source_paths:
        raise RuntimeError("role source image not found")

    slug = _storyboard_refinement_prefix(job_id, asset, "role")
    if len(source_paths) > 1:
        target = output_dir / f"{slug}_face_bundle.jpg"
        _make_storyboard_asset_contact_sheet(source_paths[:6], target)
    else:
        source = source_paths[0]
        target = output_dir / f"{slug}_reference{source.suffix.lower() or '.jpg'}"
        shutil.copy2(source, target)
    if not target.exists() or target.stat().st_size <= 0:
        raise RuntimeError("role refined bundle was not created")

    warning = ""
    status = "ready"
    if asset.get("source_collision") or str(asset.get("source_identity_status") or "") == "ambiguous_same_frames":
        warning = "当前人物源帧和其他角色撞车，这张提纯图只能当候选，最好补手工身份标注或替换为清晰角色图。"
        status = "needs_manual_review"
    elif str(asset.get("track_status") or "") == "face_not_found":
        warning = "没有检测到清晰人脸，当前只是人物关键帧候选，建议手工换成清晰角色图。"
        status = "needs_manual_review"

    asset.update({
        "refinement_status": status,
        "refinement_method": "role_face_bundle",
        "refinement_kind": "role_reference_bundle",
        "refined_source_image": str(target),
        "refined_source_images": [str(path) for path in source_paths],
        "refined_mask_image": "",
        "refined_cutout_image": "",
        "refinement_prompt": "role identity bundle from visually selected face/person crops",
        "refinement_warning": warning,
        "refinement_error": "",
        "refinement_job_id": job_id,
        "refinement_source_path": str(source_paths[0]),
        "refined_at": time.time(),
    })
    add_log(f"> 角色提纯完成: {asset.get('name')} -> {target.name}")
    return {"asset_id": asset.get("id"), "status": status, "path": str(target), "warning": warning}


def _refine_storyboard_scene_asset(
    root: Path,
    asset: dict[str, Any],
    *,
    client: ComfyClient,
    job_id: str,
    add_log,
) -> dict[str, Any]:
    source = _storyboard_asset_primary_source_path(asset, prefer_crop=False)
    if source is None:
        raise RuntimeError("scene source frame not found")
    output_dir = _storyboard_asset_refinement_dir(root, asset)
    remote_name = client.upload_file(source)
    prefix = _storyboard_refinement_prefix(job_id, asset, "scene")
    workflow, output_nodes = _mode2_lama_scene_workflow(remote_name, filename_prefix=prefix)
    add_log(f"> 场景提纯提交 ComfyUI: {asset.get('name')} / source={source.name}")
    result = _mode2_run_image_workflow(
        client,
        workflow,
        output_nodes,
        output_dir=output_dir,
        local_prefix=prefix,
        add_log=add_log,
    )
    clean_bg = str(result.get("clean_bg") or "")
    if not clean_bg:
        raise RuntimeError("LaMa finished but clean background output was missing")
    provenance, provenance_path = _mode2_write_refinement_provenance(
        output_dir,
        prefix,
        asset,
        source_path=source,
        outputs={
            "refined_source_image": clean_bg,
            "refined_mask_image": str(result.get("mask") or ""),
            "refined_segment_image": str(result.get("segment") or ""),
        },
        job_id=job_id,
        prompt_id=str(result.get("prompt_id") or ""),
        method="grounding_dino_sam_lama",
        kind="clean_background",
    )
    provenance_source = provenance.get("source") if isinstance(provenance.get("source"), dict) else {}
    asset.update({
        "refinement_status": "ready",
        "refinement_method": "grounding_dino_sam_lama",
        "refinement_kind": "clean_background",
        "refined_source_image": clean_bg,
        "refined_source_images": [clean_bg],
        "refined_mask_image": str(result.get("mask") or ""),
        "refined_cutout_image": "",
        "refinement_prompt": "person . human . people",
        "refinement_warning": "",
        "refinement_error": "",
        "refinement_job_id": job_id,
        "refinement_prompt_id": str(result.get("prompt_id") or ""),
        "refinement_source_path": str(source),
        "refinement_source_hash": str(provenance_source.get("hash") or ""),
        "refinement_source_fingerprint": str(provenance_source.get("fingerprint") or ""),
        "refinement_provenance": provenance,
        "refinement_provenance_path": provenance_path,
        "refinement_provenance_status": "verified",
        "refinement_provenance_verified": True,
        "refinement_provenance_warning": "",
        "refined_at": time.time(),
    })
    add_log(f"> 场景提纯完成: {asset.get('name')} -> {Path(clean_bg).name}")
    return {"asset_id": asset.get("id"), "status": "ready", "path": clean_bg}


def _refine_storyboard_prop_asset(
    root: Path,
    asset: dict[str, Any],
    *,
    client: ComfyClient,
    job_id: str,
    add_log,
) -> dict[str, Any]:
    source = _storyboard_asset_primary_source_path(asset, prefer_crop=False)
    if source is None:
        raise RuntimeError("prop source frame not found")
    output_dir = _storyboard_asset_refinement_dir(root, asset)
    prompt = _storyboard_prop_grounding_prompt(asset)
    remote_name = client.upload_file(source)
    prefix = _storyboard_refinement_prefix(job_id, asset, "prop")
    workflow, output_nodes = _mode2_grounded_sam_workflow(
        remote_name,
        prompt=prompt,
        filename_prefix=prefix,
        threshold=0.28,
        crop=True,
    )
    add_log(f"> 物品提纯提交 ComfyUI: {asset.get('name')} / prompt={prompt} / source={source.name}")
    result = _mode2_run_image_workflow(
        client,
        workflow,
        output_nodes,
        output_dir=output_dir,
        local_prefix=prefix,
        add_log=add_log,
    )
    crop = str(result.get("crop") or result.get("segment") or "")
    if not crop:
        raise RuntimeError("GroundingDINO+SAM finished but object crop output was missing")
    mask_path = str(result.get("mask") or "")
    mask_quality = _mode2_mask_quality(mask_path)
    warning = "物品提纯按目标词检测得到，仍建议人工核对是否真是该物体。"
    status = "ready"
    try:
        area_ratio = float(mask_quality.get("area_ratio") or 0.0)
        bbox_ratio = float(mask_quality.get("bbox_ratio") or 0.0)
        component_count = int(mask_quality.get("component_count") or 0)
    except (TypeError, ValueError):
        area_ratio = bbox_ratio = 0.0
        component_count = 0
    if not mask_quality:
        status = "needs_manual_review"
        warning = "没有拿到可评估的物品 mask，请人工核对提纯图。"
    elif area_ratio <= 0.0005:
        status = "needs_manual_review"
        warning = "物品 mask 几乎为空，模型可能没有找到目标物。"
    elif area_ratio >= 0.22 or bbox_ratio >= 0.58 or component_count >= 7:
        status = "needs_manual_review"
        warning = (
            "物品 mask 范围过大或过散，可能把背景/人物/多个同类物一起选进来了；"
            "请人工核对，必要时换更清晰的物品源图。"
        )
    provenance, provenance_path = _mode2_write_refinement_provenance(
        output_dir,
        prefix,
        asset,
        source_path=source,
        outputs={
            "refined_source_image": crop,
            "refined_cutout_image": crop,
            "refined_mask_image": mask_path,
            "refined_segment_image": str(result.get("segment") or ""),
        },
        job_id=job_id,
        prompt_id=str(result.get("prompt_id") or ""),
        method="grounding_dino_sam",
        kind="prop_cutout",
    )
    provenance_source = provenance.get("source") if isinstance(provenance.get("source"), dict) else {}
    asset.update({
        "refinement_status": status,
        "refinement_method": "grounding_dino_sam",
        "refinement_kind": "prop_cutout",
        "refined_source_image": crop,
        "refined_source_images": [crop],
        "refined_mask_image": mask_path,
        "refined_cutout_image": crop,
        "refinement_prompt": prompt,
        "refinement_warning": warning,
        "refinement_error": "",
        "refinement_job_id": job_id,
        "refinement_prompt_id": str(result.get("prompt_id") or ""),
        "refinement_source_path": str(source),
        "refinement_source_hash": str(provenance_source.get("hash") or ""),
        "refinement_source_fingerprint": str(provenance_source.get("fingerprint") or ""),
        "refinement_provenance": provenance,
        "refinement_provenance_path": provenance_path,
        "refinement_provenance_status": "verified",
        "refinement_provenance_verified": True,
        "refinement_provenance_warning": "",
        "refinement_quality": mask_quality,
        "refined_at": time.time(),
    })
    add_log(f"> 物品提纯完成: {asset.get('name')} -> {Path(crop).name} / status={status}")
    return {"asset_id": asset.get("id"), "status": status, "path": crop, "prompt": prompt, "quality": mask_quality}


def _storyboard_asset_index(asset_id: Any) -> int:
    try:
        return int(str(asset_id or "").rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        return 0


def _storyboard_asset_keywords(asset: dict[str, Any]) -> tuple[str, ...]:
    name = str(asset.get("name") or "").strip()
    if name in {"粗链条", "粗铁链"}:
        return ("粗链条", "粗铁链", "铁链", "锁链", "链条")
    if name == "车辆":
        return ("车辆", "汽车", "轿车", "车流", "车内")
    if name == "床":
        return ("床上", "床边", "床铺", "床")
    return (name,) if name else ()


def _storyboard_role_keywords(asset: dict[str, Any]) -> tuple[str, ...]:
    ignored = {"", "角色", "角色1", "角色2", "角色3", "待识别角色", "待命名", "占位"}
    values = [
        str(asset.get("name") or "").strip(),
        str(asset.get("tag") or "").strip(),
        str(asset.get("alias") or "").strip(),
        str(asset.get("visual_label") or "").strip(),
    ]
    prompt = ""
    for token in ("未婚夫", "丈夫", "妻子", "白衣男子", "黑衣男子", "女子", "男人", "女人"):
        if token in prompt:
            values.append(token)
    keywords: list[str] = []
    for value in values:
        if value in ignored or len(value) < 2:
            continue
        if value not in keywords:
            keywords.append(value)
    return tuple(keywords)


def _storyboard_shot_usage_text(shot: dict[str, Any]) -> str:
    parts: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text:
            parts.append(text)

    for key in (
        "description",
        "summary",
        "visual_description",
        "raw_caption",
        "caption",
        "key_action",
        "action",
        "dialogue",
        "dialog",
        "scene",
        "scene_name",
        "location",
    ):
        add(shot.get(key))

    for key in ("characters", "character_names", "detected_characters"):
        value = shot.get(key)
        if isinstance(value, list):
            for item in value:
                add(item.get("name") if isinstance(item, dict) else item)
        elif isinstance(value, dict):
            for item in value.values():
                add(item.get("name") if isinstance(item, dict) else item)
        else:
            add(value)

    semantic_items: list[Any] = []
    if isinstance(shot.get("semantic_scene"), dict):
        semantic_items.append(shot.get("semantic_scene"))
    if isinstance(shot.get("semantic_scenes"), list):
        semantic_items.extend(shot.get("semantic_scenes") or [])
    for item in semantic_items:
        if not isinstance(item, dict):
            continue
        for key in (
            "name",
            "scene",
            "scene_name",
            "location",
            "description",
            "summary",
            "key_action",
            "action",
            "dialogue",
        ):
            add(item.get(key))
        for char in item.get("characters") or []:
            add(char.get("name") if isinstance(char, dict) else char)

    return " ".join(parts)


def _storyboard_role_anchor_shots(asset: dict[str, Any]) -> set[str]:
    shots: set[str] = {
        str(value or "").strip()
        for value in (asset.get("present_shots") or [])
        if str(value or "").strip()
    }
    for anchor in asset.get("identity_anchors") or []:
        if not isinstance(anchor, dict):
            continue
        shot_id = str(anchor.get("shot_id") or "").strip()
        if shot_id:
            shots.add(shot_id)
    return shots


def _storyboard_role_should_attach_to_shot(
    asset: dict[str, Any],
    *,
    shot_id: str,
    shot_text: str,
    shot_source_ids: set[str],
    asset_source_ids: set[str],
) -> tuple[bool, str]:
    anchored_shots = _storyboard_role_anchor_shots(asset)
    if anchored_shots:
        return shot_id in anchored_shots, "identity_track_or_anchor"

    keywords = _storyboard_role_keywords(asset)
    if keywords and any(keyword in shot_text for keyword in keywords):
        return True, "explicit_text_keyword"

    if asset_source_ids and shot_source_ids and len(asset_source_ids) <= 6:
        return bool(asset_source_ids & shot_source_ids), "narrow_source_overlap"

    return False, "identity_unconfirmed"


def _attach_storyboard_asset_usage(
    assets: list[dict[str, Any]],
    shots: list[dict[str, Any]],
) -> None:
    usage: dict[str, list[str]] = {
        str(asset.get("id") or ""): []
        for asset in assets
        if str(asset.get("id") or "")
    }
    evidence_usage: dict[str, list[str]] = {
        str(asset.get("id") or ""): []
        for asset in assets
        if str(asset.get("id") or "")
    }
    for shot in shots:
        shot_id = str(shot.get("segment_id") or "")
        shot_source_ids = {
            str(value or "").strip()
            for value in (shot.get("source_segment_ids") or [])
            if str(value or "").strip()
        }
        text = _storyboard_shot_usage_text(shot)
        try:
            person_count = int(shot.get("person_count") or -1)
        except (TypeError, ValueError):
            person_count = -1

        used_asset_ids: list[str] = []
        for asset in assets:
            asset_id = str(asset.get("id") or "")
            if not asset_id:
                continue
            kind = str(asset.get("kind") or "")
            should_use = False
            match_strategy = ""
            if kind == "scene" and _mode2_scene_asset_is_superseded(asset):
                continue
            asset_source_ids = {
                str(value or "").strip()
                for value in (asset.get("source_segment_ids") or [])
                if str(value or "").strip()
            }
            if asset_source_ids and shot_source_ids and (asset_source_ids & shot_source_ids):
                evidence_usage.setdefault(asset_id, []).append(shot_id)
            if kind == "role":
                if bool(asset.get("placeholder")) or str(asset.get("status") or "") == "needs_identification":
                    should_use = False
                    match_strategy = "placeholder"
                else:
                    should_use, match_strategy = _storyboard_role_should_attach_to_shot(
                        asset,
                        shot_id=shot_id,
                        shot_text=text,
                        shot_source_ids=shot_source_ids,
                        asset_source_ids=asset_source_ids,
                    )
                    if should_use and person_count == 0:
                        should_use = False
                        match_strategy = "no_person_in_shot"
            elif kind == "scene" and str(asset.get("split_parent_asset_id") or "").strip():
                assigned_shots = set(_string_list(asset.get("split_assigned_shot_ids")))
                should_use = shot_id in assigned_shots
                match_strategy = "scene_split_time_window" if should_use else ""
            elif asset_source_ids and shot_source_ids:
                should_use = bool(asset_source_ids & shot_source_ids)
                match_strategy = "source_overlap"
            elif kind == "scene":
                should_use = not asset_source_ids
                match_strategy = "global_scene_fallback" if should_use else ""
            elif kind == "prop":
                should_use = any(keyword and keyword in text for keyword in _storyboard_asset_keywords(asset))
                match_strategy = "prop_keyword" if should_use else ""

            if should_use:
                used_asset_ids.append(asset_id)
                usage.setdefault(asset_id, []).append(shot_id)
                strategies = asset.setdefault("_usage_match_strategies", [])
                if isinstance(strategies, list) and match_strategy and match_strategy not in strategies:
                    strategies.append(match_strategy)
        shot["asset_ids"] = used_asset_ids

    for asset in assets:
        asset_id = str(asset.get("id") or "")
        used_shots = [shot_id for shot_id in usage.get(asset_id, []) if shot_id]
        evidence_shots = [shot_id for shot_id in evidence_usage.get(asset_id, []) if shot_id]
        asset["used_shots"] = used_shots
        asset["usage_count"] = len(used_shots)
        asset["evidence_shots"] = evidence_shots
        asset["evidence_coverage_count"] = len(evidence_shots)
        strategies = [
            str(value or "").strip()
            for value in (asset.get("_usage_match_strategies") or [])
            if str(value or "").strip()
        ]
        if strategies:
            asset["usage_match_strategies"] = strategies
        asset.pop("_usage_match_strategies", None)


def _storyboard_asset_label(asset: dict[str, Any], assets: list[dict[str, Any]]) -> str:
    try:
        index = assets.index(asset) + 1
    except ValueError:
        index = 1
    return f"图片{index}"


def _storyboard_asset_ref(
    asset: dict[str, Any],
    assets: list[dict[str, Any]],
) -> dict[str, Any]:
    representative_image = str(asset.get("representative_image") or "")
    representative_is_clean = bool(asset.get("representative_is_clean"))
    return {
        "id": str(asset.get("id") or ""),
        "kind": str(asset.get("kind") or ""),
        "name": str(asset.get("name") or ""),
        "tag": str(asset.get("tag") or ""),
        "image_label": _storyboard_asset_label(asset, assets),
        "target_image": str(asset.get("target_image") or ""),
        "manual_asset_status": str(asset.get("manual_asset_status") or ""),
        "source_usage_role": str(asset.get("source_usage_role") or ""),
        "refinement_status": str(asset.get("refinement_status") or ""),
        "refinement_kind": str(asset.get("refinement_kind") or ""),
        "representative_image": representative_image,
        "representative_status": str(asset.get("representative_status") or ""),
        "representative_is_clean": representative_is_clean,
        "representative_warning": str(asset.get("representative_warning") or ""),
        "refined_source_image": representative_image if representative_is_clean else "",
        "refined_mask_image": str(asset.get("refined_mask_image") or ""),
    }


def _compile_storyboard_prompt(
    shot: dict[str, Any],
    assets: list[dict[str, Any]],
    project_config: dict[str, str] | None = None,
) -> str:
    config = _normalize_storyboard_project_config(project_config or {})
    by_id = {str(asset.get("id") or ""): asset for asset in assets}
    used_assets = [
        by_id[asset_id]
        for asset_id in shot.get("asset_ids", [])
        if asset_id in by_id
    ]
    used_assets = [
        asset for asset in used_assets
        if str(asset.get("manual_asset_status") or "") != "ignored"
        and not (
            str(asset.get("manual_asset_status") or "") == "needs_replacement"
            and not str(asset.get("target_image") or "").strip()
        )
    ]
    roles = [asset for asset in used_assets if asset.get("kind") == "role"]
    scenes = [asset for asset in used_assets if asset.get("kind") == "scene"]
    props = [asset for asset in used_assets if asset.get("kind") == "prop"]

    def binding_lines(items: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for asset in items:
            label = _storyboard_asset_label(asset, assets)
            kind = str(asset.get("kind") or "")
            kind_name = {"role": "角色", "scene": "场景", "prop": "道具"}.get(kind, "资产")
            tag = str(asset.get("tag") or "").strip()
            name = str(asset.get("name") or "").strip()
            suffix = f"（{tag}）" if tag else ""
            lines.append(f"{name}@{label}={kind_name}{suffix}")
        return lines

    role_line = "、".join(binding_lines(roles)) or "无明确角色，按原视频画面判断。"
    scene_line = "、".join(binding_lines(scenes)) or "按原视频空间关系生成。"
    prop_line = "、".join(binding_lines(props)) or "无强制道具。"
    description = str(shot.get("description") or "待模型根据原视频画面反推剧情动作").strip()
    duration = float(shot.get("duration") or 0.0)
    start = float(shot.get("start") or 0.0)
    end = float(shot.get("end") or start + duration)
    segment_id = str(shot.get("segment_id") or "")
    rebuild_meta = STORYBOARD_REBUILD_GOALS[config["rebuild_goal"]]
    style_meta = STORYBOARD_STYLE_PRESETS[config["style_preset"]]
    language_meta = STORYBOARD_LANGUAGE_PRESETS[config["target_language"]]
    motion_meta = STORYBOARD_MOTION_PRESETS[config["motion_constraint"]]
    mask_meta = STORYBOARD_MASK_SOURCE_PRESETS[config["mask_source"]]

    lines = [
        "面部五官清晰稳定不变形，同一角色全程外貌一致。人体结构正常比例自然，动作连续不跳帧。视频全程同一时刻每名角色只生成一个唯一个体，不出现重复人物、分身或双胞胎效果。无模糊、无重影、无字幕、无文字覆盖、无水印、无背景音乐。",
        f"【镜头编号】{segment_id}",
        f"【项目默认】{_storyboard_project_config_summary(config)}",
        *rebuild_meta["prompt"],
        f"【画面风格】{style_meta['label']}：{style_meta['prompt']}",
        f"【目标语境】{language_meta['label']}：{language_meta['prompt']} 当前镜头先专注画面，不生成任何字幕文字。",
        f"【输出比例】{config['output_ratio']}；构图需适配该比例，主体不要被裁切出画。",
        f"【蒙版来源】{mask_meta['label']}：{mask_meta['prompt']}",
        f"【参考时段】原视频 {start:.2f}-{end:.2f} 秒，生成约 {duration:.1f} 秒；不要扩写到其他剧情，不要把前后镜头动作混进来。",
        f"【本镜头剧情】{description}",
        f"【角色绑定】{role_line}",
        f"【场景绑定】{scene_line}",
        f"【道具绑定】{prop_line}",
        f"【镜头约束】{motion_meta['label']}：{motion_meta['prompt']}",
        "【原片运动骨架】原视频只作为镜头底稿，保持此时间段的人物站位、动作姿态、运动方向、互动节奏、镜头景别、遮挡关系和空间距离；不要重新编排动作，不要改变人物左右/前后关系。",
        "【生成要求】根据本镜头关联资产重新生成短剧画面。角色外貌、年龄、脸型、发型、身材比例、服装颜色和服装轮廓分别跟对应参考图一致；场景和道具跟对应参考图一致，但镜头构图和动作关系必须跟原视频此时间段一致，并整体服从项目风格设定。",
        "【避免】多余人物、角色互换、混脸、串衣服、肢体变形、改动作、改站位、重新跳舞、舞台背景、字幕、水印、文字。",
    ]
    return "\n".join(lines)


def _compile_storyboard_prompts(
    assets: list[dict[str, Any]],
    shots: list[dict[str, Any]],
    project_config: dict[str, str] | None = None,
) -> None:
    by_id = {str(asset.get("id") or ""): asset for asset in assets}
    for shot in shots:
        refs = [
            _storyboard_asset_ref(by_id[asset_id], assets)
            for asset_id in shot.get("asset_ids", [])
            if asset_id in by_id
        ]
        shot["asset_refs"] = refs
        analysis_prompt = _compile_storyboard_prompt(shot, assets, project_config=project_config)
        seedance_prompt = _compile_storyboard_seedance_prompt(shot, assets, project_config=project_config)
        shot["analysis_prompt"] = analysis_prompt
        shot["compiled_prompt_long"] = analysis_prompt
        shot["seedance_prompt"] = seedance_prompt
        shot["prompt"] = seedance_prompt
        shot["compiled_prompt"] = seedance_prompt


def _compile_storyboard_seedance_prompt(
    shot: dict[str, Any],
    assets: list[dict[str, Any]],
    project_config: dict[str, str] | None = None,
) -> str:
    """Seedance2 works best with a short remake instruction; keep analysis internal."""
    config = _normalize_storyboard_project_config(project_config or {})
    by_id = {str(asset.get("id") or ""): asset for asset in assets}
    role_count = sum(
        1
        for asset_id in shot.get("asset_ids", [])
        if asset_id in by_id
        and str(by_id[asset_id].get("kind") or "") == "role"
        and str(by_id[asset_id].get("manual_asset_status") or "") != "ignored"
    )
    lines = [
        "把视频里的原人物替换成参考图中的人物。",
        "保持原视频的动作、表情、镜头和剧情节奏。",
    ]
    if role_count > 1:
        lines.append("多人按参考图顺序对应替换。")
    if config["rebuild_goal"] == "english_europe_foreign_cast":
        lines.append("背景换成欧洲风格，人物换成外国人形象。")
    else:
        lines.append("画面重新生成得自然真实。")
    lines.append("不要字幕、水印、文字。")
    return "".join(lines)


def _storyboard_seedance_target_seconds(
    total_duration: float,
    reference_segments: list[dict[str, Any]] | None = None,
) -> float:
    """Pick a Seedance-friendly storyboard window for mode2."""
    min_seconds = 4.0
    max_seconds = 15.0
    total = max(0.0, float(total_duration or 0.0))
    if total <= min_seconds:
        return round(max(0.1, total or min_seconds), 3)

    segment_durations: list[float] = []
    for item in reference_segments or []:
        try:
            start = float(item.get("start") or 0.0)
            end = float(item.get("end") or 0.0)
        except (TypeError, ValueError):
            continue
        piece = end - start
        if piece > 0:
            segment_durations.append(piece)

    if segment_durations:
        average = sum(segment_durations) / len(segment_durations)
        multiplier = 1.6 if len(segment_durations) >= 4 else 1.45 if len(segment_durations) >= 2 else 1.3
        target = average * multiplier
    else:
        if total <= 12:
            target = max(4.0, total / 2.0)
        elif total <= 30:
            target = 5.5
        elif total <= 60:
            target = 6.5
        elif total <= 120:
            target = 7.5
        else:
            target = 8.5

    return round(max(min_seconds, min(max_seconds, target)), 3)


def _storyboard_mode2_video_hash(video_path: str) -> str:
    path = Path(video_path)
    try:
        stat = path.stat()
        identity = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    except Exception:  # noqa: BLE001
        identity = str(path)
    return hashlib.sha256(identity.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _storyboard_mode2_project_dir(video_path: str) -> Path:
    return STORYBOARD_MODE2_PROJECT_ROOT / _storyboard_mode2_video_hash(video_path)


def _storyboard_mode2_understanding_path(project_dir: str | Path) -> Path:
    return Path(project_dir) / "understanding" / "pre_director.json"


def _storyboard_same_source_path(left: Any, right: Any) -> bool:
    try:
        return str(Path(str(left or "")).resolve()).casefold() == str(Path(str(right or "")).resolve()).casefold()
    except Exception:  # noqa: BLE001
        return str(left or "").strip().casefold() == str(right or "").strip().casefold()


def _storyboard_mode2_has_audio(video_path: str) -> bool:
    try:
        from spvideo.ffmpeg_tools import probe_video

        meta = probe_video(Path(video_path))
        return bool(str(meta.audio_codec or "").strip())
    except Exception:  # noqa: BLE001
        return False


def _storyboard_mode2_sample_times(duration: float, max_frames: int = 160) -> list[float]:
    total = max(0.0, float(duration or 0.0))
    if total <= 0.1:
        return [0.1]
    end_time = max(0.05, total - min(0.5, total * 0.05))
    if total <= 30:
        interval = 0.5
    elif total <= 180:
        interval = 1.0
    else:
        interval = 2.0
    count = min(max(3, int(math.ceil(total / interval)) + 1), max(3, int(max_frames)))
    if count <= 1:
        return [round(min(total, 0.1), 3)]
    values = [
        min(end_time, index * end_time / (count - 1))
        for index in range(count)
    ]
    result: list[float] = []
    seen: set[int] = set()
    for value in values:
        time_value = round(max(0.05, value), 3)
        key = int(round(time_value * 1000))
        if key not in seen:
            seen.add(key)
            result.append(time_value)
    return result


def _storyboard_mode2_sample_frame_features(
    video_path: str,
    project_dir: str | Path,
    *,
    duration: float,
    max_frames: int = 160,
) -> list[Any]:
    frames_dir = Path(project_dir) / "understanding" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    features: list[Any] = []
    previous_pixels = None
    for index, time_seconds in enumerate(_storyboard_mode2_sample_times(duration, max_frames=max_frames), 1):
        frame_path = frames_dir / f"frame_{index:04d}_{int(round(time_seconds * 1000)):08d}ms.jpg"
        if not frame_path.exists() or frame_path.stat().st_size <= 0:
            try:
                extract_frame(video_path, time_seconds, frame_path)
            except Exception:  # noqa: BLE001
                continue
        if not frame_path.exists() or frame_path.stat().st_size <= 0:
            continue
        try:
            frame_features, previous_pixels = analyze_frame(frame_path, time_seconds, previous_pixels)
        except Exception:  # noqa: BLE001
            continue
        features.append(frame_features)
    return features


def _storyboard_mode2_understanding_view(
    plan: dict[str, Any],
    *,
    project_dir: str | Path,
    cache_path: str | Path,
    source_path: str,
    cache_hit: bool = False,
) -> dict[str, Any]:
    status = str(plan.get("status") or "unknown")
    return {
        "status": status,
        "summary": str(plan.get("summary") or plan.get("story_summary") or ""),
        "characters": [
            item for item in (plan.get("characters") or [])
            if isinstance(item, dict)
        ],
        "scenes": [
            item for item in (plan.get("scenes") or [])
            if isinstance(item, dict)
        ],
        "boundary_hints": [
            item for item in (plan.get("boundary_hints") or [])
            if isinstance(item, dict)
        ],
        "source": str(plan.get("source") or "mode2_pre_director"),
        "source_path": source_path,
        "cache_path": str(cache_path),
        "project_dir": str(project_dir),
        "cache_hit": bool(cache_hit),
        "generated_at": plan.get("generated_at"),
        "sampled_frame_count": int(plan.get("sampled_frame_count") or 0),
        "analysis_frame_count": int(plan.get("analysis_frame_count") or 0),
        "audio_status": str(plan.get("audio_status") or ""),
        "audio_understanding": bool(plan.get("audio_understanding")),
        "note": str(plan.get("note") or ""),
        "error": str(plan.get("error") or ""),
        "asset_manifest": (
            plan.get("asset_manifest")
            if isinstance(plan.get("asset_manifest"), dict)
            else {"results": []}
        ),
        "analysis_frame_manifest": [
            item
            for item in (
                plan.get("analysis_frame_manifest")
                or plan.get("frame_manifest")
                or []
            )
            if isinstance(item, dict)
        ],
        "understanding_analysis_fresh": bool(
            plan.get("understanding_analysis_fresh") and not cache_hit
        ),
    }


def _mode2_curator_manifest_results(understanding: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = understanding.get("asset_manifest")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("results"), list):
        return []
    return [item for item in manifest["results"] if isinstance(item, dict)]


def _mode2_curator_frame_manifest(understanding: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in (understanding.get("analysis_frame_manifest") or [])
        if isinstance(item, dict)
    ]


def _mode2_curator_visual_groups(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        asset_id = str(asset.get("id") or "").strip()
        kind = str(asset.get("kind") or "").strip().lower()
        if not asset_id or kind not in {"role", "scene", "prop"}:
            continue
        frames: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        for raw in asset.get("keyframes") or []:
            if not isinstance(raw, dict):
                continue
            path = str(raw.get("path") or raw.get("frame_path") or "").strip()
            if not path or path in seen_paths or not Path(path).is_file():
                continue
            try:
                seconds = float(raw.get("time") or 0.0)
            except (TypeError, ValueError):
                seconds = 0.0
            seen_paths.add(path)
            frames.append({"path": path, "time": round(seconds, 3)})
        if not frames:
            continue
        groups.append({
            "group_id": asset_id,
            "kind": kind,
            "frame_paths": [item["path"] for item in frames],
            "frame_times": [item["time"] for item in frames],
        })
    return groups


def _mode2_curator_manifest_for_groups(
    groups: list[dict[str, Any]],
    understanding: dict[str, Any],
) -> dict[str, Any]:
    entries = _mode2_curator_manifest_results(understanding)
    frame_manifest = _mode2_curator_frame_manifest(understanding)
    frame_time_by_index: dict[int, float] = {}
    for fallback_index, item in enumerate(frame_manifest, 1):
        try:
            index = int(item.get("index") or fallback_index)
            frame_time_by_index[index] = float(item.get("time") or 0.0)
        except (TypeError, ValueError):
            continue

    mapped: list[dict[str, Any]] = []
    used_group_ids: set[str] = set()
    for entry in entries:
        kind = str(entry.get("kind") or "").strip().lower()
        evidence_times: list[float] = []
        for value in entry.get("evidence_times") or []:
            try:
                evidence_times.append(float(value))
            except (TypeError, ValueError):
                continue
        source_group_id = str(entry.get("group_id") or "").strip()
        candidates: list[tuple[float, dict[str, Any]]] = []
        for group in groups:
            group_id = str(group.get("group_id") or "")
            if group_id in used_group_ids:
                continue
            if kind in {"role", "scene", "prop"} and str(group.get("kind") or "") != kind:
                continue
            frame_times = [float(value) for value in (group.get("frame_times") or [])]
            nearest = min(
                (abs(left - right) for left in evidence_times for right in frame_times),
                default=999.0,
            )
            score = -nearest
            if source_group_id == group_id:
                score += 1000.0
            if evidence_times and nearest <= 2.0:
                score += 100.0
            if source_group_id != group_id and (not evidence_times or nearest > 2.0):
                continue
            candidates.append((score, group))
        if not candidates:
            continue
        group = max(candidates, key=lambda item: item[0])[1]
        group_id = str(group["group_id"])
        group_times = [float(value) for value in (group.get("frame_times") or [])]
        try:
            source_rep_index = int(entry.get("representative_frame_index"))
        except (TypeError, ValueError):
            source_rep_index = 0
        representative_time = frame_time_by_index.get(source_rep_index)
        if representative_time is None and evidence_times:
            representative_time = evidence_times[len(evidence_times) // 2]
        local_index = (
            min(range(len(group_times)), key=lambda index: abs(group_times[index] - representative_time))
            if group_times and representative_time is not None
            else 0 if group_times else None
        )
        converted = dict(entry)
        converted["source_manifest_group_id"] = source_group_id
        converted["group_id"] = group_id
        converted["representative_frame_index"] = local_index
        mapped.append(converted)
        used_group_ids.add(group_id)
    return {"results": mapped}


def _mode2_curator_needs_review(group: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "group_id": str(group.get("group_id") or ""),
        "status": "needs_review",
        "usable": False,
        "needs_review": True,
        "kind": str(group.get("kind") or "ignore"),
        "name": "",
        "confidence": 0.0,
        "representative_frame_path": "",
        "representative_frame_index": None,
        "reason": reason,
        "source": "mode2_cost_gate",
        "validation_errors": [reason],
    }


def _curate_storyboard_assets_v2(
    assets: list[dict[str, Any]],
    understanding: dict[str, Any],
    *,
    project_dir: str | Path,
    api_key: str = "",
    base_url: str = "",
    model: str = "qwen3.5-omni-flash",
    allow_model_call: bool = False,
) -> dict[str, Any]:
    groups = _mode2_curator_visual_groups(assets)
    mapped_manifest = _mode2_curator_manifest_for_groups(groups, understanding)
    mapped_ids = {
        str(item.get("group_id") or "")
        for item in mapped_manifest.get("results") or []
        if isinstance(item, dict)
    }
    call_groups = groups if allow_model_call else [
        group for group in groups if str(group.get("group_id") or "") in mapped_ids
    ]
    results_by_id = {
        str(group.get("group_id") or ""): _mode2_curator_needs_review(
            group,
            "asset_manifest_missing_or_unmapped",
        )
        for group in groups
    }
    if call_groups:
        from spvideo.mode2_asset_curator import curate_visual_groups

        curated = curate_visual_groups(
            call_groups,
            cache_dir=Path(project_dir) / "assets" / "curator",
            asset_manifest=mapped_manifest,
            story_context=understanding.get("summary") or "",
            known_roles=understanding.get("characters") or [],
            api_key=api_key if allow_model_call else "",
            base_url=base_url,
            model=model,
            allow_single_group_fallback=False,
        )
        for result in curated:
            if isinstance(result, dict) and str(result.get("group_id") or ""):
                results_by_id[str(result["group_id"])] = result

    asset_by_id = {str(asset.get("id") or ""): asset for asset in assets}
    for group_id, result in results_by_id.items():
        asset = asset_by_id.get(group_id)
        if asset is None:
            continue
        usable = bool(result.get("usable")) and str(result.get("status") or "") == "ready"
        asset["curator_status"] = str(result.get("status") or "needs_review")
        asset["curator_needs_review"] = not usable
        asset["curator_usable"] = usable
        asset["curator_confidence"] = round(float(result.get("confidence") or 0.0), 4)
        asset["curator_name"] = str(result.get("name") or "").strip()
        asset["curator_representative_image"] = str(
            result.get("representative_frame_path") or ""
        ).strip()
        asset["curator_reason"] = str(result.get("reason") or "").strip()
        asset["curator_source"] = str(result.get("source") or "").strip()
        if usable and asset["curator_name"]:
            current_name = str(asset.get("name") or "").strip()
            if not current_name or re.match(r"^(scene|prop|场景|物品)\s*[_-]?\d*$", current_name, re.I):
                asset["name"] = asset["curator_name"]
        if not usable:
            asset["asset_hidden_by_default"] = True

    statuses = [str(item.get("status") or "needs_review") for item in results_by_id.values()]
    return {
        "status": "ready" if "ready" in statuses else "needs_review",
        "group_count": len(groups),
        "manifest_result_count": len(_mode2_curator_manifest_results(understanding)),
        "mapped_manifest_count": len(mapped_ids),
        "ready_count": statuses.count("ready"),
        "needs_review_count": statuses.count("needs_review"),
        "model": model,
        "model_call_allowed": bool(allow_model_call),
        "model_call_scope_count": max(0, len(call_groups) - len(mapped_ids)),
        "single_group_fallback": False,
        "cache_dir": str(Path(project_dir) / "assets" / "curator"),
    }

def _load_or_run_storyboard_mode2_understanding(
    *,
    video_path: str,
    project_dir: str | Path,
    duration: float,
    payload: dict[str, Any],
    add_log,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    project_root = Path(project_dir)
    project_root.mkdir(parents=True, exist_ok=True)
    cache_path = _storyboard_mode2_understanding_path(project_root)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    force = bool(payload.get("force_understand"))
    use_understand = bool(payload.get("use_pre_director", True))
    if not use_understand:
        plan = {
            "version": 1,
            "status": "disabled",
            "source_path": video_path,
            "cache_path": str(cache_path),
            "generated_at": time.time(),
            "story_summary": "",
            "summary": "",
            "characters": [],
            "scenes": [],
            "boundary_hints": [],
            "sampled_frames": [],
            "sampled_frame_count": 0,
            "asset_manifest": {"results": []},
            "analysis_frame_manifest": [],
            "understanding_analysis_fresh": False,
            "note": "Mode2 pre-director disabled by request.",
        }
        return (
            _storyboard_mode2_understanding_view(
                plan,
                project_dir=project_root,
                cache_path=cache_path,
                source_path=video_path,
            ),
            [],
        )
    if not force and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                isinstance(cached, dict)
                and str(cached.get("status") or "") == "ready"
                and _storyboard_same_source_path(cached.get("source_path"), video_path)
            ):
                add_log(f"> Mode2 understanding cache hit: {cache_path}")
                return (
                    _storyboard_mode2_understanding_view(
                        cached,
                        project_dir=project_root,
                        cache_path=cache_path,
                        source_path=video_path,
                        cache_hit=True,
                    ),
                    [
                        item for item in (cached.get("sampled_frames") or [])
                        if isinstance(item, dict)
                    ],
                )
        except Exception as exc:  # noqa: BLE001
            add_log(f"> Mode2 understanding cache ignored: {exc}")
    try:
        from spvideo.pre_director import analyze_pre_director

        max_frames = int(payload.get("understand_max_frames") or 160)
        add_log("> Mode2 understanding: sampling full-video frames...")
        frame_features = _storyboard_mode2_sample_frame_features(
            video_path, project_root, duration=duration, max_frames=max_frames
        )
        sampled_frames = [item.to_dict() for item in frame_features]
        add_log(f"> Mode2 understanding: sampled {len(sampled_frames)} frames")
        api_key = str(payload.get("pre_director_api_key") or _wan_api_key())
        base_url = str(
            payload.get("pre_director_base_url")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        model = str(payload.get("pre_director_model") or "qwen3.5-omni-plus")
        plan = analyze_pre_director(
            frame_features,
            video_path=video_path,
            has_audio=_storyboard_mode2_has_audio(video_path),
            duration=duration,
            output_path=cache_path,
            api_key=api_key,
            base_url=base_url,
            model=model,
            on_progress=add_log,
            include_asset_manifest=True,
        )
        if not isinstance(plan, dict):
            plan = {"status": "failed", "error": "pre_director returned non-dict result"}
        plan["source_path"] = video_path
        plan["cache_path"] = str(cache_path)
        plan["mode2_project_dir"] = str(project_root)
        plan["summary"] = str(plan.get("summary") or plan.get("story_summary") or "")
        plan["sampled_frames"] = sampled_frames
        plan["sampled_frame_count"] = len(sampled_frames)
        plan["understanding_analysis_fresh"] = True
        plan.setdefault("asset_manifest", {"results": []})
        plan.setdefault("analysis_frame_manifest", [])
        cache_path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        add_log(f"> Mode2 understanding status: {plan.get('status')}")
        return (
            _storyboard_mode2_understanding_view(
                plan,
                project_dir=project_root,
                cache_path=cache_path,
                source_path=video_path,
            ),
            sampled_frames,
        )
    except Exception as exc:  # noqa: BLE001
        plan = {
            "version": 1,
            "status": "failed",
            "source_path": video_path,
            "cache_path": str(cache_path),
            "mode2_project_dir": str(project_root),
            "generated_at": time.time(),
            "story_summary": "",
            "summary": "",
            "characters": [],
            "scenes": [],
            "boundary_hints": [],
            "sampled_frames": [],
            "sampled_frame_count": 0,
            "asset_manifest": {"results": []},
            "analysis_frame_manifest": [],
            "understanding_analysis_fresh": False,
            "error": str(exc),
            "note": "Mode2 pre-director failed; storyboard draft will fall back to rule-based shots.",
        }
        cache_path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        add_log(f"> Mode2 understanding failed, fallback to rule draft: {exc}")
        return (
            _storyboard_mode2_understanding_view(
                plan,
                project_dir=project_root,
                cache_path=cache_path,
                source_path=video_path,
            ),
            [],
        )


def _storyboard_reference_segments_from_understanding(
    understanding: dict[str, Any],
    *,
    duration: float,
) -> list[dict[str, Any]]:
    scenes = [
        item for item in (understanding.get("scenes") or [])
        if isinstance(item, dict)
    ]
    result: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes, 1):
        try:
            start = max(0.0, min(float(scene.get("start") or 0.0), duration))
            end = max(start, min(float(scene.get("end") or duration), duration))
        except (TypeError, ValueError):
            continue
        if end - start < 0.05:
            continue
        characters = _string_list(scene.get("characters"))
        details = [
            item for item in (scene.get("character_details") or [])
            if isinstance(item, dict)
        ]
        person_count = len(characters) or len(details) or -1
        description_bits = [
            str(scene.get("description") or "").strip(),
            str(scene.get("key_action") or "").strip(),
        ]
        result.append({
            "segment_id": f"U{index:03d}",
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "person_count": person_count,
            "segment_type": "semantic_scene",
            "source": "mode2_pre_director",
            "source_video_path": understanding.get("source_path") or "",
            "description": " ".join(value for value in description_bits if value).strip(),
            "characters": characters,
            "character_details": details,
            "key_action": str(scene.get("key_action") or "").strip(),
            "semantic_scene": scene,
        })
    return result


def _storyboard_enrich_reference_segments_with_understanding(
    reference_segments: list[dict[str, Any]],
    understanding: dict[str, Any],
    *,
    duration: float,
) -> list[dict[str, Any]]:
    semantic_segments = _storyboard_reference_segments_from_understanding(understanding, duration=duration)
    if not reference_segments:
        return semantic_segments
    scenes = [
        item for item in (understanding.get("scenes") or [])
        if isinstance(item, dict)
    ]
    if not scenes:
        return [dict(item) for item in reference_segments]
    enriched: list[dict[str, Any]] = []
    for raw in reference_segments:
        item = dict(raw)
        try:
            start = float(item.get("start") or 0.0)
            end = float(item.get("end") or start)
        except (TypeError, ValueError):
            enriched.append(item)
            continue
        overlaps: list[dict[str, Any]] = []
        for scene in scenes:
            try:
                scene_start = float(scene.get("start") or 0.0)
                scene_end = float(scene.get("end") or scene_start)
            except (TypeError, ValueError):
                continue
            if min(end, scene_end) - max(start, scene_start) > 0:
                overlaps.append(scene)
        if overlaps:
            descriptions = [
                str(scene.get("description") or scene.get("key_action") or "").strip()
                for scene in overlaps
                if str(scene.get("description") or scene.get("key_action") or "").strip()
            ]
            if descriptions and not str(item.get("description") or "").strip():
                item["description"] = " ".join(descriptions[:2])
            characters: list[str] = []
            for scene in overlaps:
                for name in _string_list(scene.get("characters")):
                    if name not in characters:
                        characters.append(name)
            item["semantic_scenes"] = [
                {
                    "start": scene.get("start"),
                    "end": scene.get("end"),
                    "description": scene.get("description"),
                    "characters": scene.get("characters") or [],
                    "key_action": scene.get("key_action") or "",
                }
                for scene in overlaps[:4]
            ]
            item["semantic_characters"] = characters
            try:
                person_count = int(item.get("person_count") or -1)
            except (TypeError, ValueError):
                person_count = -1
            if person_count < 0 and characters:
                item["person_count"] = len(characters)
        enriched.append(item)
    return enriched


def _storyboard_mode2_fallback_visual_segments(
    duration: float,
    reference_frames: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    total = max(0.1, float(duration or 0.0))
    frame_boundaries: list[float] = []
    for frame in reference_frames or []:
        if not isinstance(frame, dict):
            continue
        try:
            time_value = float(frame.get("time") or 0.0)
            diff_prev = float(frame.get("diff_prev") or 0.0)
        except (TypeError, ValueError):
            continue
        if 0.8 < time_value < total - 0.8 and diff_prev >= 0.22:
            frame_boundaries.append(time_value)

    boundaries = [0.0]
    for time_value in sorted(frame_boundaries):
        if time_value - boundaries[-1] >= 0.8:
            boundaries.append(round(time_value, 3))
    if total - boundaries[-1] < 0.8 and len(boundaries) > 1:
        boundaries.pop()
    boundaries.append(round(total, 3))

    if len(boundaries) <= 2:
        target = 6.0 if total <= 60 else 8.0
        boundaries = [0.0]
        cursor = 0.0
        while cursor + target < total - 0.4:
            cursor += target
            boundaries.append(round(cursor, 3))
        boundaries.append(round(total, 3))

    segments: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), 1):
        if end <= start:
            continue
        frames: list[dict[str, Any]] = []
        for item in reference_frames or []:
            if not isinstance(item, dict):
                continue
            try:
                frame_time = float(item.get("time") or 0.0)
            except (TypeError, ValueError):
                continue
            if start <= frame_time < end:
                frames.append(item)
        person_counts: list[int] = []
        for frame in frames:
            try:
                count = int(frame.get("person_count") or -1)
            except (TypeError, ValueError):
                count = -1
            if count >= 0:
                person_counts.append(count)
        person_count = max(person_counts) if person_counts else -1
        segments.append({
            "segment_id": f"V{index:03d}",
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "person_count": person_count,
            "segment_type": "visual_shot",
            "source": "mode2_visual_fallback",
            "source_video_path": "",
            "description": "",
            "characters": [],
            "character_details": [],
            "key_action": "",
            "visual_primary": True,
        })
    return segments


def _storyboard_mode2_visual_reference_segments(
    video_path: str,
    *,
    duration: float,
    reference_frames: list[dict[str, Any]] | None,
    add_log,
) -> list[dict[str, Any]]:
    try:
        from spvideo.scene_detector import two_pass_segmentation

        add_log("> Mode2 visual timeline: PySceneDetect + YOLO first; semantics are attached later")
        result = two_pass_segmentation(
            video_path,
            sample_interval=1.0,
            min_scene_duration=0.8,
            min_sub_duration=0.8,
            yolo_conf_threshold=0.35,
            device=None,
            yolo_batch_size=None,
            use_omnishotcut=False,
            use_pyscene_detect=True,
        )
        raw_segments = [
            item for item in (result.get("sub_segments") or [])
            if isinstance(item, dict)
        ]
        segments: list[dict[str, Any]] = []
        for index, item in enumerate(raw_segments, 1):
            try:
                start = max(0.0, min(float(item.get("start") or 0.0), duration))
                end = max(start, min(float(item.get("end") or start), duration))
            except (TypeError, ValueError):
                continue
            if end - start < 0.05:
                continue
            segments.append({
                "segment_id": f"V{index:03d}",
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "person_count": int(item.get("person_count") or -1),
                "segment_type": "visual_shot",
                "source": "mode2_visual_detector",
                "source_video_path": video_path,
                "description": "",
                "characters": [],
                "character_details": [],
                "key_action": "",
                "visual_primary": True,
                "is_pure_background": bool(item.get("is_pure_background")),
                "shot_index": item.get("shot_index"),
                "start_sources": item.get("start_sources") or [],
                "end_sources": item.get("end_sources") or [],
                "representative_frame": str(item.get("representative_frame") or ""),
            })
        if segments:
            stats = result.get("stats") or {}
            add_log(
                f"> Visual timeline ready: {len(segments)} visual segments "
                f"(hard cuts={stats.get('total_hard_cuts', '?')}, people={stats.get('person_segment_count', '?')})"
            )
            return segments
        add_log("> Visual detector returned no segments; using frame-diff fallback")
    except Exception as exc:  # noqa: BLE001
        add_log(f"> Visual timeline detector failed; using frame-diff fallback: {exc}")

    segments = _storyboard_mode2_fallback_visual_segments(duration, reference_frames)
    for item in segments:
        item["source_video_path"] = video_path
    add_log(f"> Visual fallback ready: {len(segments)} visual windows")
    return segments


def _storyboard_visual_boundary_hints(reference_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    visual_segments = [
        item for item in reference_segments
        if bool(item.get("visual_primary")) or str(item.get("segment_type") or "") == "visual_shot"
    ]
    for previous, current in zip(visual_segments, visual_segments[1:]):
        try:
            boundary = float(current.get("start") or previous.get("end") or 0.0)
        except (TypeError, ValueError):
            continue
        if boundary <= 0:
            continue
        hints.append({
            "time": round(boundary, 3),
            "confidence": 0.98,
            "kind": "visual_cut",
            "reason": "画面镜头边界；模式2动作骨架优先使用此边界。",
        })
    return hints


def _storyboard_mode2_cached_sam3_identity_timeline(
    root: Path,
    *,
    duration: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read already-created Mode2 SAM3 color-mask candidates as timeline clues.

    This never calls SAM3. It only reuses cached result.json files created by the
    identity annotation panel, so it is safe when the Comfy/SAM3 server is off.
    """
    total = max(0.1, float(duration or 0.0))
    base = root / "assets" / "identity_candidates"
    if not base.exists():
        return [], []

    snapshots: list[dict[str, Any]] = []
    for result_path in sorted(base.glob("maskcand_*/result.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:  # noqa: BLE001
            logging.warning("Mode2 SAM3 identity timeline ignored: %s / %s", result_path, exc)
            continue
        if not isinstance(result, dict):
            continue
        if result.get("ok") is False or str(result.get("mask_status") or "") not in {"", "ready"}:
            continue
        try:
            time_value = float(result.get("time") or 0.0)
        except (TypeError, ValueError):
            continue
        if time_value < 0.0 or time_value > total:
            continue
        objects: list[dict[str, Any]] = []
        for obj in result.get("objects") or []:
            if not isinstance(obj, dict):
                continue
            try:
                area_ratio = float(obj.get("area_ratio") or 0.0)
                score = float(obj.get("score") or 0.0)
            except (TypeError, ValueError):
                area_ratio = score = 0.0
            if area_ratio < 0.003:
                continue
            center = obj.get("center_point") if isinstance(obj.get("center_point"), list) else []
            objects.append({
                "object_id": obj.get("object_id"),
                "color_key": str(obj.get("color_key") or ""),
                "area_ratio": round(area_ratio, 5),
                "score": round(score, 4),
                "center_point": center[:2] if len(center) >= 2 else [],
                "bbox": obj.get("bbox") or obj.get("bbox_xywh") or [],
            })
        objects.sort(key=lambda item: (
            float(item.get("center_point", [0.5])[0]) if item.get("center_point") else 0.5,
            -float(item.get("area_ratio") or 0.0),
        ))
        if not objects:
            continue
        centers = [
            float(item.get("center_point", [0.5])[0])
            for item in objects
            if isinstance(item.get("center_point"), list) and len(item.get("center_point")) >= 2
        ]
        snapshots.append({
            "time": round(time_value, 3),
            "candidate_id": str(result.get("candidate_id") or result_path.parent.name),
            "result_path": str(result_path),
            "object_count": len(objects),
            "total_area_ratio": round(sum(float(item.get("area_ratio") or 0.0) for item in objects), 5),
            "center_signature": [round(value, 3) for value in centers],
            "objects": objects[:8],
            "colored_mask": str(result.get("colored_mask") or ""),
            "overlay_image": str(result.get("overlay_image") or ""),
            "original_frame": str(result.get("original_frame") or ""),
        })

    snapshots.sort(key=lambda item: float(item.get("time") or 0.0))
    deduped: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in snapshots:
        key = int(round(float(item.get("time") or 0.0) * 10))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    hints: list[dict[str, Any]] = []
    for previous, current in zip(deduped, deduped[1:]):
        try:
            prev_time = float(previous.get("time") or 0.0)
            cur_time = float(current.get("time") or 0.0)
            prev_count = int(previous.get("object_count") or 0)
            cur_count = int(current.get("object_count") or 0)
            prev_area = float(previous.get("total_area_ratio") or 0.0)
            cur_area = float(current.get("total_area_ratio") or 0.0)
        except (TypeError, ValueError):
            continue
        if cur_time - prev_time < 0.45:
            continue
        prev_centers = [float(value) for value in previous.get("center_signature") or []]
        cur_centers = [float(value) for value in current.get("center_signature") or []]
        center_shift = 0.0
        if prev_centers and cur_centers and len(prev_centers) == len(cur_centers):
            center_shift = max(abs(a - b) for a, b in zip(prev_centers, cur_centers))
        area_delta = abs(cur_area - prev_area)

        kind = ""
        confidence = 0.0
        reason = ""
        if prev_count != cur_count:
            kind = "sam3_person_count_change"
            confidence = 0.86
            reason = f"cached SAM3 person objects changed {prev_count}->{cur_count}"
        elif center_shift >= 0.22:
            kind = "sam3_layout_shift"
            confidence = 0.78
            reason = f"cached SAM3 person layout shifted {center_shift:.2f}"
        elif area_delta >= 0.12:
            kind = "sam3_person_scale_change"
            confidence = 0.74
            reason = f"cached SAM3 person mask area changed {area_delta:.2f}"
        if not kind:
            continue
        hints.append({
            "time": round(cur_time, 3),
            "confidence": confidence,
            "kind": kind,
            "reason": reason,
            "source": "cached_sam3_identity_candidates",
            "previous_candidate_id": previous.get("candidate_id"),
            "candidate_id": current.get("candidate_id"),
        })

    merged_hints: list[dict[str, Any]] = []
    for hint in hints:
        hint_time = float(hint.get("time") or 0.0)
        if not (0.4 < hint_time < total - 0.4):
            continue
        if merged_hints and hint_time - float(merged_hints[-1].get("time") or 0.0) < 0.8:
            if float(hint.get("confidence") or 0.0) > float(merged_hints[-1].get("confidence") or 0.0):
                merged_hints[-1] = hint
            continue
        merged_hints.append(hint)
    return deduped, merged_hints


def _storyboard_refine_visual_segments_with_hints(
    visual_segments: list[dict[str, Any]],
    understanding: dict[str, Any],
    reference_frames: list[dict[str, Any]] | None,
    *,
    duration: float,
    extra_boundary_hints: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    total = max(0.1, float(duration or 0.0))
    raw_boundaries: list[dict[str, Any]] = [
        {"time": 0.0, "source": "video_start", "priority": 9},
        {"time": total, "source": "video_end", "priority": 9},
    ]

    for segment in visual_segments:
        if not isinstance(segment, dict):
            continue
        for key in ("start", "end"):
            try:
                time_value = float(segment.get(key) or 0.0)
            except (TypeError, ValueError):
                continue
            if 0.0 <= time_value <= total:
                raw_boundaries.append({
                    "time": round(time_value, 3),
                    "source": "visual_detector",
                    "priority": 8,
                })

    for hint in understanding.get("boundary_hints") or []:
        if not isinstance(hint, dict):
            continue
        try:
            time_value = float(hint.get("time") or 0.0)
            confidence = float(hint.get("confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        if 0.4 < time_value < total - 0.4 and confidence >= 0.72:
            raw_boundaries.append({
                "time": round(time_value, 3),
                "source": "semantic_hint",
                "priority": 6,
                "reason": str(hint.get("reason") or ""),
                "kind": str(hint.get("kind") or ""),
            })

    for hint in extra_boundary_hints or []:
        if not isinstance(hint, dict):
            continue
        try:
            time_value = float(hint.get("time") or 0.0)
            confidence = float(hint.get("confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        if 0.4 < time_value < total - 0.4 and confidence >= 0.70:
            raw_boundaries.append({
                "time": round(time_value, 3),
                "source": str(hint.get("source") or "cached_sam3_identity_candidates"),
                "priority": 7,
                "reason": str(hint.get("reason") or ""),
                "kind": str(hint.get("kind") or "sam3_identity_change"),
            })

    for frame in reference_frames or []:
        if not isinstance(frame, dict):
            continue
        try:
            time_value = float(frame.get("time") or 0.0)
            diff_prev = float(frame.get("diff_prev") or 0.0)
        except (TypeError, ValueError):
            continue
        if 0.5 < time_value < total - 0.5 and diff_prev >= 0.28:
            raw_boundaries.append({
                "time": round(time_value, 3),
                "source": "frame_diff",
                "priority": 5,
                "reason": f"frame diff={diff_prev:.3f}",
            })

    raw_boundaries.sort(key=lambda item: (float(item.get("time") or 0.0), -int(item.get("priority") or 0)))
    merged: list[dict[str, Any]] = []
    tolerance = 0.35
    for boundary in raw_boundaries:
        time_value = float(boundary.get("time") or 0.0)
        if not merged:
            merged.append(boundary)
            continue
        previous = merged[-1]
        previous_time = float(previous.get("time") or 0.0)
        if abs(time_value - previous_time) <= tolerance:
            previous_sources = {
                str(value)
                for value in (
                    previous.get("sources")
                    if isinstance(previous.get("sources"), list)
                    else [previous.get("source")]
                )
                if str(value)
            }
            previous_sources.add(str(boundary.get("source") or ""))
            if int(boundary.get("priority") or 0) > int(previous.get("priority") or 0):
                boundary["sources"] = sorted(previous_sources)
                merged[-1] = boundary
            else:
                previous["sources"] = sorted(previous_sources)
            continue
        merged.append(boundary)

    boundaries: list[dict[str, Any]] = []
    for boundary in merged:
        time_value = max(0.0, min(total, float(boundary.get("time") or 0.0)))
        if not boundaries:
            item = dict(boundary)
            item["time"] = round(time_value, 3)
            boundaries.append(item)
            continue
        previous_time = float(boundaries[-1].get("time") or 0.0)
        if time_value - previous_time < 0.65 and 0.0 < time_value < total:
            if int(boundary.get("priority") or 0) > int(boundaries[-1].get("priority") or 0):
                item = dict(boundary)
                item["time"] = round(time_value, 3)
                boundaries[-1] = item
            continue
        item = dict(boundary)
        item["time"] = round(time_value, 3)
        boundaries.append(item)

    if len(boundaries) < 2:
        return visual_segments
    if float(boundaries[-1].get("time") or 0.0) < total:
        boundaries.append({"time": round(total, 3), "source": "video_end", "priority": 9})

    original_segments = [
        item for item in visual_segments
        if isinstance(item, dict)
    ]
    refined: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(boundaries, boundaries[1:]), 1):
        start = float(left.get("time") or 0.0)
        end = float(right.get("time") or start)
        if end - start < 0.05:
            continue
        overlapping_visual: list[dict[str, Any]] = []
        for item in original_segments:
            try:
                item_start = float(item.get("start") or 0.0)
                item_end = float(item.get("end") or item_start)
            except (TypeError, ValueError):
                continue
            if item_end > start and item_start < end:
                overlapping_visual.append(item)
        frame_hits: list[dict[str, Any]] = []
        for item in reference_frames or []:
            if not isinstance(item, dict):
                continue
            try:
                frame_time = float(item.get("time") or 0.0)
            except (TypeError, ValueError):
                continue
            if start <= frame_time < end:
                frame_hits.append(item)
        person_counts: list[int] = []
        for item in overlapping_visual:
            try:
                count = int(item.get("person_count") or -1)
            except (TypeError, ValueError):
                count = -1
            if count >= 0:
                person_counts.append(count)
        for frame in frame_hits:
            try:
                count = int(frame.get("person_count") or -1)
            except (TypeError, ValueError):
                count = -1
            if count >= 0:
                person_counts.append(count)
        left_sources = left.get("sources") if isinstance(left.get("sources"), list) else [left.get("source")]
        right_sources = right.get("sources") if isinstance(right.get("sources"), list) else [right.get("source")]
        refined.append({
            "segment_id": f"V{index:03d}",
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "person_count": max(person_counts) if person_counts else -1,
            "segment_type": "visual_shot",
            "source": "mode2_visual_semantic_fused",
            "source_video_path": str((overlapping_visual[0].get("source_video_path") if overlapping_visual else "") or ""),
            "description": "",
            "characters": [],
            "character_details": [],
            "key_action": "",
            "visual_primary": True,
            "visual_source_segment_ids": [
                str(item.get("segment_id") or "")
                for item in overlapping_visual
                if str(item.get("segment_id") or "")
            ],
            "cut_sources": sorted({str(value) for value in [*left_sources, *right_sources] if str(value)}),
            "cut_reasons": [
                str(value or "")
                for value in (left.get("reason"), right.get("reason"))
                if str(value or "").strip()
            ][:3],
        })
    return refined or visual_segments


def _storyboard_character_segment_ids(
    character: dict[str, Any],
    reference_segments: list[dict[str, Any]],
) -> list[str]:
    ranges = [
        value for value in (character.get("time_ranges") or [])
        if isinstance(value, (list, tuple)) and len(value) >= 2
    ]
    visual_label = str(character.get("visual_label") or "").strip()
    result: list[str] = []
    for segment in reference_segments:
        segment_id = str(segment.get("segment_id") or "").strip()
        if not segment_id:
            continue
        include = False
        if ranges:
            try:
                start = float(segment.get("start") or 0.0)
                end = float(segment.get("end") or start)
            except (TypeError, ValueError):
                start = end = 0.0
            include = any(
                min(end, float(item[1] or 0.0)) - max(start, float(item[0] or 0.0)) > 0
                for item in ranges
            )
        if not include and visual_label:
            include = visual_label in _string_list(segment.get("characters")) or visual_label in _string_list(segment.get("semantic_characters"))
        if include and segment_id not in result:
            result.append(segment_id)
    return result


def _storyboard_auto_director_with_understanding(
    auto_director_plan: dict[str, Any] | None,
    understanding: dict[str, Any],
    reference_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    result = dict(auto_director_plan or {})
    story = dict(result.get("story") or {})
    existing_characters = [
        item for item in (story.get("characters") or [])
        if isinstance(item, dict)
    ]
    understood_characters: list[dict[str, Any]] = []
    for index, character in enumerate(understanding.get("characters") or [], 1):
        if not isinstance(character, dict):
            continue
        visual_label = str(character.get("visual_label") or "").strip() or f"role_{index}"
        role_candidates = _string_list(character.get("role_candidates"))
        role_name = str(character.get("role_name") or "").strip() or (role_candidates[0] if role_candidates else visual_label)
        understood_characters.append({
            "role_name": role_name,
            "visual_label": visual_label,
            "role_candidates": role_candidates,
            "relationships": _string_list(character.get("relationships")),
            "description": str(character.get("description") or "").strip(),
            "confidence": character.get("confidence", 0.0),
            "time_ranges": character.get("time_ranges") or [],
            "segment_ids": _storyboard_character_segment_ids(character, reference_segments),
            "source": "mode2_pre_director",
        })
    if understood_characters and not existing_characters:
        story["characters"] = understood_characters
    elif understood_characters:
        story["mode2_understood_characters"] = understood_characters
    if understanding.get("summary"):
        story.setdefault("summary", understanding.get("summary"))
    if understanding.get("scenes"):
        story["mode2_understood_scenes"] = understanding.get("scenes")
    result["story"] = story
    result["mode2_understanding_status"] = understanding.get("status")
    result["mode2_understanding_cache_path"] = understanding.get("cache_path")
    return result


def _storyboard_boundary_hints_for_window(
    boundary_hints: list[dict[str, Any]],
    *,
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for hint in boundary_hints:
        if not isinstance(hint, dict):
            continue
        try:
            time_value = float(hint.get("time") or 0.0)
        except (TypeError, ValueError):
            continue
        if start < time_value < end:
            result.append({
                "time": round(time_value, 3),
                "confidence": hint.get("confidence"),
                "kind": hint.get("kind"),
                "reason": hint.get("reason"),
            })
    return result


def _storyboard_next_seedance_end(
    *,
    start: float,
    duration: float,
    target_seconds: float,
    boundary_hints: list[dict[str, Any]],
) -> float:
    minimum = 4.0
    maximum = 15.0
    if duration - start <= maximum:
        return duration
    desired = min(duration, start + target_seconds)
    candidates: list[tuple[float, float]] = []
    for hint in boundary_hints:
        if not isinstance(hint, dict):
            continue
        try:
            time_value = float(hint.get("time") or 0.0)
            confidence = float(hint.get("confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        if start + minimum <= time_value <= min(duration, start + maximum):
            candidates.append((abs(time_value - desired) - confidence * 0.25, time_value))
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    return min(duration, start + target_seconds)


def _storyboard_mode2_cached_visual_timeline(
    project_dir: str | Path,
    video_path: str,
) -> dict[str, list[dict[str, Any]]] | None:
    store_path = _storyboard_mode2_asset_store_path(Path(project_dir))
    if not store_path.exists():
        return None
    try:
        cached = json.loads(store_path.read_text(encoding="utf-8-sig"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(cached, dict):
        return None
    if not _storyboard_same_source_path(cached.get("video_path"), video_path):
        return None

    timeline: dict[str, list[dict[str, Any]]] = {}
    for key in ("raw_visual_segments", "visual_segments", "reference_segments"):
        segments = [
            dict(item)
            for item in (cached.get(key) or [])
            if isinstance(item, dict)
        ]
        if not segments:
            return None
        timeline[key] = segments
    return timeline


def _storyboard_segment_boundary_signature(segments: list[dict[str, Any]]) -> tuple[tuple[float, float], ...]:
    signature: list[tuple[float, float]] = []
    for item in segments:
        if not isinstance(item, dict):
            continue
        try:
            start = round(float(item.get("start") or 0.0), 3)
            end = round(float(item.get("end") or start), 3)
        except (TypeError, ValueError):
            continue
        signature.append((start, end))
    return tuple(signature)


def _run_storyboard_draft_job_v2(job_id: str, payload: dict[str, Any]) -> None:
    def add_log(message: str) -> None:
        snapshot = None
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job.setdefault("logs", []).append(message)
                snapshot = dict(job)
        if snapshot is not None:
            _write_storyboard_job_snapshot(snapshot)

    try:
        video_path = str(payload.get("video_path") or "").strip()
        requested_project_dir = str(payload.get("project_dir") or "").strip()
        reference_strategy = _normalize_storyboard_reference_strategy(payload.get("reference_strategy"))
        project_config = _normalize_storyboard_project_config(payload.get("project_config"))
        if (
            reference_strategy != "new_only"
            and bool(payload.get("use_pre_director", True))
            and not bool(payload.get("allow_paid_understanding_with_old_reference"))
        ):
            payload["use_pre_director"] = False
            payload["force_understand"] = False
            add_log("> Cost guard: old project reference mode disables full-video understanding by default")
        if bool(payload.get("use_pre_director", True)):
            add_log("> Mode2 storyboard draft: understand full video first, then build shots")
        else:
            add_log("> Mode2 storyboard draft: use cached/reference data and visual rules; no full-video model call")
        add_log(f"> Reference strategy: {_storyboard_reference_strategy_label(reference_strategy)}")
        add_log(f"> Project config: {_storyboard_project_config_summary(project_config)}")

        reference_result: dict[str, Any] = {}
        reference_project_dir = ""
        if reference_strategy == "new_only":
            if requested_project_dir:
                add_log(f"> Ignoring old project for new_only mode: {requested_project_dir}")
        elif not requested_project_dir:
            add_log("> No old project selected; using Mode2 standalone cache")
            reference_strategy = "new_only"
        else:
            if reference_strategy == "hybrid_reference":
                add_log("> Hybrid reference: old project is story/role hint only; Mode2 keeps its own cache")
            try:
                reference_result = _load_project_result(requested_project_dir)
                reference_project_dir = str(reference_result.get("project_dir") or requested_project_dir)
                add_log(f"> Loaded old project as reference only: {reference_project_dir}")
                if not video_path:
                    video_path = str((reference_result.get("meta") or {}).get("source_path") or "")
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"old project reference load failed: {exc}") from exc

        if not video_path:
            raise ValueError("missing source video path")
        if not Path(video_path).exists():
            raise ValueError(f"source video does not exist: {video_path}")

        duration = _video_duration(video_path)
        if duration <= 0:
            duration = max(
                15.0,
                max((float(item.get("end") or 0.0) for item in reference_result.get("segments", [])), default=0.0),
            )
        add_log(f"> Source duration: {duration:.2f}s")

        mode2_project_dir = _storyboard_mode2_project_dir(video_path)
        project_dir = str(mode2_project_dir)
        mode2_project_dir.mkdir(parents=True, exist_ok=True)
        add_log(f"> Mode2 project cache: {project_dir}")

        understanding, sampled_reference_frames = _load_or_run_storyboard_mode2_understanding(
            video_path=video_path,
            project_dir=mode2_project_dir,
            duration=duration,
            payload=payload,
            add_log=add_log,
        )
        understanding_status = str(understanding.get("status") or "unknown")
        if understanding_status == "ready":
            add_log(
                f"> Understanding ready: {len(understanding.get('characters') or [])} characters, "
                f"{len(understanding.get('scenes') or [])} scenes"
            )
        else:
            add_log(f"> Understanding fallback status: {understanding_status}")

        legacy_reference_segments = [
            item for item in (reference_result.get("segments") or [])
            if isinstance(item, dict)
        ]
        reference_frames: list[dict[str, Any]] = []
        legacy_reference_frames = [
            item for item in (reference_result.get("frames") or [])
            if isinstance(item, dict)
        ]
        legacy_source_path = str((reference_result.get("meta") or {}).get("source_path") or "")
        if legacy_reference_frames and _same_video_path(legacy_source_path, video_path):
            reference_frames.extend(legacy_reference_frames)
            add_log(f"> Reusing old frame probes because source video matches: {len(legacy_reference_frames)} frames")
        elif legacy_reference_frames:
            add_log("> Ignoring old frame probes because old project source video differs from current video")
        if sampled_reference_frames:
            reference_frames.extend(sampled_reference_frames)
        sam3_identity_snapshots, sam3_boundary_hints = _storyboard_mode2_cached_sam3_identity_timeline(
            mode2_project_dir,
            duration=duration,
        )
        if sam3_identity_snapshots:
            add_log(
                f"> Cached SAM3 identity clues: {len(sam3_identity_snapshots)} frames, "
                f"{len(sam3_boundary_hints)} boundary hints; SAM3 server not called"
            )
        force_visual_reanalysis = bool(payload.get("force_visual_reanalysis"))
        cached_visual_timeline = (
            None
            if force_visual_reanalysis
            else _storyboard_mode2_cached_visual_timeline(mode2_project_dir, video_path)
        )
        if cached_visual_timeline:
            raw_visual_segments = cached_visual_timeline["raw_visual_segments"]
            visual_segments = cached_visual_timeline["visual_segments"]
            if sam3_boundary_hints:
                sam3_refined_visual_segments = _storyboard_refine_visual_segments_with_hints(
                    raw_visual_segments,
                    understanding,
                    reference_frames,
                    duration=duration,
                    extra_boundary_hints=sam3_boundary_hints,
                )
                if (
                    _storyboard_segment_boundary_signature(sam3_refined_visual_segments)
                    != _storyboard_segment_boundary_signature(visual_segments)
                ):
                    visual_segments = sam3_refined_visual_segments
                    reference_segments = _storyboard_enrich_reference_segments_with_understanding(
                        visual_segments,
                        understanding,
                        duration=duration,
                    )
                    add_log(
                        f"> Cached visual timeline refreshed by cached SAM3 identity clues: "
                        f"{len(cached_visual_timeline['visual_segments'])} -> {len(visual_segments)} segments"
                    )
                else:
                    reference_segments = cached_visual_timeline["reference_segments"]
            else:
                reference_segments = cached_visual_timeline["reference_segments"]
            add_log(
                f"> Reusing cached Mode2 visual timeline: {len(visual_segments)} segments; "
                "PySceneDetect/YOLO skipped"
            )
        else:
            if force_visual_reanalysis:
                add_log("> Forced Mode2 visual reanalysis requested; ignoring cached visual timeline")
            raw_visual_segments = _storyboard_mode2_visual_reference_segments(
                video_path,
                duration=duration,
                reference_frames=reference_frames,
                add_log=add_log,
            )
            visual_segments = _storyboard_refine_visual_segments_with_hints(
                raw_visual_segments,
                understanding,
                reference_frames,
                duration=duration,
                extra_boundary_hints=sam3_boundary_hints,
            )
            if len(visual_segments) != len(raw_visual_segments):
                add_log(
                    f"> Visual timeline refined by image/semantic clues: "
                    f"{len(raw_visual_segments)} -> {len(visual_segments)} segments"
                )
            reference_segments = _storyboard_enrich_reference_segments_with_understanding(
                visual_segments,
                understanding,
                duration=duration,
            )
        if legacy_reference_segments:
            add_log(
                f"> Old project segments kept as soft reference only: {len(legacy_reference_segments)}; "
                f"Mode2 timeline uses {len(visual_segments)} visual segments"
            )
        auto_director_plan = (
            reference_result.get("auto_director")
            if isinstance(reference_result.get("auto_director"), dict)
            else {}
        )
        if not auto_director_plan and reference_project_dir:
            auto_director_plan = load_auto_director(reference_project_dir)
        auto_director_plan = _storyboard_auto_director_with_understanding(
            auto_director_plan,
            understanding,
            reference_segments,
        )

        target_seconds = _storyboard_seedance_target_seconds(duration, reference_segments)
        semantic_boundary_hints = [
            item for item in (understanding.get("boundary_hints") or [])
            if isinstance(item, dict)
        ]
        visual_boundary_hints = _storyboard_visual_boundary_hints(reference_segments)
        boundary_hints = visual_boundary_hints + sam3_boundary_hints
        shot_metadata_boundary_hints = visual_boundary_hints + sam3_boundary_hints + semantic_boundary_hints
        min_tail_seconds = 4.0
        max_shots = 160
        shots: list[dict[str, Any]] = []
        start = 0.0
        index = 1
        while start < duration - 0.05 and index <= max_shots:
            end = _storyboard_next_seedance_end(
                start=start,
                duration=duration,
                target_seconds=target_seconds,
                boundary_hints=boundary_hints,
            )
            if duration - end < min_tail_seconds and duration - end > 0:
                end = duration
            if end <= start + 0.05:
                break
            overlaps: list[dict[str, Any]] = []
            for item in reference_segments:
                try:
                    item_start = float(item.get("start") or 0.0)
                    item_end = float(item.get("end") or item_start)
                except (TypeError, ValueError):
                    continue
                if item_end > start and item_start < end:
                    overlaps.append(item)
            descriptions = [
                str(item.get("description") or "").strip()
                for item in overlaps
                if str(item.get("description") or "").strip()
            ]
            people = [
                _safe_int(item.get("person_count"), -1)
                for item in overlaps
                if _safe_int(item.get("person_count"), -1) >= 0
            ]
            description = (
                " ".join(descriptions[:2])
                if descriptions
                else "Model should infer this 4-15s shot action, emotion, and spatial relation from the source video."
            )
            person_count = max(people) if people else -1
            semantic_scenes: list[dict[str, Any]] = []
            for item in overlaps:
                for scene in item.get("semantic_scenes") or []:
                    if isinstance(scene, dict):
                        semantic_scenes.append(scene)
                if isinstance(item.get("semantic_scene"), dict):
                    semantic_scenes.append(item["semantic_scene"])
            shots.append({
                "segment_id": f"S{index:03d}",
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "source_segment_ids": [
                    str(item.get("segment_id") or "").strip()
                    for item in overlaps
                    if str(item.get("segment_id") or "").strip()
                ],
                "person_count": person_count,
                "segment_type": "storyboard_draft",
                "source_video_path": video_path,
                "output_path": video_path,
                "description": description,
                "prompt": "",
                "status": "draft",
                "understanding_status": understanding_status,
                "semantic_scenes": semantic_scenes[:4],
                "boundary_hints": _storyboard_boundary_hints_for_window(
                    shot_metadata_boundary_hints,
                    start=start,
                    end=end,
                ),
            })
            start = end
            index += 1

        assets = _build_storyboard_assets_v2(
            video_path,
            reference_segments,
            auto_director_plan,
            understanding=understanding,
        )
        _attach_storyboard_asset_usage(assets, shots)
        _attach_storyboard_asset_source_images(
            video_path,
            assets,
            shots,
            reference_segments=reference_segments,
            reference_frames=reference_frames,
        )
        curator_allow_model_call = bool(
            payload.get("allow_paid_asset_curator")
            and understanding.get("understanding_analysis_fresh")
            and understanding_status == "ready"
        )
        asset_curator = _curate_storyboard_assets_v2(
            assets,
            understanding,
            project_dir=mode2_project_dir,
            api_key=(str(payload.get("pre_director_api_key") or _wan_api_key())
                     if curator_allow_model_call else ""),
            base_url=str(
                payload.get("pre_director_base_url")
                or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            model=str(payload.get("asset_curator_model") or "qwen3.5-omni-flash"),
            allow_model_call=curator_allow_model_call,
        )
        add_log(
            f"> Mode2 asset curator: groups={asset_curator.get('group_count', 0)} "
            f"manifest={asset_curator.get('mapped_manifest_count', 0)} "
            f"ready={asset_curator.get('ready_count', 0)}"
        )
        _flag_storyboard_role_source_collisions(assets)
        _flag_storyboard_asset_source_quality(assets)
        _mark_storyboard_mode2_candidate_asset_stage(assets)
        _compile_storyboard_prompts(assets, shots, project_config=project_config)
        add_log(f"> Built asset/shot index: {len(assets)} assets")
        add_log(f"> Compiled Seedance-ready shot prompts around {target_seconds:.0f}s windows")

        storyboard_asset_store_path = Path(project_dir) / "assets" / "storyboard_assets.json"
        storyboard_asset_store = {
            "version": 2,
            "job_id": job_id,
            "project_dir": project_dir,
            "reference_project_dir": reference_project_dir,
            "video_path": video_path,
            "reference_strategy": reference_strategy,
            "project_config": project_config,
            "created_at": time.time(),
            "understanding": understanding,
            "asset_manifest": understanding.get("asset_manifest") or {"results": []},
            "analysis_frame_manifest": understanding.get("analysis_frame_manifest") or [],
            "asset_curator": asset_curator,
            "raw_visual_segments": raw_visual_segments,
            "visual_segments": visual_segments,
            "sam3_identity_snapshots": sam3_identity_snapshots,
            "sam3_boundary_hints": sam3_boundary_hints,
            "semantic_segments": _storyboard_reference_segments_from_understanding(understanding, duration=duration),
            "legacy_reference_segments": legacy_reference_segments,
            "reference_segments": reference_segments,
            "auto_director": auto_director_plan,
            "assets": assets,
            "shots": shots,
            "asset_stage": "candidate_after_timeline",
            "timeline_stage": "visual_timeline_built",
            "storyboard_stage": "draft_shots_built",
            "mode_boundary": "Mode2 only; no Mode1 transfer/render dependency.",
        }
        _refresh_mode2_structured_fields(mode2_project_dir, storyboard_asset_store)
        storyboard_asset_store_path.parent.mkdir(parents=True, exist_ok=True)
        storyboard_asset_store_path.write_text(
            json.dumps(storyboard_asset_store, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        asset_audit: dict[str, Any] = {}
        try:
            audited_result = _audit_storyboard_mode2_assets({"project_dir": project_dir})
            assets = [
                item for item in (audited_result.get("assets") or [])
                if isinstance(item, dict)
            ]
            shots = [
                item for item in (audited_result.get("segments") or [])
                if isinstance(item, dict)
            ]
            asset_audit = audited_result.get("asset_audit") if isinstance(audited_result.get("asset_audit"), dict) else {}
            add_log(
                f"> Asset preflight complete: blocked={asset_audit.get('blocked', 0)} "
                f"warning={asset_audit.get('warning', 0)} ok={asset_audit.get('ok', 0)}"
            )
        except Exception as exc:  # noqa: BLE001
            add_log(f"> Asset preflight skipped: {exc}")

        summary = (
            "Mode2 storyboard draft is ready. It first tries full-video understanding for "
            "characters, scenes and semantic boundaries, then falls back to rule-based "
            "Seedance 4-15s windows if understanding is unavailable. "
            f"understanding_status={understanding_status}; "
            f"reference_strategy={_storyboard_reference_strategy_label(reference_strategy)}; "
            f"project_config={_storyboard_project_config_summary(project_config)}."
        )
        result = {
            "job_id": job_id,
            "project_dir": project_dir,
            "reference_project_dir": reference_project_dir,
            "video_path": video_path,
            "reference_strategy": reference_strategy,
            "project_config": project_config,
            "segment_count": len(shots),
            "segments": shots,
            "assets": assets,
            "asset_curator": asset_curator,
            "asset_audit": asset_audit,
            "reference_segments": reference_segments,
            "raw_visual_segments": raw_visual_segments,
            "visual_segments": visual_segments,
            "semantic_segments": _storyboard_reference_segments_from_understanding(understanding, duration=duration),
            "auto_director": auto_director_plan,
            "asset_stage": "candidate_after_timeline",
            "timeline_stage": "visual_timeline_built",
            "storyboard_stage": "draft_shots_built",
            "mode_boundary": "Mode2 only; no Mode1 transfer/render dependency.",
            "meta": {
                "source_path": video_path,
                "project_config": project_config,
                "mode2_project_dir": project_dir,
                "reference_project_dir": reference_project_dir,
            },
            "storyboard": {
                "status": "draft",
                "source": "storyboard_draft",
                "summary": summary,
                "reference_strategy": reference_strategy,
                "project_config": project_config,
                "understanding_status": understanding_status,
                "understanding": understanding,
                "sam3_identity_snapshots": sam3_identity_snapshots,
                "sam3_boundary_hints": sam3_boundary_hints,
                "cut_rule": (
                    "Mode2 uses visual shot/person timeline as the motion skeleton, groups it into "
                    "Seedance 4-15s windows, optionally reuses cached SAM3 identity-mask clues to refine "
                    "person/layout boundaries, then attaches full-video semantic understanding as labels. "
                    "Semantics and SAM3 candidates are clues; cached SAM3 is never run during draft."
                ),
                "asset_stage": "candidate_after_timeline",
                "seedance_window_seconds": {
                    "min": 4.0,
                    "target": target_seconds,
                    "max": 15.0,
                },
                "asset_store_path": str(storyboard_asset_store_path),
            },
        }
        with JOBS_LOCK:
            job = JOBS[job_id]
            job["status"] = "done"
            job["result"] = result
            job["finished_at"] = time.time()
            job.setdefault("logs", []).append(f"> storyboard draft done: {len(shots)} shots")
            snapshot = dict(job)
        _write_storyboard_job_snapshot(snapshot)
    except Exception as exc:  # noqa: BLE001
        with JOBS_LOCK:
            job = JOBS[job_id]
            job["status"] = "failed"
            job["error"] = str(exc)
            job["finished_at"] = time.time()
            job.setdefault("logs", []).append(f"> storyboard draft failed: {exc}")
            snapshot = dict(job)
        _write_storyboard_job_snapshot(snapshot)


def _run_storyboard_draft_job(job_id: str, payload: dict[str, Any]) -> None:
    return _run_storyboard_draft_job_v2(job_id, payload)


def _storyboard_mode2_role_track_dir(root: Path, role_id: str, job_id: str) -> Path:
    safe_role = _safe_output_name(role_id or "role")
    return root / "assets" / "role_tracks" / safe_role / job_id[:12]


def _storyboard_mode2_track_quality(
    track_dir: Path,
    *,
    prompt_frame: int,
    point: list[float],
    source_shape: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    mask_paths = sorted(track_dir.glob("mask_*.png"))
    if not mask_paths:
        return {"coverage": 0.0, "mean_area_ratio": 0.0, "warnings": ["no_masks"]}
    area_ratios: list[float] = []
    point_covered = False
    shape_overlap: float | None = None
    for path in mask_paths:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        height, width = mask.shape[:2]
        selected = mask > 127
        area_ratios.append(float(np.count_nonzero(selected)) / float(max(1, width * height)))
        if path.name == f"mask_{prompt_frame:04d}.png":
            px = max(0, min(width - 1, int(round(float(point[0]) * width))))
            py = max(0, min(height - 1, int(round(float(point[1]) * height))))
            radius = max(2, min(width, height) // 100)
            neighborhood = mask[max(0, py - radius): py + radius + 1, max(0, px - radius): px + radius + 1]
            point_covered = bool(neighborhood.size and np.any(neighborhood > 127))
            shape_overlap = _identity_shape_mask_overlap(mask, source_shape)
    warnings: list[str] = []
    mean_area = sum(area_ratios) / len(area_ratios) if area_ratios else 0.0
    if mean_area < 0.01:
        warnings.append("mask_too_small")
    if mean_area > 0.65:
        warnings.append("mask_too_large")
    if not point_covered:
        warnings.append("point_not_covered")
    if shape_overlap is not None and shape_overlap < SAM3_SHAPE_MIN_OVERLAP:
        warnings.append("shape_seed_not_covered")
    return {
        "coverage": round(len(area_ratios) / max(1, len(mask_paths)), 4),
        "mean_area_ratio": round(mean_area, 4),
        "min_area_ratio": round(min(area_ratios), 4) if area_ratios else 0.0,
        "max_area_ratio": round(max(area_ratios), 4) if area_ratios else 0.0,
        "prompt_point_covered": point_covered,
        "source_shape_overlap": round(shape_overlap, 4) if shape_overlap is not None else None,
        "warnings": warnings,
    }


def _storyboard_mode2_extract_role_source_bundle(
    *,
    video_path: str,
    role: dict[str, Any],
    track_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    import cv2
    import numpy as np

    mask_paths = sorted(track_dir.glob("mask_*.png"))
    if not mask_paths:
        raise ValueError("role_track_masks_missing")
    picked_masks = _evenly_pick(mask_paths, min(4, len(mask_paths)))
    role_id = str(role.get("id") or "role")
    crop_dir = track_dir / "source_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    for stale in crop_dir.glob("*.jpg"):
        stale.unlink(missing_ok=True)

    clip_start_time = float(summary.get("clip_start_time") or 0.0)
    track_fps = float(summary.get("track_fps") or 0.0) or 15.0
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"cannot_open_video_for_role_bundle: {video_path}")
    crop_paths: list[Path] = []
    source_mask_paths: list[Path] = []
    try:
        for mask_path in picked_masks:
            try:
                mask_index = int(mask_path.stem.rsplit("_", 1)[-1])
            except (TypeError, ValueError):
                continue
            time_seconds = max(0.0, clip_start_time + mask_index / track_fps)
            capture.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            height, width = frame.shape[:2]
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            selected = mask > 127
            if not np.any(selected):
                continue
            ys, xs = np.where(selected)
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            box_w = x2 - x1
            box_h = y2 - y1
            pad_x = max(16, int(box_w * 0.35))
            pad_y = max(20, int(box_h * 0.24))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(width, x2 + pad_x)
            y2 = min(height, y2 + pad_y)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            crop_path = crop_dir / f"{role_id}_{int(round(time_seconds * 1000)):08d}ms.jpg"
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if crop_path.exists() and crop_path.stat().st_size > 0:
                crop_paths.append(crop_path)
                source_mask_paths.append(mask_path)
    finally:
        capture.release()

    if not crop_paths:
        raise ValueError("role_track_source_crops_empty")
    sheet_path = track_dir / f"{role_id}_identity_sheet.jpg"
    if not _make_storyboard_asset_contact_sheet(crop_paths, sheet_path):
        raise ValueError("role_track_identity_sheet_failed")
    return {
        "source_image": str(sheet_path),
        "source_images": [str(path) for path in crop_paths],
        "source_crop_paths": [str(path) for path in crop_paths],
        "source_mask_paths": [str(path) for path in source_mask_paths],
        "source_frame_count": len(crop_paths),
    }


def _storyboard_mode2_attach_role_presence_to_shots(
    role: dict[str, Any],
    shots: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    role_id = str(role.get("id") or "")
    if not role_id:
        return
    start = float(summary.get("clip_start_time") or 0.0)
    track_fps = float(summary.get("track_fps") or 0.0) or 15.0
    duration = float(summary.get("num_frames") or 0) / track_fps if track_fps > 0 else 0.0
    end = start + max(0.0, duration)
    present_shots: list[str] = []
    for shot in shots:
        try:
            shot_start = float(shot.get("start") or 0.0)
            shot_end = float(shot.get("end") or shot_start)
        except (TypeError, ValueError):
            continue
        overlap = max(0.0, min(end, shot_end) - max(start, shot_start))
        if overlap <= 0:
            continue
        shot_id = str(shot.get("segment_id") or "")
        if shot_id:
            present_shots.append(shot_id)
        role_ids = [str(value) for value in (shot.get("role_ids") or []) if str(value).strip()]
        if role_id not in role_ids:
            role_ids.append(role_id)
        shot["role_ids"] = role_ids
        asset_ids = [str(value) for value in (shot.get("asset_ids") or []) if str(value).strip()]
        if role_id not in asset_ids:
            asset_ids.append(role_id)
        shot["asset_ids"] = asset_ids
        tracks = [item for item in (shot.get("role_tracks") or []) if isinstance(item, dict)]
        tracks = [item for item in tracks if str(item.get("role_id") or "") != role_id]
        tracks.append({
            "role_id": role_id,
            "track_id": str(summary.get("role_track_id") or ""),
            "track_dir": str(summary.get("output_dir") or ""),
            "time_overlap": round(overlap, 3),
            "coverage": summary.get("track_quality", {}).get("coverage") if isinstance(summary.get("track_quality"), dict) else None,
        })
        shot["role_tracks"] = tracks
    role["present_time_ranges"] = [[round(start, 3), round(end, 3)]]
    role["present_shots"] = present_shots


def _storyboard_mode2_track_role_with_sam3(
    *,
    root: Path,
    video_path: str,
    role: dict[str, Any],
    anchor: dict[str, Any],
    job_id: str,
    add_log,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    from spvideo.ffmpeg_tools import probe_video
    from spvideo.sam3_tracker import SAM3Tracker

    role_id = str(role.get("id") or "")
    role_name = str(role.get("name") or role_id or "角色")
    point = [float(anchor["point"][0]), float(anchor["point"][1])]
    source_kind = str(anchor.get("source_kind") or "").strip()
    source_shape = _normalize_identity_shape(anchor.get("source_shape") or anchor.get("shape"))
    shape_seed_points = _identity_shape_seed_points(source_shape, point)
    anchor_time = max(0.0, float(anchor.get("time") or role.get("source_time") or 0.1))
    if source_shape and shape_seed_points:
        add_log(
            f"> [{role_name}] 准备 SAM3 身份分轨: time={anchor_time:.2f}s "
            f"手绘区域种子={len(shape_seed_points)} 点"
        )
    else:
        add_log(f"> [{role_name}] 准备 SAM3 身份分轨: time={anchor_time:.2f}s point=({point[0]:.3f},{point[1]:.3f})")
    meta = probe_video(video_path)
    source_fps = float(meta.fps or 0.0) or 24.0
    source_frame_idx = max(0, int(round(anchor_time * source_fps)))
    preferred_object_id: int | None = None
    try:
        preferred_object_id = int(anchor.get("sam3_object_id"))
    except (TypeError, ValueError):
        preferred_object_id = None
    track_id = f"{role_id}_{job_id[:10]}"
    output_dir = _storyboard_mode2_role_track_dir(root, role_id, job_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("mask_*.png"):
        stale.unlink(missing_ok=True)
    (output_dir / "track_summary.json").unlink(missing_ok=True)

    with SAM3_TRACK_LOCK:
        add_log(f"> [{role_name}] 正在截取 SAM3 跟踪窗口...")
        clip_path, prompt_frame, start_frame, clip_frames, track_fps = _make_sam3_window_clip(
            video_path,
            frame_idx=source_frame_idx,
            max_frames=SAM3_PROTECTION_MAX_FRAMES,
            job_id=f"{job_id}_{role_id}",
        )
        add_log(
            f"> [{role_name}] SAM3 窗口已准备: start_frame={start_frame}, "
            f"prompt_frame={prompt_frame}, frames={clip_frames}, fps={track_fps:.2f}"
        )
        tracker = SAM3Tracker()
        try:
            if preferred_object_id is None:
                if source_shape and shape_seed_points:
                    add_log(f"> [{role_name}] 使用手绘区域内多点提示重跑 SAM3...")
                else:
                    add_log(f"> [{role_name}] 蒙版候选不可信/未选颜色块，使用手动点提示重跑 SAM3...")
                tracked = tracker.track_by_point(
                    video_path=str(clip_path),
                    point=shape_seed_points or point,
                    frame_idx=prompt_frame,
                    max_frames=clip_frames,
                    propagation_direction="both",
                )
            else:
                add_log(f"> [{role_name}] 使用已选颜色对象 object={preferred_object_id} 重新匹配 SAM3...")
                tracked = tracker.track_text_objects(
                    video_path=str(clip_path),
                    text_prompt="person",
                    frame_idx=prompt_frame,
                    max_frames=clip_frames,
                    propagation_direction="both",
                )
        finally:
            tracker.close()

    clip_start_time = start_frame / source_fps
    anchor_frame = max(0, min(clip_frames - 1, int(round((anchor_time - clip_start_time) * track_fps))))
    prompt_source = (
        "mode2_sam3_shape_prompt"
        if preferred_object_id is None and source_shape
        else ("mode2_sam3_point_prompt" if preferred_object_id is None else "mode2_sam3_text_objects")
    )
    prompt_mode = str(tracked.get("prompt_mode") or ("points" if preferred_object_id is None else "text_objects"))
    best_object_id: int | None = None
    object_match_source = "manual_shape_prompt" if preferred_object_id is None and source_shape else ("manual_point_prompt" if preferred_object_id is None else "nearest_identity_point")
    identity_distance = 0.0
    source_shape_overlap: float | None = None

    if preferred_object_id is None:
        for frame_index, mask in enumerate(tracked.get("masks") or []):
            if mask is None:
                continue
            mask_array = np.squeeze(mask).astype(np.uint8) * 255
            ok, encoded = cv2.imencode(".png", mask_array)
            if ok:
                (output_dir / f"mask_{frame_index:04d}.png").write_bytes(encoded.tobytes())
        tracked_frames = int(tracked.get("tracked_frames") or 0)
        total_frames = int(tracked.get("num_frames") or 0)
    else:
        object_ids = [int(value) for value in tracked.get("object_ids") or []]
        if not object_ids:
            raise ValueError(f"{role_name} SAM3 没有检测到 person object")

        def object_point_distance(obj_id: int) -> float:
            mask = tracked["objects"][obj_id]["masks"][anchor_frame]
            if mask is None:
                return 10.0
            mask_array = np.squeeze(mask).astype(np.uint8)
            height, width = mask_array.shape[:2]
            px = max(0, min(width - 1, int(round(point[0] * width))))
            py = max(0, min(height - 1, int(round(point[1] * height))))
            if mask_array[py, px] > 0:
                return 0.0
            distances = cv2.distanceTransform((1 - mask_array).astype(np.uint8), cv2.DIST_L2, 3)
            return float(distances[py, px]) / float(max(width, height))

        def object_shape_overlap(obj_id: int) -> float | None:
            mask = tracked["objects"][obj_id]["masks"][anchor_frame]
            if mask is None:
                return None
            mask_array = np.squeeze(mask).astype(np.uint8)
            return _identity_shape_mask_overlap(mask_array, source_shape)

        def object_cost(obj_id: int) -> float:
            distance = object_point_distance(obj_id)
            if not source_shape:
                return distance
            overlap = object_shape_overlap(obj_id)
            return distance + max(0.0, SAM3_SHAPE_OBJECT_TARGET_OVERLAP - float(overlap or 0.0))

        if preferred_object_id in object_ids:
            preferred_distance = object_point_distance(int(preferred_object_id))
            preferred_cost = object_cost(int(preferred_object_id))
            match_threshold = SAM3_SHAPE_OBJECT_MATCH_THRESHOLD if source_shape else 0.05
            if preferred_cost <= match_threshold:
                best_object_id = int(preferred_object_id)
                identity_distance = preferred_distance
                object_match_source = "saved_sam3_object_id"
            else:
                best_object_id = min(object_ids, key=object_cost)
                identity_distance = object_point_distance(best_object_id)
                object_match_source = "saved_object_id_missed_fallback_to_shape" if source_shape else "saved_object_id_missed_fallback_to_point"
        else:
            best_object_id = min(object_ids, key=object_cost)
            identity_distance = object_point_distance(best_object_id)
        source_shape_overlap = object_shape_overlap(best_object_id)
        object_result = tracked["objects"][best_object_id]
        for frame_index, mask in enumerate(object_result["masks"]):
            if mask is None:
                continue
            mask_array = np.squeeze(mask).astype(np.uint8) * 255
            ok, encoded = cv2.imencode(".png", mask_array)
            if ok:
                (output_dir / f"mask_{frame_index:04d}.png").write_bytes(encoded.tobytes())
        tracked_frames = int(object_result.get("tracked_frames") or 0)
        total_frames = int(tracked.get("num_frames") or 0)

    status = "ready"
    warning = ""
    try:
        _validate_sam3_role_track(
            output_dir=output_dir,
            point=point,
            source_shape=source_shape,
            prompt_frame=anchor_frame,
            tracked_frames=tracked_frames,
            total_frames=total_frames,
            role_name=role_name,
        )
        if best_object_id is not None and identity_distance > 0.05:
            status = "needs_manual_review"
            warning = f"SAM3 object 距离身份点偏远: {identity_distance:.3f}"
    except Exception as exc:  # noqa: BLE001
        status = "needs_manual_review"
        warning = str(exc)

    summary = {
        "role_track_id": track_id,
        "prompt_source": prompt_source,
        "prompt_mode": prompt_mode,
        "role_id": role_id,
        "role_name": role_name,
        "anchor": anchor,
        "source_kind": source_kind,
        "source_shape_used": bool(source_shape),
        "source_shape": source_shape or {},
        "prompt_points": shape_seed_points or [point],
        "sam3_object_id": best_object_id,
        "saved_sam3_object_id": preferred_object_id,
        "object_match_source": object_match_source,
        "identity_point_distance": identity_distance,
        "source_shape_overlap": source_shape_overlap,
        "video_path": video_path,
        "clip_path": str(clip_path),
        "clip_start_frame": start_frame,
        "clip_start_time": clip_start_time,
        "text_prompt": "person",
        "text_prompt_frame": prompt_frame,
        "clip_prompt_frame": anchor_frame,
        "source_fps": source_fps,
        "track_fps": track_fps,
        "prompt_point": point,
        "candidate_time": anchor_time,
        "propagation_direction": "both",
        "num_frames": total_frames,
        "tracked_frames": tracked_frames,
        "output_dir": str(output_dir),
        "status": status,
        "warning": warning,
    }
    quality = _storyboard_mode2_track_quality(output_dir, prompt_frame=anchor_frame, point=point, source_shape=source_shape)
    quality["identity_point_distance"] = round(identity_distance, 4)
    if source_shape_overlap is not None:
        quality["source_shape_overlap"] = round(source_shape_overlap, 4)
    summary["track_quality"] = quality
    bundle = _storyboard_mode2_extract_role_source_bundle(
        video_path=video_path,
        role=role,
        track_dir=output_dir,
        summary=summary,
    )
    mask_paths = sorted(output_dir.glob("mask_*.png"))
    preview_path = output_dir / f"role_track_preview_{role_id}_{job_id[:8]}.mp4"
    try:
        from spvideo.scail2_client import Scail2Client

        Scail2Client._render_colored_mask_preview_video(
            [mask_paths],
            preview_path,
            track_fps,
        )
        summary["track_path"] = str(preview_path)
        summary["color_preview_path"] = str(preview_path)
        summary["mask_path"] = str(preview_path)
        add_log(f"> [{role_name}] 分轨检查视频已生成: {preview_path}")
    except Exception as exc:  # noqa: BLE001
        summary["track_path"] = ""
        summary["color_preview_path"] = ""
        summary["mask_path"] = ""
        add_log(f"> [{role_name}] 分轨检查视频生成失败，保留逐帧 PNG: {exc}")
    summary_path = output_dir / "track_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    first_mask = next(iter(mask_paths), None)
    display_mask = Path(str(summary.get("mask_path") or "")) if str(summary.get("mask_path") or "").strip() else first_mask
    role_anchor_list = [
        item for item in (role.get("identity_anchors") or [])
        if isinstance(item, dict)
    ]
    if not any(str(item.get("id") or "") == str(anchor.get("id") or "") for item in role_anchor_list):
        role_anchor_list.append(anchor)
    role_anchor_list = sorted(
        role_anchor_list,
        key=lambda item: (float(item.get("time") or 0.0), float(item.get("updated_at") or 0.0)),
    )
    role.update({
        "identity_status": "tracked" if status == "ready" else "needs_review",
        "identity_anchors": role_anchor_list,
        "role_track_id": track_id,
        "track_dir": str(output_dir),
        "track_summary": str(summary_path),
        "track_source": prompt_source,
        "track_status": status,
        "track_quality": quality,
        "track_path": str(summary.get("track_path") or ""),
        "mask_path": str(display_mask or ""),
        "source_image": bundle["source_image"],
        "source_images": bundle["source_images"],
        "source_crop_paths": bundle["source_crop_paths"],
        "source_mask_paths": bundle["source_mask_paths"],
        "source_frame_count": bundle["source_frame_count"],
        "source_kind": "mode2_role_identity_track_bundle",
        "source_identity_status": "tracked_identity",
        "source_collision": False,
        "source_collision_with": [],
        "refinement_status": status,
        "refinement_method": "mode2_sam3_identity_track",
        "refinement_kind": "role_identity_track_bundle",
        "refined_source_image": bundle["source_image"],
        "refined_source_images": bundle["source_images"],
        "refined_mask_image": str(display_mask or ""),
        "refined_cutout_image": "",
        "refinement_prompt": "mode2 SAM3 identity track from manual anchor",
        "refinement_warning": warning,
        "refinement_error": "",
        "refinement_job_id": job_id,
        "refinement_source_path": bundle["source_images"][0],
        "refinement_quality": quality,
        "refined_at": time.time(),
        "selection_reason": (
            f"SAM3 点提示身份轨 @ {anchor_time:.2f}s，{tracked_frames}/{summary['num_frames']} 帧"
            if best_object_id is None
            else f"SAM3 身份轨 @ {anchor_time:.2f}s，object={best_object_id}，{tracked_frames}/{summary['num_frames']} 帧"
        ),
    })
    if best_object_id is None:
        add_log(f"> [{role_name}] SAM3 point prompt {tracked_frames}/{summary['num_frames']} 帧 status={status}")
    else:
        add_log(f"> [{role_name}] SAM3 object={best_object_id} {tracked_frames}/{summary['num_frames']} 帧 status={status}")
    return summary


def _storyboard_mode2_track_role_with_remote_sam3(
    *,
    root: Path,
    video_path: str,
    role: dict[str, Any],
    anchor: dict[str, Any],
    job_id: str,
    add_log,
) -> dict[str, Any]:
    from spvideo.ffmpeg_tools import probe_video
    from spvideo.scail2_client import Scail2Client

    role_id = str(role.get("id") or "")
    role_name = str(role.get("name") or role_id or "角色")
    ref_image = str(role.get("target_image") or role.get("refined_image") or role.get("representative_image") or role.get("source_image") or "").strip()
    if not ref_image:
        raise ValueError(f"{role_name} 缺少参考图，无法提交服务器分轨")
    point = [float(anchor["point"][0]), float(anchor["point"][1])]
    source_shape = _normalize_identity_shape(anchor.get("source_shape") or anchor.get("shape"))
    anchor_time = max(0.0, float(anchor.get("time") or role.get("source_time") or 0.1))
    meta = probe_video(video_path)
    source_fps = float(meta.fps or 0.0) or 24.0
    source_frame_idx = max(0, int(round(anchor_time * source_fps)))
    output_dir = _storyboard_mode2_role_track_dir(root, role_id, job_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    add_log(f"> [{role_name}] 正在提交服务器 SAM3 分轨...")
    clip_path, prompt_frame, clip_start_frame, clip_frames, track_fps = _make_sam3_window_clip(
        video_path,
        frame_idx=source_frame_idx,
        max_frames=SAM3_PROTECTION_MAX_FRAMES,
        job_id=f"{job_id}_{role_id}_remote",
        max_side=640,
    )
    add_log(
        f"> [{role_name}] Remote SAM3 proxy ready: frames={clip_frames}, "
        f"fps={track_fps:.2f}, prompt_frame={prompt_frame}, input={clip_path.name}"
    )
    client = Scail2Client()
    result = client.inspect_masks(
        video_path=str(clip_path),
        ref_images=[ref_image],
        role_names=[role_name],
        video_window={
            "force_rate": max(1, int(round(track_fps or source_fps))),
            "frame_load_cap": clip_frames,
            "skip_first_frames": 0,
            "select_every_nth": 1,
        },
        sampler_preset="fast",
        source_identity_points=[point],
        source_identity_shapes=[source_shape],
        output_dir=output_dir,
        on_progress=add_log,
    )
    output_path = str(result.get("output_path") or "").strip()
    mask_output_paths = result.get("mask_output_paths") if isinstance(result.get("mask_output_paths"), dict) else {}
    pose_paths = [str(path) for path in (mask_output_paths.get("pose") or []) if str(path or "").strip()]
    summary_path = Path(output_path).with_suffix(".json") if output_path else (Path(video_path).parent / f"{role_id}_{job_id[:8]}_remote_sam3_summary.json")
    status = "ready" if output_path and not (result.get("warnings") or []) else "needs_review"
    warning = ""
    warnings = [str(item) for item in (result.get("warnings") or []) if str(item or "").strip()]
    if warnings:
        warning = "; ".join(warnings[:4])
    summary = {
        "role_track_id": f"{role_id}_{job_id[:10]}",
        "prompt_source": "remote_sam3_color_mask",
        "prompt_mode": "remote_sam3_color_mask",
        "role_id": role_id,
        "role_name": role_name,
        "anchor": anchor,
        "source_kind": "remote_sam3_color_mask",
        "source_shape_used": bool(source_shape),
        "source_shape": source_shape or {},
        "prompt_points": [point],
        "sam3_object_id": None,
        "saved_sam3_object_id": None,
        "object_match_source": "remote_sam3_color_mask",
        "identity_point_distance": 0.0,
        "source_shape_overlap": None,
        "video_path": video_path,
        "clip_path": str(clip_path),
        "clip_start_frame": clip_start_frame,
        "clip_start_time": clip_start_frame / source_fps,
        "text_prompt": "person",
        "text_prompt_frame": prompt_frame,
        "clip_prompt_frame": prompt_frame,
        "source_fps": source_fps,
        "track_fps": track_fps,
        "prompt_point": point,
        "candidate_time": anchor_time,
        "propagation_direction": "both",
        "num_frames": 0,
        "tracked_frames": 0,
        "output_dir": str(Path(output_path).parent if output_path else output_dir),
        "status": status,
        "warning": warning,
        "track_quality": {
            "coverage": 1.0 if output_path else 0.0,
            "mean_area_ratio": 0.0,
            "warnings": warnings,
        },
        "remote_result": result,
        "color_preview_path": output_path,
        "mask_output_paths": mask_output_paths,
        "source_mask_paths": pose_paths,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    role.update({
        "identity_status": "tracked" if status == "ready" else "needs_review",
        "identity_anchors": sorted(
            list((role.get("identity_anchors") or [])) + [anchor],
            key=lambda item: (float(item.get("time") or 0.0), float(item.get("updated_at") or 0.0)),
        ),
        "role_track_id": summary["role_track_id"],
        "track_dir": str(Path(output_path).parent if output_path else summary_path.parent),
        "track_summary": str(summary_path),
        "track_source": "remote_sam3_color_mask",
        "track_status": status,
        "track_quality": summary["track_quality"],
        "track_path": output_path,
        "mask_path": output_path,
        "source_image": str(ref_image),
        "source_images": [str(ref_image)],
        "source_crop_paths": [],
        "source_mask_paths": pose_paths,
        "source_frame_count": 0,
        "source_kind": "remote_sam3_color_mask",
        "source_identity_status": "tracked_identity",
        "source_collision": False,
        "source_collision_with": [],
        "refinement_status": status,
        "refinement_method": "remote_sam3_color_mask",
        "refinement_kind": "role_identity_track_bundle",
        "refined_source_image": str(ref_image),
        "refined_source_images": [str(ref_image)],
        "refined_mask_image": output_path,
        "refined_cutout_image": "",
        "refinement_prompt": "remote SAM3 color mask track",
        "refinement_warning": warning,
        "refinement_error": "",
        "refinement_job_id": job_id,
        "refinement_source_path": ref_image,
    })
    add_log(f"> [{role_name}] 服务器 SAM3 分轨完成: {output_path or '无输出'}")
    if warnings:
        for item in warnings[:4]:
            add_log(f"> [{role_name}] 蒙版提醒: {item}")
    return summary


def _run_storyboard_role_track_job(job_id: str, payload: dict[str, Any]) -> None:
    def add_log(message: str) -> None:
        snapshot = None
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job.setdefault("logs", []).append(message)
                snapshot = dict(job)
        if snapshot is not None:
            _write_storyboard_job_snapshot(snapshot)

    try:
        root = _resolve_storyboard_mode2_project_dir(payload.get("project_dir") or "")
        data = _load_storyboard_mode2_store(root)
        assets = [item for item in (data.get("assets") or []) if isinstance(item, dict)]
        shots = [item for item in (data.get("shots") or []) if isinstance(item, dict)]
        video_path = str(data.get("video_path") or "")
        if not video_path or not Path(video_path).exists():
            raise ValueError(f"source_video_not_found: {video_path}")
        requested_ids = set(_string_list(payload.get("role_ids") or payload.get("asset_ids")))
        roles = [
            item for item in assets
            if str(item.get("kind") or "") == "role"
            and (not requested_ids or str(item.get("id") or "") in requested_ids)
        ]
        if not roles:
            raise ValueError("no matching role assets")
        annotations = [
            item for item in (data.get("identity_annotations") or [])
            if isinstance(item, dict)
        ]
        role_tracks = data.get("role_tracks") if isinstance(data.get("role_tracks"), dict) else {}
        if not isinstance(role_tracks, dict):
            role_tracks = {}
        summary: dict[str, Any] = {
            "job_id": job_id,
            "project_dir": str(root),
            "started_at": time.time(),
            "processed": 0,
            "ready": 0,
            "needs_manual_review": 0,
            "failed": 0,
            "items": [],
        }
        data["role_tracks_version"] = int(data.get("role_tracks_version") or 1)
        data["role_tracks"] = role_tracks
        data.setdefault("role_track_history", [])
        project_config = _normalize_storyboard_project_config(data.get("project_config") or {})
        track_mode = str(payload.get("mode") or project_config.get("mask_source") or DEFAULT_STORYBOARD_PROJECT_CONFIG["mask_source"]).strip()
        use_remote_track = track_mode == "remote_sam3_color"
        summary["track_mode"] = "remote_sam3_color" if use_remote_track else "local_sam3"
        add_log(
            "> 角色分轨来源: "
            + ("服务器 SAM3 彩色蒙版" if use_remote_track else "本机 SAM3")
        )

        def persist() -> None:
            data["assets"] = assets
            data["shots"] = shots
            _write_storyboard_mode2_store(root, data)

        for index, role in enumerate(roles, 1):
            role_id = str(role.get("id") or "")
            role_name = str(role.get("name") or role_id)
            anchor_candidates = [
                item for item in annotations
                if str(item.get("role_id") or "") == role_id
            ]
            anchor = max(
                anchor_candidates,
                key=lambda item: (
                    1 if _normalize_identity_shape(item.get("source_shape") or item.get("shape")) else 0,
                    float(item.get("updated_at") or item.get("time") or 0.0),
                ),
                default=None,
            )
            if not anchor:
                role["identity_status"] = "needs_anchor"
                role["track_status"] = "needs_anchor"
                summary["failed"] += 1
                summary["items"].append({"role_id": role_id, "status": "needs_anchor", "error": "missing identity anchor"})
                add_log(f"> [{index}/{len(roles)}] {role_name} 缺少身份点，跳过")
                persist()
                continue
            if len(anchor_candidates) > 1:
                add_log(f"> [{index}/{len(roles)}] {role_name} 已有 {len(anchor_candidates)} 个锚点，本次使用最近的手绘/点位种子")
            role["identity_status"] = "tracking"
            role["track_status"] = "tracking"
            persist()
            add_log(f"> [{index}/{len(roles)}] 跑 {'服务器' if use_remote_track else '本机'} SAM3 身份分轨: {role_name} @ {float(anchor.get('time') or 0):.2f}s")
            try:
                if use_remote_track:
                    remote_error = ""
                    try:
                        item = _storyboard_mode2_track_role_with_remote_sam3(
                            root=root,
                            video_path=video_path,
                            role=role,
                            anchor=anchor,
                            job_id=job_id,
                            add_log=add_log,
                        )
                    except Exception as exc:  # noqa: BLE001
                        remote_error = str(exc)
                        add_log(f"> {role_name} 服务器 SAM3 分轨失败，改用本机兜底: {remote_error}")
                        item = _storyboard_mode2_track_role_with_sam3(
                            root=root,
                            video_path=video_path,
                            role=role,
                            anchor=anchor,
                            job_id=job_id,
                            add_log=add_log,
                        )
                        item["remote_fallback_used"] = True
                        item["remote_error"] = remote_error
                        item["track_mode"] = "remote_failed_local_sam3"
                        add_log(f"> {role_name} 本机 SAM3 兜底完成: status={item.get('status')}")
                else:
                    item = _storyboard_mode2_track_role_with_sam3(
                        root=root,
                        video_path=video_path,
                        role=role,
                        anchor=anchor,
                        job_id=job_id,
                        add_log=add_log,
                    )
                _storyboard_mode2_attach_role_presence_to_shots(role, shots, item)
                role_tracks[role_id] = item
                summary["processed"] += 1
                if item.get("status") == "ready":
                    summary["ready"] += 1
                else:
                    summary["needs_manual_review"] += 1
                summary["items"].append({"role_id": role_id, "status": item.get("status"), "track_dir": item.get("output_dir"), "warning": item.get("warning")})
            except Exception as exc:  # noqa: BLE001
                error = str(exc)
                role["identity_status"] = "failed"
                role["track_status"] = "failed"
                role["track_error"] = error
                summary["processed"] += 1
                summary["failed"] += 1
                summary["items"].append({"role_id": role_id, "status": "failed", "error": error})
                add_log(f"> {role_name} SAM3 身份分轨失败: {error}")
            finally:
                project_config = _normalize_storyboard_project_config(data.get("project_config") or {})
                _compile_storyboard_prompts(assets, shots, project_config=project_config)
                persist()

        summary["finished_at"] = time.time()
        data["last_role_track"] = summary
        if isinstance(data.get("role_track_history"), list):
            data["role_track_history"].append(summary)
            data["role_track_history"] = data["role_track_history"][-100:]
        persist()
        try:
            _audit_storyboard_mode2_assets({"project_dir": str(root)})
        except Exception as exc:  # noqa: BLE001
            logging.warning("Mode2 asset audit skipped after role track update: %s", exc)
        result = _load_mode2_storyboard_result(root, _storyboard_mode2_asset_store_path(root))
        result["role_track"] = summary
        with JOBS_LOCK:
            job = JOBS[job_id]
            job["status"] = "done"
            job["result"] = result
            job["finished_at"] = time.time()
            job.setdefault("logs", []).append(
                f"> Mode2 SAM3 身份分轨完成: ready={summary['ready']} manual={summary['needs_manual_review']} failed={summary['failed']}"
            )
            snapshot = dict(job)
        _write_storyboard_job_snapshot(snapshot)
    except Exception as exc:  # noqa: BLE001
        with JOBS_LOCK:
            job = JOBS[job_id]
            job["status"] = "failed"
            job["error"] = str(exc)
            job["finished_at"] = time.time()
            job.setdefault("logs", []).append(f"> Mode2 SAM3 身份分轨失败: {exc}")
            snapshot = dict(job)
        _write_storyboard_job_snapshot(snapshot)


def _run_storyboard_asset_refine_job(job_id: str, payload: dict[str, Any]) -> None:
    def add_log(message: str) -> None:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job.setdefault("logs", []).append(message)

    def check_cancel() -> None:
        with JOBS_LOCK:
            if JOBS.get(job_id, {}).get("_cancel"):
                raise RuntimeError("用户取消")

    try:
        root = _resolve_storyboard_mode2_project_dir(payload.get("project_dir") or "")
        store_path = _storyboard_mode2_asset_store_path(root)
        if not store_path.exists():
            raise ValueError(f"mode2 storyboard asset store not found: {store_path}")
        data = json.loads(store_path.read_text(encoding="utf-8-sig"))
        _refresh_mode2_structured_fields(root, data)
        assets = [
            item for item in (data.get("assets") or [])
            if isinstance(item, dict)
        ]
        shots = [
            item for item in (data.get("shots") or [])
            if isinstance(item, dict)
        ]
        if not assets:
            raise ValueError("storyboard assets are empty")

        requested_ids = set(_string_list(payload.get("asset_ids")))
        requested_kinds = set(_string_list(payload.get("kinds") or payload.get("asset_kinds")))
        if str(payload.get("kind") or "").strip():
            requested_kinds.add(str(payload.get("kind") or "").strip())
        force = bool(payload.get("force"))
        selected_assets = [
            asset for asset in assets
            if (not requested_ids or str(asset.get("id") or "") in requested_ids)
            and (not requested_kinds or str(asset.get("kind") or "") in requested_kinds)
        ]
        if not selected_assets:
            raise ValueError("no matching assets to refine")

        add_log(
            f"> 开始 Mode2 资产提纯: {len(selected_assets)}/{len(assets)} 个资产；"
            f"force={force}"
        )
        data["assets"] = assets
        if not isinstance(data.get("refinement_history"), list):
            data["refinement_history"] = []
        summary: dict[str, Any] = {
            "job_id": job_id,
            "project_dir": str(root),
            "store_path": str(store_path),
            "started_at": time.time(),
            "processed": 0,
            "ready": 0,
            "failed": 0,
            "skipped": 0,
            "needs_manual_review": 0,
            "items": [],
        }

        def persist() -> None:
            data["assets"] = assets
            data["shots"] = shots
            data["last_refinement"] = summary
            store_path.parent.mkdir(parents=True, exist_ok=True)
            store_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

        client: ComfyClient | None = None

        def comfy() -> ComfyClient:
            nonlocal client
            if client is None:
                client = ComfyClient(COMFY_URL)
            return client

        for index, asset in enumerate(selected_assets, 1):
            check_cancel()
            asset_id = str(asset.get("id") or "")
            kind = str(asset.get("kind") or "")
            name = str(asset.get("name") or asset_id or "asset")
            if (
                not force
                and str(asset.get("refinement_status") or "") == "ready"
                and str(asset.get("refined_source_image") or "").strip()
                and (
                    kind not in {"scene", "prop"}
                    or bool(asset.get("representative_is_clean"))
                )
            ):
                summary["skipped"] += 1
                summary["items"].append({"asset_id": asset_id, "kind": kind, "status": "skipped"})
                add_log(f"> 跳过已提纯资产: {name}")
                continue

            asset["refinement_status"] = "running"
            asset["refinement_error"] = ""
            asset["refinement_job_id"] = job_id
            persist()
            add_log(f"> [{index}/{len(selected_assets)}] 提纯 {kind or 'asset'}: {name}")
            try:
                if bool(asset.get("placeholder")) or str(asset.get("status") or "") == "needs_identification":
                    raise RuntimeError("placeholder asset cannot be refined before full-video understanding")
                if kind == "role":
                    item = _refine_storyboard_role_asset(root, asset, job_id=job_id, add_log=add_log)
                elif kind == "scene":
                    item = _refine_storyboard_scene_asset(root, asset, client=comfy(), job_id=job_id, add_log=add_log)
                elif kind == "prop":
                    item = _refine_storyboard_prop_asset(root, asset, client=comfy(), job_id=job_id, add_log=add_log)
                else:
                    raise RuntimeError(f"unsupported asset kind: {kind}")
                summary["processed"] += 1
                if item.get("status") == "needs_manual_review":
                    summary["needs_manual_review"] += 1
                else:
                    summary["ready"] += 1
                summary["items"].append(item)
            except Exception as exc:  # noqa: BLE001
                summary["processed"] += 1
                summary["failed"] += 1
                error = str(exc)
                _set_storyboard_asset_refinement_failure(asset, job_id=job_id, error=error)
                summary["items"].append({"asset_id": asset_id, "kind": kind, "status": "failed", "error": error})
                add_log(f"> 资产提纯失败: {name} / {error}")
            finally:
                persist()

        summary["finished_at"] = time.time()
        project_config = _normalize_storyboard_project_config(data.get("project_config") or {})
        _compile_storyboard_prompts(assets, shots, project_config=project_config)
        data["assets"] = assets
        data["shots"] = shots
        data["last_refinement"] = summary
        data["refinement_history"].append(summary)
        persist()
        try:
            _audit_storyboard_mode2_assets({"project_dir": str(root)})
        except Exception as exc:  # noqa: BLE001
            logging.warning("Mode2 asset audit skipped after refinement update: %s", exc)

        result = _load_mode2_storyboard_result(root, store_path)
        result["refinement"] = summary
        with JOBS_LOCK:
            job = JOBS[job_id]
            job["status"] = "done"
            job["result"] = result
            job["finished_at"] = time.time()
            job.setdefault("logs", []).append(
                f"> 资产提纯完成: ready={summary['ready']} failed={summary['failed']} "
                f"manual={summary['needs_manual_review']} skipped={summary['skipped']}"
            )
            snapshot = dict(job)
        _write_storyboard_job_snapshot(snapshot)
    except Exception as exc:  # noqa: BLE001
        with JOBS_LOCK:
            job = JOBS[job_id]
            job["status"] = "failed"
            job["error"] = str(exc)
            job["finished_at"] = time.time()
            job.setdefault("logs", []).append(f"> Mode2 资产提纯失败: {exc}")
            snapshot = dict(job)
        _write_storyboard_job_snapshot(snapshot)


def _run_auto_director_job(job_id: str, payload: dict[str, Any]) -> None:
    def add_log(message: str) -> None:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job.setdefault("logs", []).append(message)

    try:
        project_dir = str(payload.get("project_dir") or "").strip()
        result = analyze_auto_director_project(
            project_dir,
            use_story_model=bool(payload.get("use_story_model", False)),
            api_key=str(payload.get("api_key") or "").strip(),
            base_url=str(payload.get("base_url") or "").strip(),
            model=str(payload.get("model") or "").strip(),
            scan_faces=bool(payload.get("scan_faces", True)),
            on_progress=add_log,
        )
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job["status"] = "done"
                job["result"] = result
                job["finished_at"] = time.time()
                job.setdefault("logs", []).append("> 自动导演问题库已保存")
    except Exception as exc:  # noqa: BLE001
        import traceback
        tb = traceback.format_exc()
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job["status"] = "failed"
                job["error"] = str(exc)
                job["finished_at"] = time.time()
                job.setdefault("logs", []).append("✖ 自动导演失败: " + str(exc))
                job.setdefault("logs", []).append("=== traceback ===")
                for line in tb.splitlines():
                    job.setdefault("logs", []).append(line)


def _run_split_job(job_id: str, payload: dict[str, Any]) -> None:
    root_logger = logging.getLogger("spvideo")
    root_logger.setLevel(logging.INFO)
    handler = JobLogHandler(job_id)
    handler.setLevel(logging.INFO)
    root_logger.addHandler(handler)

    def add_log(message: str) -> None:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job.setdefault("logs", []).append(message)

    try:
        video_path = str(payload.get("video_path") or "").strip()
        project_dir = str(payload.get("project_dir") or "").strip()
        if not video_path:
            raise ValueError("缺少原视频路径")
        if not project_dir:
            raise ValueError("缺少项目目录")

        add_log(f"> 开始切分: {video_path}")
        add_log("> 正在读取视频信息并准备批量抽帧...")
        def check_cancel():
            with JOBS_LOCK:
                if JOBS.get(job_id, {}).get("_cancel"):
                    raise RuntimeError("用户取消")

        check_cancel()
        result = run_segmentation(
            video_path=video_path,
            project_dir=project_dir,
            sample_interval=float(payload.get("sample_interval") or 0.1),
            min_segment_duration=float(payload.get("min_segment_duration") or 1.0),
            max_segment_duration=float(payload.get("max_segment_duration") or 6.0),
            export_video=bool(payload.get("export_video", True)),
            extract_backgrounds=bool(payload.get("extract_backgrounds", False)),
            use_two_pass=bool(payload.get("use_two_pass", False)),
            use_omnishotcut=bool(payload.get("use_omnishotcut", True)),
            use_scene_detect=bool(payload.get("use_scene_detect", True)),
            use_face_id=bool(payload.get("use_face_id", True)),
            use_visual_model=bool(payload.get("use_visual_model", False)),
            use_sam3_finalize=bool(payload.get("use_sam3_finalize", False)),
            use_visual_merge=bool(payload.get("use_visual_merge", True)),
            use_pre_director=bool(payload.get("use_pre_director", True)),
            pre_director_api_key=str(payload.get("pre_director_api_key") or _wan_api_key()),
            pre_director_base_url=str(
                payload.get("pre_director_base_url")
                or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            pre_director_model=str(payload.get("pre_director_model") or "qwen3.5-omni-plus"),
            yolo_conf_threshold=float(payload.get("yolo_conf_threshold") or 0.35),
            device=payload.get("device") or None,
            yolo_batch_size=int(payload.get("yolo_batch_size") or 8),
        )
        summary = {
            "project_dir": result.get("project_dir"),
            "clips_dir": result.get("clips_dir"),
            "report_path": result.get("report_path"),
            "segment_count": result.get("segment_count"),
            "background_count": result.get("background_count"),
            "visual_merge": result.get("visual_merge"),
            "sam3_finalize": result.get("sam3_finalize"),
            "pre_director": result.get("pre_director"),
            "segments": result.get("segments", []),
        }
        with JOBS_LOCK:
            job = JOBS[job_id]
            job["status"] = "done"
            job["result"] = summary
            job["finished_at"] = time.time()
            job.setdefault("logs", []).append("✓ 切分完成")
    except Exception as exc:  # noqa: BLE001
        with JOBS_LOCK:
            job = JOBS[job_id]
            job["status"] = "failed"
            job["error"] = str(exc)
            job["finished_at"] = time.time()
            job.setdefault("logs", []).append(f"✖ 失败: {exc}")
    finally:
        root_logger.removeHandler(handler)


def _run_sam3_track_job(job_id: str, payload: dict[str, Any]) -> None:
    def add_log(message: str) -> None:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job.setdefault("logs", []).append(message)

    project_dir = str(payload.get("project_dir") or "").strip()
    annotation_id = str(payload.get("annotation_id") or "").strip()
    try:
        if not project_dir:
            raise ValueError("missing_project_dir")
        if not annotation_id:
            raise ValueError("missing_annotation_id")

        store = load_asset_store(project_dir)
        annotation = next((item for item in store.get("annotations", []) if item.get("id") == annotation_id), None)
        if not annotation:
            raise ValueError("annotation_not_found")

        video_path = str(annotation.get("video_path") or payload.get("video_path") or "").strip()
        point = annotation.get("point")
        if not video_path or not Path(video_path).exists():
            raise ValueError("video_not_found")
        if not point or len(point) != 2:
            raise ValueError("point_prompt_required")

        update_annotation(
            project_dir,
            annotation_id,
            track_status="tracking",
            track_error="",
        )

        fps = _video_fps(video_path)
        frame_idx = max(0, int(round(float(annotation.get("time") or 0) * fps)))
        requested_frames = int(payload.get("max_frames") or SAM3_PROTECTION_MAX_FRAMES)
        max_frames = max(1, min(requested_frames, SAM3_PROTECTION_MAX_FRAMES))
        output_dir = Path(project_dir) / "assets" / "tracks" / f"{annotation_id}_sam3"
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale_mask in output_dir.glob("mask_*.png"):
            stale_mask.unlink(missing_ok=True)
        (output_dir / "track_summary.json").unlink(missing_ok=True)
        clip_path, clip_prompt_frame, clip_start_frame, clip_frames, track_fps = _make_sam3_window_clip(
            video_path,
            frame_idx=frame_idx,
            max_frames=max_frames,
            job_id=job_id,
        )

        add_log(
            f"> SAM3 loading short clip, source_frame={frame_idx}, "
            f"clip_frame={clip_prompt_frame}, frames={clip_frames}, fps={fps:.2f}"
        )
        from spvideo.sam3_tracker import SAM3Tracker

        with SAM3_TRACK_LOCK:
            tracker = SAM3Tracker()
            try:
                result = tracker.track_by_point(
                    video_path=str(clip_path),
                    point=[float(point[0]), float(point[1])],
                    frame_idx=clip_prompt_frame,
                    max_frames=clip_frames,
                    output_dir=str(output_dir),
                )
            finally:
                tracker.close()

        summary = {
            "annotation_id": annotation_id,
            "video_path": video_path,
            "clip_path": str(clip_path),
            "clip_start_frame": clip_start_frame,
            "clip_start_time": clip_start_frame / fps,
            "frame_idx": frame_idx,
            "clip_prompt_frame": clip_prompt_frame,
            "max_frames": max_frames,
            "source_fps": fps,
            "track_fps": track_fps,
            "output_dir": str(output_dir),
            "num_frames": int(result.get("num_frames") or 0),
            "tracked_frames": int(result.get("tracked_frames") or 0),
        }
        (output_dir / "track_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        update_annotation(
            project_dir,
            annotation_id,
            track_status="ready",
            track_dir=str(output_dir),
            track_summary=str(output_dir / "track_summary.json"),
            tracked_frames=summary["tracked_frames"],
            track_frames=summary["num_frames"],
        )
        with JOBS_LOCK:
            job = JOBS[job_id]
            job["status"] = "done"
            job["result"] = summary
            job["finished_at"] = time.time()
            job.setdefault("logs", []).append(
                f"✓ SAM3 tracked {summary['tracked_frames']}/{summary['num_frames']} frames"
            )
            job.setdefault("logs", []).append(f"> masks: {output_dir}")
    except Exception as exc:  # noqa: BLE001
        if project_dir and annotation_id:
            update_annotation(
                project_dir,
                annotation_id,
                track_status="failed",
                track_error=str(exc),
            )
        with JOBS_LOCK:
            job = JOBS[job_id]
            job["status"] = "failed"
            job["error"] = str(exc)
            job.setdefault("logs", []).append("✖ SAM3 track failed: " + str(exc))


def _video_fps(video_path: str) -> float:
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        finally:
            cap.release()
        return fps if fps > 0 else 24.0
    except Exception:
        return 24.0


def _video_duration(video_path: str) -> float:
    try:
        from spvideo.ffmpeg_tools import probe_video

        return max(0.0, float(probe_video(Path(video_path)).duration or 0.0))
    except Exception:
        return 0.0


def _clamp_video_time(video_path: str, time_seconds: float) -> float:
    value = max(0.0, float(time_seconds or 0.0))
    duration = _video_duration(video_path)
    if duration <= 0:
        return value
    return min(value, max(0.0, duration - 0.05))


def _same_video_path(left: Any, right: Any) -> bool:
    left_value = str(left or "").strip()
    right_value = str(right or "").strip()
    if not left_value or not right_value:
        return False
    try:
        return Path(left_value).resolve() == Path(right_value).resolve()
    except OSError:
        return os.path.normcase(os.path.abspath(left_value)) == os.path.normcase(os.path.abspath(right_value))


def _make_sam3_window_clip(
    video_path: str,
    *,
    frame_idx: int,
    max_frames: int,
    job_id: str,
    max_side: int = 720,
    mode: str = "centered",
) -> tuple[Path, int, int, int, float]:
    """Build a short SAM3 tracking clip with ffmpeg for more reliable seeking."""
    import subprocess
    from spvideo.ffmpeg_tools import ffmpeg_path, probe_video, subprocess_no_window_kwargs

    meta = probe_video(Path(video_path))
    fps = float(meta.fps or 0) or 24.0
    duration = max(0.0, float(meta.duration or 0.0))
    total = max(1, int(round(duration * fps)) if duration > 0 else int(round(fps * 1)))

    max_frames = max(1, min(int(max_frames), SAM3_PROTECTION_MAX_FRAMES))
    track_fps = min(fps, SAM3_PROTECTION_FPS)
    total_duration = total / fps if total > 0 else duration
    if total_duration <= 0:
        raise ValueError("empty_video")
    prompt_time = max(0.0, min(max(0.0, total_duration - 0.05), frame_idx / fps))

    if mode == "forward":
        start_time = prompt_time
        window_duration = min(max_frames / track_fps, max(0.001, total_duration - start_time))
    else:
        window_duration = min(total_duration, max_frames / track_fps)
        start_time = max(0.0, prompt_time - window_duration / 2.0)
        start_time = min(start_time, max(0.0, total_duration - window_duration))

    output_frames = max(1, min(max_frames, int(math.ceil(window_duration * track_fps))))
    start = max(0, int(round(start_time * fps)))
    if mode == "forward":
        prompt_frame = 0
    else:
        prompt_frame = max(
            0,
            min(output_frames - 1, int(round((prompt_time - start_time) * track_fps))),
        )

    tmp_dir = ROOT.parent / ".sam3_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    clip_path = tmp_dir / f"{job_id}_window.mp4"

    ffmpeg = ffmpeg_path()
    filter_candidates = [
        f"scale={max_side}:{max_side}:force_original_aspect_ratio=decrease",
        None,
    ]
    result = None
    for scale_filter in filter_candidates:
        clip_path.unlink(missing_ok=True)
        command = [
            ffmpeg, "-y", "-loglevel", "error",
            "-ss", f"{start_time:.3f}",
            "-i", str(video_path),
            "-t", f"{window_duration:.3f}",
            "-r", f"{track_fps:.6f}",
            "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p",
        ]
        if scale_filter is not None:
            command.extend(["-vf", scale_filter])
        command.append(str(clip_path))
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            **subprocess_no_window_kwargs(),
        )
        if result.returncode == 0 and clip_path.exists() and clip_path.stat().st_size > 0:
            break
    if result is None or result.returncode != 0 or not clip_path.exists() or clip_path.stat().st_size <= 0:
        detail = (result.stderr or result.stdout or "ffmpeg clip failed").strip()[-240:]
        raise ValueError(f"cannot_create_sam3_clip: {detail}")

    # Prefer actual produced frame count if available.
    try:
        import cv2
        cap = cv2.VideoCapture(str(clip_path))
        try:
            written = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            cap.release()
        if written > 0:
            output_frames = written
            prompt_frame = min(prompt_frame, max(0, written - 1))
    except Exception:
        pass

    if output_frames <= 0:
        raise ValueError("empty_sam3_clip")
    return clip_path, prompt_frame, start, output_frames, track_fps


def _pick_path(kind: str, initial: str = "") -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    initial_path = Path(initial) if initial else Path.cwd()
    if initial_path.is_file():
        initial_dir = initial_path.parent
    elif initial_path.exists():
        initial_dir = initial_path
    else:
        initial_dir = Path.cwd()

    if kind == "dir":
        picked = filedialog.askdirectory(
            title="选择项目目录",
            initialdir=str(initial_dir),
            parent=root,
        )
    elif kind == "image":
        picked = filedialog.askopenfilename(
            title="选择替换图片",
            initialdir=str(initial_dir),
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.webp *.bmp *.gif"),
                ("All Files", "*.*"),
            ],
            parent=root,
        )
    else:
        picked = filedialog.askopenfilename(
            title="选择原视频",
            initialdir=str(initial_dir),
            filetypes=[
                ("Video Files", "*.mp4 *.mov *.mkv *.avi *.webm"),
                ("All Files", "*.*"),
            ],
            parent=root,
        )
    root.destroy()
    return str(picked or "")




_RVM_MODEL = None
_RVM_DEVICE = None


def _rvm_auto_matting(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract the current frame with RVM and register it as a reusable asset."""
    global _RVM_MODEL, _RVM_DEVICE

    video_path = str(payload.get("video_path") or "").strip()
    project_value = str(payload.get("project_dir") or "").strip()
    label_name = str(payload.get("label_name") or "").strip()
    asset_name = str(payload.get("asset_name") or "").strip()
    time_seconds = max(0.0, float(payload.get("time") or 0))

    if not video_path or not Path(video_path).exists():
        raise ValueError("video_not_found")
    if not project_value:
        raise ValueError("missing_project_dir")
    if not label_name:
        raise ValueError("missing_character_name")

    import cv2
    import numpy as np
    import torch
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("cannot_open_video")
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, time_seconds * 1000.0)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise ValueError("frame_extraction_failed")

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).float().div(255.0).permute(2, 0, 1).unsqueeze(0)

    with RVM_MODEL_LOCK:
        if _RVM_MODEL is None:
            _RVM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            _RVM_MODEL = torch.hub.load(
                "PeterL1n/RobustVideoMatting",
                "mobilenetv3",
                trust_repo=True,
            ).to(_RVM_DEVICE).eval()

        with torch.no_grad():
            fgr, pha, *_ = _RVM_MODEL(tensor.to(_RVM_DEVICE))

    fgr_np = np.clip(
        fgr[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0,
        0,
        255,
    ).astype(np.uint8)
    alpha_np = np.clip(
        pha[0, 0].detach().cpu().numpy() * 255.0,
        0,
        255,
    ).astype(np.uint8)

    binary_mask = np.where(alpha_np >= 16, 255, 0).astype(np.uint8)
    mask_box = _rvm_mask_box(binary_mask)
    if mask_box is None:
        raise ValueError("rvm_foreground_not_found")

    project_dir = Path(project_value)
    mask_dir = project_dir / "assets" / "masks"
    original_dir = project_dir / "assets" / "originals"
    mask_dir.mkdir(parents=True, exist_ok=True)
    original_dir.mkdir(parents=True, exist_ok=True)

    token = f"rvm_{uuid.uuid4().hex[:10]}"
    mask_path = mask_dir / f"{token}.png"
    context_path = original_dir / f"{token}_context.png"
    cutout_path = original_dir / f"{token}_cutout.png"

    Image.fromarray(alpha_np).save(mask_path)
    Image.fromarray(img_rgb).save(context_path)

    height, width = alpha_np.shape
    x, y, box_w, box_h = mask_box
    pad_x = max(6, int(box_w * width * 0.03))
    pad_y = max(6, int(box_h * height * 0.03))
    x1 = max(0, int(x * width) - pad_x)
    y1 = max(0, int(y * height) - pad_y)
    x2 = min(width, int((x + box_w) * width) + pad_x)
    y2 = min(height, int((y + box_h) * height) + pad_y)
    rgba = np.dstack([fgr_np, alpha_np])
    Image.fromarray(rgba[y1:y2, x1:x2]).save(cutout_path)

    # 生成绿色叠加 PNG（人物区域半透明绿，背景透明），前端直接 img 叠加
    overlay_path = original_dir / f"{token}_overlay.png"
    green_rgb = np.full((height, width, 3), [0, 210, 80], dtype=np.uint8)
    green_alpha = (alpha_np.astype(np.float32) * 0.45).astype(np.uint8)
    overlay_rgba = np.dstack([green_rgb, green_alpha])
    Image.fromarray(overlay_rgba).save(overlay_path)

    point = [
        round(x + box_w / 2.0, 6),
        round(y + box_h / 2.0, 6),
    ]
    annotation = add_annotation(
        project_dir,
        video_path=video_path,
        time_seconds=time_seconds,
        label_id=int(payload.get("label_id") or 0),
        label_name=label_name,
        kind="person",
        point=point,
        box=mask_box,
        segment_id=str(payload.get("segment_id") or ""),
    )
    update_annotation_mask(
        project_dir,
        str(annotation["id"]),
        mask_path=str(mask_path),
        mask_type="rvm_matting",
        mask_box=mask_box,
    )
    original = add_original_asset(
        project_dir,
        annotation_id=str(annotation["id"]),
        label_id=int(annotation.get("label_id") or 0),
        label_name=label_name,
        kind="person",
        asset_name=asset_name or f"{label_name}_{time_seconds:.2f}s",
        role="source_to_keep_or_replace",
        video_path=video_path,
        time_seconds=time_seconds,
        crop_path=str(cutout_path),
        context_path=str(context_path),
        cutout_path=str(cutout_path),
        mask_path=str(mask_path),
        box=mask_box,
    )
    mask = {
        "id": f"mask_{annotation['id']}",
        "annotation_id": str(annotation["id"]),
        "type": "rvm_matting",
        "rvm": True,
        "path": str(mask_path),
        "overlay_path": str(overlay_path),
        "cutout_path": str(cutout_path),
        "box": mask_box,
        "time": time_seconds,
    }
    return {
        "mask": mask,
        "original": original,
        "assets": load_asset_store(project_dir),
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "rgba_path": str(cutout_path),
        "time": time_seconds,
    }


def _rvm_mask_box(mask) -> list[float] | None:
    import cv2
    import numpy as np

    height, width = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(32.0, height * width * 0.0005)
    contours = [item for item in contours if cv2.contourArea(item) >= min_area]
    if not contours:
        return None

    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    return [
        round(x / width, 6),
        round(y / height, 6),
        round(w / width, 6),
        round(h / height, 6),
    ]




def run_server(host: str = "127.0.0.1", port: int = 7861, open_browser: bool = True) -> None:
    server = ThreadingHTTPServer((host, port), SplitterHandler)
    url = f"http://{host}:{port}/"
    print(f"[web] loaded {Path(__file__).resolve()} delete_routes=yes", flush=True)
    print(f"SP 视频切分 Web 界面: {url}")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
