from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

from .settings_store import ensure_runtime_dirs, load_settings


JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def _job_root() -> Path:
    settings = load_settings()
    ensure_runtime_dirs(settings)
    return Path(str(settings["foreign_dub"]["paths"]["job_root"]))


def job_snapshot_path(job_id: str) -> Path:
    safe = "".join(ch for ch in str(job_id or "") if ch.isalnum() or ch in ("-", "_"))
    return _job_root() / f"{safe}.json"


def write_job_snapshot(job: dict[str, Any]) -> None:
    path = job_snapshot_path(str(job.get("id") or ""))
    if not path.name.endswith(".json"):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = dict(job)
    snapshot.pop("_cancel", None)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_job_snapshot(job_id: str) -> dict[str, Any]:
    path = job_snapshot_path(job_id)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    data.setdefault("id", job_id)
    data["restored_from_snapshot"] = True
    if data.get("status") == "running":
        data["status"] = "failed"
        data["error"] = data.get("error") or "任务在后端重启后中断，请重新提交。"
    return data


def create_job(job_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    job = {
        "id": job_id,
        "type": job_type,
        "status": "queued",
        "progress": 0,
        "payload": payload or {},
        "logs": [f"> {job_type} 已提交"],
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    write_job_snapshot(job)
    return job


def get_job(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = dict(JOBS.get(job_id) or {})
        if "logs" in job:
            job["logs"] = list(job["logs"])
    if job:
        return job
    return load_job_snapshot(job_id)


def update_job(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(updates)
        job["updated_at"] = time.time()
        snapshot = dict(job)
    write_job_snapshot(snapshot)


def append_log(job_id: str, line: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.setdefault("logs", []).append(line)
        job["updated_at"] = time.time()
        snapshot = dict(job)
    write_job_snapshot(snapshot)


def cancel_job(job_id: str) -> bool:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return False
        job["_cancel"] = True
        job["status"] = "cancelled"
        job.setdefault("logs", []).append("> 用户已取消")
        snapshot = dict(job)
    write_job_snapshot(snapshot)
    return True


def should_cancel(job_id: str) -> bool:
    with JOBS_LOCK:
        return bool(JOBS.get(job_id, {}).get("_cancel"))


def run_background(job: dict[str, Any], target: Callable[[str, dict[str, Any]], None]) -> None:
    job_id = str(job["id"])

    def runner() -> None:
        update_job(job_id, status="running", progress=1)
        try:
            target(job_id, dict(job.get("payload") or {}))
            if get_job(job_id).get("status") not in {"cancelled", "failed"}:
                update_job(job_id, status="done", progress=100)
                append_log(job_id, "> 任务完成")
        except Exception as exc:
            update_job(job_id, status="failed", error=str(exc), traceback=traceback.format_exc())
            append_log(job_id, f"> 任务失败: {exc}")

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
