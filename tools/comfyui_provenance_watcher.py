#!/usr/bin/env python3
"""ComfyUI provenance watcher.

For every completed ComfyUI prompt, write a sidecar `<output-stem>.config.json`
next to the generated file and append one row to `output/_provenance/manifest.*`.

Design goals:
- no ComfyUI restart, no workflow edit, stdlib only;
- manual GUI runs are captured through ComfyUI's /history API;
- if history is unavailable, new files still get a filesystem-only sidecar;
- never modifies or renames generated media.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
MEDIA_EXTS = VIDEO_EXTS | {".png", ".jpg", ".jpeg", ".webp", ".gif"}
INTEREST_KEYS = {
    "steps", "width", "height", "length", "frames", "frame_count", "fps", "frame_rate",
    "seed", "noise_seed", "cfg", "guidance_scale", "sampler_name", "scheduler", "denoise",
    "shift", "base_shift", "model", "ckpt_name", "unet_name", "vae_name", "clip_name",
    "lora_name", "strength_model", "strength_clip", "video", "image", "filename", "path", "upload",
}
ACCEL_RE = re.compile(r"hyperstep|solattn|sage|triton|flash|skip|accelerat|cache|teacache|pab|parar", re.I)
REF_RE = re.compile(r"\.(mp4|webm|mov|mkv|avi|png|jpe?g|webp|wav|mp3|flac)$", re.I)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def http_json(base_url: str, path: str, timeout: float = 10.0) -> Any:
    url = base_url.rstrip("/") + path
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - owner-controlled local ComfyUI
        return json.loads(resp.read().decode("utf-8"))


def gpu_info() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def compact_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in (inputs or {}).items() if scalar(v)}


def summarize_workflow(prompt: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "node_count": 0,
        "class_types": {},
        "params": {},
        "references": [],
        "acceleration_nodes": [],
    }
    if not isinstance(prompt, dict):
        return summary
    summary["node_count"] = len(prompt)
    refs: list[str] = []
    accels: list[dict[str, Any]] = []
    for node_id, node in prompt.items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type") or "")
        inputs = compact_inputs(node.get("inputs") or {})
        summary["class_types"][class_type] = summary["class_types"].get(class_type, 0) + 1
        for key, value in inputs.items():
            low = str(key).lower()
            if low in INTEREST_KEYS and key not in summary["params"]:
                summary["params"][key] = value
            if isinstance(value, str) and REF_RE.search(value):
                refs.append(value)
        hay = json.dumps({"class_type": class_type, "inputs": inputs}, ensure_ascii=False)
        if ACCEL_RE.search(hay):
            accels.append({"node_id": str(node_id), "class_type": class_type, "inputs": inputs})
    summary["references"] = sorted(set(refs))
    summary["acceleration_nodes"] = accels
    return summary


def history_outputs(record: dict[str, Any]) -> list[dict[str, str]]:
    files: list[dict[str, str]] = []
    outputs = record.get("outputs") or {}
    if not isinstance(outputs, dict):
        return files
    for node_id, node_out in outputs.items():
        if not isinstance(node_out, dict):
            continue
        for group in ("videos", "images", "gifs", "animated", "audio"):
            items = node_out.get(group) or []
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                filename = str(item.get("filename") or "")
                if not filename:
                    continue
                ext = Path(filename).suffix.lower()
                if ext and ext not in MEDIA_EXTS and group != "audio":
                    continue
                files.append({
                    "node_id": str(node_id),
                    "group": group,
                    "filename": filename,
                    "subfolder": str(item.get("subfolder") or ""),
                    "type": str(item.get("type") or "output"),
                })
    return files


def read_accel_snapshots(output_dir: Path, limit: int = 5) -> list[dict[str, Any]]:
    snaps: list[dict[str, Any]] = []
    try:
        candidates = sorted(output_dir.glob("*last_run*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        candidates = []
    for path in candidates[:limit]:
        try:
            snaps.append({
                "path": str(path),
                "mtime": path.stat().st_mtime,
                "data": json.loads(path.read_text(encoding="utf-8")),
                "note": "same output dir last-run snapshot; verify timestamp before treating as this run's config",
            })
        except Exception:
            continue
    return snaps


def manifest_row(record: dict[str, Any], file_item: dict[str, str], sidecar: Path, output_dir: Path) -> dict[str, Any]:
    summary = record.get("summary") or {}
    params = summary.get("params") or {}
    rel_path = str(Path(file_item.get("subfolder") or "") / file_item.get("filename") or "")
    return {
        "created_at": record.get("created_at") or utc_now(),
        "prompt_id": record.get("prompt_id") or "",
        "file": file_item.get("filename") or "",
        "rel_path": rel_path,
        "width": params.get("width") or "",
        "height": params.get("height") or "",
        "steps": params.get("steps") or "",
        "length": params.get("length") or params.get("frames") or params.get("frame_count") or "",
        "fps": params.get("fps") or params.get("frame_rate") or "",
        "seed": params.get("seed") or params.get("noise_seed") or "",
        "references": ";".join(summary.get("references") or []),
        "acceleration": json.dumps(summary.get("acceleration_nodes") or [], ensure_ascii=False),
        "source": record.get("source") or "",
        "sidecar": str(sidecar.relative_to(output_dir)) if sidecar.is_relative_to(output_dir) else str(sidecar),
    }


class ProvenanceWatcher:
    def __init__(self, base_url: str, output_dir: Path, interval: float, dry_run: bool = False, backfill: bool = False):
        self.base_url = base_url.rstrip("/")
        self.output_dir = output_dir
        self.interval = interval
        self.dry_run = dry_run
        self.backfill = backfill
        self.prov_dir = output_dir / "_provenance"
        self.state_path = self.prov_dir / "seen.json"
        self.manifest_jsonl = self.prov_dir / "manifest.jsonl"
        self.manifest_csv = self.prov_dir / "manifest.csv"
        self.started = time.time()
        self.state: dict[str, Any] = {"history": [], "files": []}
        self.load_state()

    def load_state(self) -> None:
        try:
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            self.state = {"history": [], "files": []}
        self.state.setdefault("history", [])
        self.state.setdefault("files", [])

    def save_state(self) -> None:
        if self.dry_run:
            return
        self.prov_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def append_manifest(self, rows: list[dict[str, Any]]) -> None:
        if not rows or self.dry_run:
            return
        self.prov_dir.mkdir(parents=True, exist_ok=True)
        with self.manifest_jsonl.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_header = not self.manifest_csv.exists() or self.manifest_csv.stat().st_size == 0
        fields = list(rows[0].keys())
        with self.manifest_csv.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def write_sidecar(self, record: dict[str, Any], file_item: dict[str, str]) -> Path | None:
        rel = Path(file_item.get("subfolder") or "") / (file_item.get("filename") or "")
        media_path = self.output_dir / rel
        if not media_path.exists():
            return None
        sidecar = media_path.with_suffix(media_path.suffix + ".config.json")
        record = dict(record)
        record["media_path"] = str(media_path)
        record["sidecar_path"] = str(sidecar)
        if self.dry_run:
            print(f"[dry-run] sidecar {sidecar}")
            return sidecar
        sidecar.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return sidecar

    def process_history(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            history = http_json(self.base_url, "/history")
        except Exception as exc:
            print(f"[warn] history fetch failed: {exc}", file=sys.stderr)
            return rows
        if not isinstance(history, dict):
            return rows
        seen = set(self.state.get("history") or [])
        for prompt_id, record in history.items():
            if prompt_id in seen or not isinstance(record, dict):
                continue
            prompt_payload = record.get("prompt") or []
            workflow = prompt_payload[1] if isinstance(prompt_payload, list) and len(prompt_payload) > 1 else {}
            extra_data = prompt_payload[2] if isinstance(prompt_payload, list) and len(prompt_payload) > 2 else {}
            files = history_outputs(record)
            base_record = {
                "schema": "comfyui.provenance.v1",
                "source": "history",
                "created_at": utc_now(),
                "prompt_id": str(prompt_id),
                "base_url": self.base_url,
                "host": socket.gethostname(),
                "gpu": gpu_info(),
                "status": record.get("status") or {},
                "summary": summarize_workflow(workflow),
                "extra_data": extra_data,
                "prompt": workflow,
                "outputs": files,
                "acceleration_snapshots": read_accel_snapshots(self.output_dir),
            }
            for item in files:
                sidecar = self.write_sidecar(base_record, item)
                if sidecar is not None:
                    rows.append(manifest_row(base_record, item, sidecar, self.output_dir))
            seen.add(prompt_id)
            print(f"[history] {prompt_id}: {len(files)} file(s)")
        self.state["history"] = sorted(seen)
        return rows

    def process_filesystem_fallback(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen = set(self.state.get("files") or [])
        cutoff = 0.0 if self.backfill else (self.started - max(self.interval * 2, 10.0))
        snaps = read_accel_snapshots(self.output_dir)
        for media_path in sorted(self.output_dir.rglob("*")):
            if not media_path.is_file() or media_path.suffix.lower() not in MEDIA_EXTS:
                continue
            if "_provenance" in media_path.parts or media_path.name.endswith(".config.json"):
                continue
            try:
                stat = media_path.stat()
            except OSError:
                continue
            if stat.st_mtime < cutoff:
                continue
            key = f"{media_path.relative_to(self.output_dir)}|{stat.st_size}|{int(stat.st_mtime)}"
            sidecar = media_path.with_suffix(media_path.suffix + ".config.json")
            if key in seen and sidecar.exists():
                continue
            rel_parent = media_path.parent.relative_to(self.output_dir)
            item = {
                "node_id": "",
                "group": "videos" if media_path.suffix.lower() in VIDEO_EXTS else "images",
                "filename": media_path.name,
                "subfolder": "" if str(rel_parent) == "." else str(rel_parent),
                "type": "output",
            }
            record = {
                "schema": "comfyui.provenance.v1",
                "source": "filesystem_only_backfill" if self.backfill else "filesystem_only",
                "created_at": utc_now(),
                "prompt_id": "",
                "base_url": self.base_url,
                "host": socket.gethostname(),
                "gpu": gpu_info(),
                "status": {},
                "summary": {"params": {}, "references": [], "acceleration_nodes": [], "note": "ComfyUI history unavailable; config not recoverable for this file"},
                "extra_data": {},
                "prompt": {},
                "outputs": [item],
                "file_stat": {"size": stat.st_size, "mtime": stat.st_mtime},
                "acceleration_snapshots": snaps,
            }
            written = self.write_sidecar(record, item)
            if written is not None:
                rows.append(manifest_row(record, item, written, self.output_dir))
            seen.add(key)
            print(f"[fs] {media_path.relative_to(self.output_dir)}")
        self.state["files"] = sorted(seen)
        return rows

    def poll_once(self) -> None:
        rows = self.process_history()
        rows += self.process_filesystem_fallback()
        self.append_manifest(rows)
        self.save_state()
        if rows and not self.dry_run:
            print(f"[ok] wrote {len(rows)} manifest row(s)")
        elif rows:
            print(f"[dry-run] would write {len(rows)} manifest row(s)")

    def run(self) -> None:
        print(f"[start] provenance watcher base={self.base_url} output={self.output_dir} interval={self.interval}s dry_run={self.dry_run}")
        while True:
            self.poll_once()
            time.sleep(self.interval)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("COMFY_URL", "http://127.0.0.1:8189"))
    ap.add_argument("--output-dir", default="/root/ComfyUI-H3/output")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--backfill", action="store_true", help="also create filesystem-only sidecars for old media files")
    args = ap.parse_args()
    watcher = ProvenanceWatcher(args.base_url, Path(args.output_dir), args.interval, args.dry_run, args.backfill)
    if args.once:
        watcher.poll_once()
        return 0
    watcher.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
