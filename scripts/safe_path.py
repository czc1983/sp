#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=True))


def _resolve_first(matches: list[Path], *, must_exist: bool) -> Path:
    if not matches:
        raise FileNotFoundError("no matching path")
    existing = [path for path in matches if path.exists()]
    if must_exist and not existing:
        raise FileNotFoundError(str(matches[0]))
    return (existing or matches)[-1].resolve()


def _find(root: Path, name: str, *, must_exist: bool) -> Path:
    if root.is_file():
        candidates = [root] if root.name == name else []
    else:
        candidates = list(root.rglob(name))
    return _resolve_first(candidates, must_exist=must_exist)


def _json_get(data: Any, key: str) -> Any:
    current = data
    for part in key.split("."):
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            raise KeyError(key)
    return current


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve Windows Unicode paths safely.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    find_p = sub.add_parser("find")
    find_p.add_argument("--root", required=True)
    find_p.add_argument("--name", required=True)
    find_p.add_argument("--allow-missing", action="store_true")

    json_p = sub.add_parser("json-get")
    json_p.add_argument("--json", required=True)
    json_p.add_argument("--key", required=True)
    json_p.add_argument("--root")
    json_p.add_argument("--basename-search", action="store_true")

    args = parser.parse_args()

    if args.cmd == "find":
        path = _find(Path(args.root), args.name, must_exist=not args.allow_missing)
        _emit({"path": str(path), "exists": path.exists()})
        return 0

    if args.cmd == "json-get":
        data = json.loads(Path(args.json).read_text(encoding="utf-8"))
        value = _json_get(data, args.key)
        if isinstance(value, str) and args.basename_search:
            root = Path(args.root) if args.root else Path.home()
            path = _find(root, Path(value).name, must_exist=True)
            _emit({"path": str(path), "raw": value, "exists": path.exists()})
        else:
            path = Path(value) if isinstance(value, str) else None
            _emit({"path": str(path) if path else value, "exists": path.exists() if path else None})
        return 0

    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"error": str(exc), "type": type(exc).__name__}, ensure_ascii=True), file=sys.stderr)
        raise SystemExit(1)
