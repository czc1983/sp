from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
OUT_LOG = ROOT / ".web_restart_current.out.log"
ERR_LOG = ROOT / ".web_restart_current.err.log"
ALIAS_OUT_LOG = ROOT / ".web_restart_alias.out.log"
ALIAS_ERR_LOG = ROOT / ".web_restart_alias.err.log"


def _run_powershell(script: str) -> str:
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return (completed.stdout or "") + (completed.stderr or "")


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.35):
            return True
    except OSError:
        return False


def _stop_existing(port: int) -> None:
    ps = rf"""
$ErrorActionPreference = 'SilentlyContinue'
$port = {int(port)}
$pids = @()
$pids += Get-NetTCPConnection -LocalPort $port -State Listen | Select-Object -ExpandProperty OwningProcess
$pids += Get-CimInstance Win32_Process |
  Where-Object {{ $_.CommandLine -match 'start_web\.py' -or $_.CommandLine -match 'web_ui\.server' }} |
  Select-Object -ExpandProperty ProcessId
$pids | Sort-Object -Unique | ForEach-Object {{
  if ($_ -and $_ -ne $PID) {{
    Stop-Process -Id $_ -Force
  }}
}}
"""
    _run_powershell(ps)
    deadline = time.time() + 5
    while time.time() < deadline:
        if not _port_open("127.0.0.1", port):
            return
        time.sleep(0.2)


def _start_server(port: int) -> subprocess.Popen:
    OUT_LOG.write_text("", encoding="utf-8")
    ERR_LOG.write_text("", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    code = (
        "from web_ui.server import run_server; "
        f"run_server(host='127.0.0.1', port={int(port)}, open_browser=False)"
    )
    out_handle = OUT_LOG.open("a", encoding="utf-8", errors="replace")
    err_handle = ERR_LOG.open("a", encoding="utf-8", errors="replace")
    return subprocess.Popen(
        [str(PYTHON), "-u", "-c", code],
        cwd=str(ROOT),
        stdout=out_handle,
        stderr=err_handle,
        env=env,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
    )


def _start_redirect_alias(alias_port: int, target_port: int) -> subprocess.Popen:
    ALIAS_OUT_LOG.write_text("", encoding="utf-8")
    ALIAS_ERR_LOG.write_text("", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    code = f"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TARGET = "http://127.0.0.1:{int(target_port)}"

class RedirectHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(302)
        self.send_header("Location", TARGET + self.path)
        self.end_headers()

    def do_POST(self):
        self.send_response(307)
        self.send_header("Location", TARGET + self.path)
        self.end_headers()

    def log_message(self, format, *args):
        return

print("[alias] http://127.0.0.1:{int(alias_port)}/ -> " + TARGET + "/", flush=True)
ThreadingHTTPServer(("127.0.0.1", {int(alias_port)}), RedirectHandler).serve_forever()
"""
    out_handle = ALIAS_OUT_LOG.open("a", encoding="utf-8", errors="replace")
    err_handle = ALIAS_ERR_LOG.open("a", encoding="utf-8", errors="replace")
    return subprocess.Popen(
        [str(PYTHON), "-u", "-c", code],
        cwd=str(ROOT),
        stdout=out_handle,
        stderr=err_handle,
        env=env,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
    )


def _wait_ready(port: int, timeout_seconds: float) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _port_open("127.0.0.1", port):
            return True
        time.sleep(0.3)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Restart the SP web backend and open the UI.")
    parser.add_argument("--port", type=int, default=7861, help="Local web port. Default: 7861")
    parser.add_argument("--alias-port", type=int, default=11422, help="Redirect alias port. Default: 11422")
    parser.add_argument("--no-alias", action="store_true", help="Do not start the 11422 redirect alias.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser.")
    args = parser.parse_args()

    port = max(1, min(int(args.port), 65535))
    alias_port = max(1, min(int(args.alias_port), 65535))
    print(f"[restart] root: {ROOT}")
    print(f"[restart] stopping old server on port {port}...")
    _stop_existing(port)
    if not args.no_alias and alias_port != port:
        print(f"[restart] stopping old alias on port {alias_port}...")
        _stop_existing(alias_port)

    print(f"[restart] starting server on http://127.0.0.1:{port}/ ...")
    process = _start_server(port)
    if not _wait_ready(port, 25):
        print("[restart] failed: server did not open the port.")
        print(f"[restart] stdout log: {OUT_LOG}")
        print(f"[restart] stderr log: {ERR_LOG}")
        try:
            print(ERR_LOG.read_text(encoding="utf-8", errors="replace")[-4000:])
        except OSError:
            pass
        return 1

    alias_process = None
    if not args.no_alias and alias_port != port:
        print(f"[restart] starting alias http://127.0.0.1:{alias_port}/ -> http://127.0.0.1:{port}/ ...")
        alias_process = _start_redirect_alias(alias_port, port)
        if not _wait_ready(alias_port, 8):
            print("[restart] warning: alias port did not open; main server is still ready.")

    url = f"http://127.0.0.1:{port}/"
    print(f"[restart] ok: pid={process.pid}")
    print(f"[restart] open: {url}")
    if alias_process is not None:
        print(f"[restart] alias: http://127.0.0.1:{alias_port}/ pid={alias_process.pid}")
    print(f"[restart] stdout log: {OUT_LOG}")
    print(f"[restart] stderr log: {ERR_LOG}")
    if not args.no_browser:
        webbrowser.open(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
