"""SP Web 后端启动器（自带旧进程清理）。

在 VSCode 里直接运行本文件即可：启动前自动结束所有仍在监听 7861 端口
的旧后端进程，避免 Windows 下多进程共享端口导致「开了新实例却用的旧代码」。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time

PORT = 7861


def _find_listeners(port: int) -> set[int]:
    """返回正在监听指定端口的 PID 集合（排除本进程）。"""
    try:
        output = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except Exception:
        return set()
    pids: set[int] = set()
    for line in output.splitlines():
        if "LISTENING" not in line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        local_addr = parts[1]  # 形如 127.0.0.1:7861 或 [::1]:7861
        if not local_addr.endswith(f":{port}"):
            continue
        try:
            pid = int(parts[-1])
        except (ValueError, IndexError):
            continue
        if pid > 0 and pid != os.getpid():
            pids.add(pid)
    return pids


def kill_old_servers(port: int = PORT) -> None:
    pids = _find_listeners(port)
    if not pids:
        return
    print(f"[start_web] 发现 {len(pids)} 个旧后端进程仍占用 {port} 端口: {sorted(pids)}")
    for pid in pids:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F"],
                capture_output=True, timeout=15,
            )
            print(f"[start_web] 已结束旧进程 PID={pid}")
        except Exception as exc:  # noqa: BLE001
            print(f"[start_web] 结束 PID={pid} 失败: {exc}")
    # 等端口真正释放
    for _ in range(20):
        if not _find_listeners(port):
            break
        time.sleep(0.3)
    else:
        print(f"[start_web] 警告: {port} 端口仍被占用，新实例可能与旧进程抢端口！")


if __name__ == "__main__":
    kill_old_servers(PORT)
    from web_ui.server import run_server

    run_server()
