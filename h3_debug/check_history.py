# -*- coding: utf-8 -*-
"""服务器端：批量查询 8189 history，输出 JSON pid -> {status, file}"""
import json
import sys
import urllib.request

out = {}
for pid in sys.argv[1:]:
    try:
        d = json.load(urllib.request.urlopen(f"http://127.0.0.1:8189/history/{pid}", timeout=30))
    except Exception as e:
        out[pid] = {"status": "error", "err": str(e)[:100]}
        continue
    if not d:
        out[pid] = {"status": "pending"}
        continue
    h = d.get(pid, {})
    st = h.get("status", {})
    done = bool(st.get("completed")) or st.get("status_str") == "success"
    fname = None
    for node_out in (h.get("outputs") or {}).values():
        for key in ("videos", "gifs", "images"):
            for item in node_out.get(key, []) or []:
                fn = item.get("filename", "")
                if fn.endswith(".mp4"):
                    sub = item.get("subfolder") or ""
                    fname = f"{sub}/{fn}" if sub else fn
    out[pid] = {"status": "done" if done else "running", "file": fname}
print(json.dumps(out))
