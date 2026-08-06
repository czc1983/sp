# -*- coding: utf-8 -*-
"""排队一个对照工作流并等待完成，报告耗时与产物"""
import json, sys, time, urllib.request, urllib.error

BASE = "https://8189-cpod-1sqfx2anig0i.pod.compshare.cn"
label = sys.argv[1]
wait_s = int(sys.argv[2]) if len(sys.argv) > 2 else 270

api = json.load(open(rf"E:\sp\ab_{label}.json", encoding="utf-8"))

def post_prompt():
    body = json.dumps({"prompt": api}).encode()
    req = urllib.request.Request(BASE + "/prompt", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=60).read())

pid_file = rf"E:\sp\ab_{label}.pid"
import os
if os.path.exists(pid_file):
    pid = open(pid_file).read().strip()
else:
    r = post_prompt()
    pid = r["prompt_id"]
    open(pid_file, "w").write(pid)
    print(f"[{label}] 已排队 prompt_id={pid}")

t0 = time.time()
while time.time() - t0 < wait_s:
    try:
        h = json.loads(urllib.request.urlopen(BASE + "/history/" + pid, timeout=30).read())
    except urllib.error.URLError:
        time.sleep(5); continue
    if pid in h:
        rec = h[pid]
        msgs = rec.get("status", {}).get("messages", [])
        ts = {}
        for m in msgs:
            if m[0] == "execution_start": ts["start"] = m[1]["timestamp"]
            if m[0] == "execution_success": ts["end"] = m[1]["timestamp"]
        if rec.get("status", {}).get("completed") or "end" in ts:
            dur = (ts["end"] - ts["start"]) / 1000 if "start" in ts and "end" in ts else None
            print(f"[{label}] 完成! 执行耗时: {dur:.1f}s" if dur else f"[{label}] 完成!")
            outs = rec.get("outputs", {})
            for nid, o in outs.items():
                for v in o.get("videos", []) + o.get("gifs", []):
                    print(f"[{label}] 产物: {v.get('filename')} (subfolder={v.get('subfolder')})")
                for img in o.get("images", []):
                    print(f"[{label}] 图片: {img.get('filename')}")
            sys.exit(0)
        else:
            print(f"[{label}] 仍在执行... {int(time.time()-t0)}s")
    else:
        print(f"[{label}] 排队中... {int(time.time()-t0)}s")
    time.sleep(15)
print(f"[{label}] 本轮等待超时，任务仍在跑（prompt_id={pid}）")
