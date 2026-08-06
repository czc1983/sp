# -*- coding: utf-8 -*-
"""排队 D~H 五组并等待全部完成"""
import json, sys, time, os, urllib.request, urllib.error

BASE = "https://8189-cpod-1sqfx2anig0i.pod.compshare.cn"
LABELS = ["D", "E", "F", "G", "H"]
wait_s = int(sys.argv[1]) if len(sys.argv) > 1 else 270

def queue(label):
    api = json.load(open(rf"E:\sp\ab_{label}.json", encoding="utf-8"))
    body = json.dumps({"prompt": api}).encode()
    req = urllib.request.Request(BASE + "/prompt", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=60).read())["prompt_id"]

pids = {}
for L in LABELS:
    f = rf"E:\sp\ab_{L}.pid"
    if os.path.exists(f):
        pids[L] = open(f).read().strip()
    else:
        pids[L] = queue(L)
        open(f, "w").write(pids[L])
        print(f"[{L}] 已排队 {pids[L]}")
        time.sleep(1)

done = {}
t0 = time.time()
while time.time() - t0 < wait_s and len(done) < len(LABELS):
    for L in LABELS:
        if L in done:
            continue
        try:
            h = json.loads(urllib.request.urlopen(BASE + "/history/" + pids[L], timeout=30).read())
        except urllib.error.URLError:
            continue
        if pids[L] in h:
            rec = h[pids[L]]
            msgs = rec.get("status", {}).get("messages", [])
            ts = {}
            for m in msgs:
                if m[0] == "execution_start": ts["s"] = m[1]["timestamp"]
                if m[0] == "execution_success": ts["e"] = m[1]["timestamp"]
            if rec.get("status", {}).get("completed") or "e" in ts:
                dur = (ts["e"] - ts["s"]) / 1000 if "s" in ts and "e" in ts else -1
                files = []
                for nid, o in rec.get("outputs", {}).items():
                    for v in o.get("videos", []) + o.get("images", []) + o.get("gifs", []):
                        files.append(v.get("filename"))
                done[L] = dur
                print(f"[{L}] 完成! 耗时 {dur:.1f}s 产物 {files}")
    if len(done) < len(LABELS):
        time.sleep(20)
print(f"进度: {len(done)}/{len(LABELS)} 完成, 已等待 {int(time.time()-t0)}s")
