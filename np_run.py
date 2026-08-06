# -*- coding: utf-8 -*-
"""新机器(5090)测速：转换两个完整工作流(含FlashVSR尾)为API格式，排队并等待"""
import json, sys, time, os, urllib.request, urllib.error

BASE = "https://8189-cpod-1tpdn4punkor-s1.pod.compshare.cn"
SEED = 154504777461846
CONN_TYPES = {"MODEL","CLIP","VAE","AUDIO_VAE","IMAGE","LATENT","CONDITIONING","SAMPLER",
              "SIGMAS","GUIDER","NOISE","VIDEO","AUDIO","MASK","CONTROL_NET","UPSCALE_MODEL",
              "HOOKS","TIMESTEPS_RANGE","*"}
CONTROL_MODES = {"fixed","increment","decrement","randomize"}

# 1) 等服务上线 + 验证节点
oi = None
for i in range(30):
    try:
        oi = json.loads(urllib.request.urlopen(BASE + "/object_info", timeout=10).read())
        break
    except Exception:
        time.sleep(10)
        print(f"等待启动... {(i+1)*10}s")
if oi is None:
    raise SystemExit("服务未上线")
for t in ["MiniMaxH3MemoryEfficientSageAttentionPatch", "MiniMaxH3BlockCacheT8", "NBH3HyperStep", "FlashVSRNode"]:
    print(f"节点 {t}: {'OK' if t in oi else '缺失!!'}")

def widget_input_names(cls):
    spec = oi[cls]["input"]
    names, defaults = [], {}
    for grp in ("required", "optional"):
        for name, meta in spec.get(grp, {}).items():
            t = meta[0]
            if t == "COMFY_AUTOGROW_V3": continue
            if isinstance(t, list) or t not in CONN_TYPES:
                names.append(name)
                if len(meta) > 1 and isinstance(meta[1], dict) and "default" in meta[1]:
                    defaults[name] = meta[1]["default"]
    return names, defaults

def to_api(d, label):
    nodes = {n["id"]: n for n in d["nodes"] if n.get("mode", 0) == 0 and n["type"] != "MarkdownNote"}
    links = {l[0]: l for l in d["links"]}
    out = {}
    for nid, n in nodes.items():
        cls = n["type"]
        if cls not in oi:
            print(f"[{label}] !! 新机器缺少节点类型 {cls}")
            continue
        inp, fed = {}, set()
        for i in n.get("inputs", []):
            if i.get("link") is not None:
                l = links[i["link"]]
                inp[i["name"]] = [str(l[1]), l[2]]
                fed.add(i["name"])
        wv = list(n.get("widgets_values") or [])
        names_all, defaults = widget_input_names(cls)
        while len(wv) > len(names_all) and wv and (wv[-1] in CONTROL_MODES or not isinstance(wv[-1], (int, float))):
            wv.pop()
        if len(wv) > len(names_all):
            print(f"[{label}] !! {cls}({nid}) widget过多: {wv} / {names_all}")
            continue
        for k, v in zip(names_all, wv):
            if k not in fed:
                inp[k] = v
        for k in names_all[len(wv):]:
            if k not in fed and k in defaults:
                inp[k] = defaults[k]
        out[str(nid)] = {"class_type": cls, "inputs": inp}
    return out

def prep(src, label, prefix):
    d = json.load(open(src, encoding="utf-8"))
    for n in d["nodes"]:
        if n["type"] == "RandomNoise":
            n["widgets_values"] = [SEED, "fixed"]
        if n["type"] == "SaveVideo":
            n["widgets_values"][0] = prefix
    api = to_api(d, label)
    json.dump(api, open(rf"E:\sp\np_{label}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[{label}] 节点数={len(api)} 已写出 np_{label}.json")
    return api

jobs = {
    "N1": prep(r"E:\sp\chaofen.json", "N1", "abtest/N1_chaofen_5090"),        # 超分: Sage+Cache+FlashVSR
    "N2": prep(r"E:\sp\chaofen_jisu.json", "N2", "abtest/N2_jisu_5090"),      # 超分·急速: Sage+Hyper+FlashVSR
}

# 2) 排队
pids = {}
for L, api in jobs.items():
    f = rf"E:\sp\np_{L}.pid"
    if os.path.exists(f):
        pids[L] = open(f).read().strip()
    else:
        body = json.dumps({"prompt": api}).encode()
        req = urllib.request.Request(BASE + "/prompt", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        pids[L] = json.loads(urllib.request.urlopen(req, timeout=60).read())["prompt_id"]
        open(f, "w").write(pids[L])
        print(f"[{L}] 已排队 {pids[L]}")
        time.sleep(1)

# 3) 等待
wait_s = int(sys.argv[1]) if len(sys.argv) > 1 else 270
done = {}
t0 = time.time()
while time.time() - t0 < wait_s and len(done) < len(pids):
    for L, pid in pids.items():
        if L in done: continue
        try:
            h = json.loads(urllib.request.urlopen(BASE + "/history/" + pid, timeout=30).read())
        except urllib.error.URLError:
            continue
        if pid in h:
            rec = h[pid]
            msgs = rec.get("status", {}).get("messages", [])
            ts = {}
            for m in msgs:
                if m[0] == "execution_start": ts["s"] = m[1]["timestamp"]
                if m[0] == "execution_success": ts["e"] = m[1]["timestamp"]
            if rec.get("status", {}).get("completed") or "e" in ts:
                dur = (ts["e"] - ts["s"]) / 1000 if "s" in ts and "e" in ts else -1
                files = [v.get("filename") for nid, o in rec.get("outputs", {}).items()
                         for v in o.get("videos", []) + o.get("images", []) + o.get("gifs", [])]
                done[L] = dur
                print(f"[{L}] 完成! 耗时 {dur:.1f}s 产物 {files}")
    if len(done) < len(pids):
        time.sleep(20)
print(f"进度 {len(done)}/{len(pids)}，已等待 {int(time.time()-t0)}s")
