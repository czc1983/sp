# -*- coding: utf-8 -*-
"""构建剩余 5 组对照工作流：D=Sage E=Cache F=Sage+HyperStep G=Cache+HyperStep H=裸基线"""
import json, urllib.request, copy

BASE = "https://8189-cpod-1sqfx2anig0i.pod.compshare.cn"
SEED = 154504777461846
oi = json.loads(urllib.request.urlopen(BASE + "/object_info", timeout=60).read())

CONN_TYPES = {"MODEL","CLIP","VAE","AUDIO_VAE","IMAGE","LATENT","CONDITIONING","SAMPLER",
              "SIGMAS","GUIDER","NOISE","VIDEO","AUDIO","MASK","CONTROL_NET","UPSCALE_MODEL",
              "HOOKS","TIMESTEPS_RANGE","*"}
CONTROL_MODES = {"fixed","increment","decrement","randomize"}

def widget_input_names(cls):
    spec = oi[cls]["input"]
    names, defaults = [], {}
    for grp in ("required", "optional"):
        for name, meta in spec.get(grp, {}).items():
            t = meta[0]
            if t == "COMFY_AUTOGROW_V3":
                continue
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
            print(f"[{label}] !! {cls}({nid}) widget数({len(wv)}) > 输入数({len(names_all)}): {wv} / {names_all}")
            continue
        for k, v in zip(names_all, wv):
            if k not in fed:
                inp[k] = v
        for k in names_all[len(wv):]:
            if k not in fed and k in defaults:
                inp[k] = defaults[k]
        out[str(nid)] = {"class_type": cls, "inputs": inp}
    return out

def fix_seed_and_prefix(d, prefix):
    for n in d["nodes"]:
        if n["type"] == "RandomNoise":
            n["widgets_values"] = [SEED, "fixed"]
        if n["type"] == "SaveVideo":
            n["widgets_values"][0] = prefix

def strip_flashvsr(d):
    d["nodes"] = [n for n in d["nodes"] if n["id"] != 206]
    nodes = {n["id"]: n for n in d["nodes"]}
    out_link = None
    for l in d["links"]:
        if l[1] == 206: out_link = l
    d["links"] = [l for l in d["links"] if l[1] != 206 and l[3] != 206]
    new_id = max(l[0] for l in d["links"]) + 1
    d["links"].append([new_id, 122, 0, 130, out_link[4], "IMAGE"])
    for i in nodes[130]["inputs"]:
        if i.get("link") == out_link[0]:
            i["link"] = new_id
    return d

HYPER_TEMPLATE = {
    "id": 207, "type": "NBH3HyperStep",
    "pos": [-1140, 5420], "size": [320, 90], "flags": {}, "order": 25, "mode": 0,
    "inputs": [{"name": "model", "type": "MODEL", "link": None}],
    "outputs": [{"name": "model", "type": "MODEL", "links": []}],
    "properties": {"Node name for S&R": "NBH3HyperStep"},
    "widgets_values": ["Turbo - Skip 36 Blocks"],
}
PATCH_IDS = {"sage": 204, "cache": 205, "hyper": 207}
CONSUMERS = [124, 126]  # BasicScheduler, BasicGuider

def build_variant(chain, label, prefix):
    """chain: 127 之后的补丁节点链, 如 ['sage','hyper']"""
    d = strip_flashvsr(copy.deepcopy(base))
    fix_seed_and_prefix(d, prefix)
    nodes = {n["id"]: n for n in d["nodes"]}
    keep = {PATCH_IDS[c] for c in chain}
    # 加缺失的 HyperStep 节点
    if "hyper" in chain and 207 not in nodes:
        d["nodes"].append(copy.deepcopy(HYPER_TEMPLATE))
        nodes = {n["id"]: n for n in d["nodes"]}
    # 删掉不用的补丁节点
    d["nodes"] = [n for n in d["nodes"] if n["id"] not in ({204, 205, 207} - keep)]
    nodes = {n["id"]: n for n in d["nodes"]}
    # 清除旧的模型链连线（127/204/205/207 -> 204/205/207/124/126 的 MODEL link）
    chain_set = {127, 204, 205, 207} | set(CONSUMERS)
    drop = [l[0] for l in d["links"]
            if l[5] == "MODEL" and l[1] in {127, 204, 205, 207} and l[3] in chain_set]
    d["links"] = [l for l in d["links"] if l[0] not in drop]
    for n in d["nodes"]:
        for i in n.get("inputs", []):
            if i.get("link") in drop:
                i["link"] = None
        for o in n.get("outputs", []):
            if o.get("links"):
                o["links"] = [x for x in o["links"] if x not in drop]
    # 重连：127 -> patch1 -> ... -> {124,126}
    seq = [127] + [PATCH_IDS[c] for c in chain]
    nid = max(l[0] for l in d["links"]) + 1
    def connect(src, dst, iname):
        nonlocal nid
        d["links"].append([nid, src, 0, dst, 0, "MODEL"])
        for i in nodes[dst]["inputs"]:
            if i["name"] == iname:
                i["link"] = nid
        nodes[src]["outputs"][0]["links"] = (nodes[src]["outputs"][0].get("links") or []) + [nid]
        nid += 1
    for a, b in zip(seq, seq[1:]):
        connect(a, b, "model")
    for c in CONSUMERS:
        connect(seq[-1], c, "model")
    api = to_api(d, label)
    json.dump(api, open(rf"E:\sp\ab_{label}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    cls = [v["class_type"] for v in api.values()]
    print(f"[{label}] 节点数={len(api)} Sage={'MiniMaxH3MemoryEfficientSageAttentionPatch' in cls} "
          f"Cache={'MiniMaxH3BlockCacheT8' in cls} Hyper={'NBH3HyperStep' in cls} "
          f"Flash={'FlashVSRNode' in cls}")

base = json.load(open(r"E:\sp\chaofen.json", encoding="utf-8"))
build_variant(["sage"], "D", "abtest/D_sage_only")
build_variant(["cache"], "E", "abtest/E_cache_only")
build_variant(["sage", "hyper"], "F", "abtest/F_sage_hyper")
build_variant(["cache", "hyper"], "G", "abtest/G_cache_hyper")
build_variant([], "H", "abtest/H_baseline")
