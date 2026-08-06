# -*- coding: utf-8 -*-
"""构建三组对照工作流的 API 格式 prompt（同种子、同输入）"""
import json, urllib.request, copy, sys

BASE = "https://8189-cpod-1sqfx2anig0i.pod.compshare.cn"
SEED = 154504777461846

# ---------- 取节点定义 ----------
oi = json.loads(urllib.request.urlopen(BASE + "/object_info", timeout=60).read())

CONN_TYPES = {"MODEL","CLIP","VAE","AUDIO_VAE","IMAGE","LATENT","CONDITIONING","SAMPLER",
              "SIGMAS","GUIDER","NOISE","VIDEO","AUDIO","MASK","CONTROL_NET","UPSCALE_MODEL",
              "HOOKS","TIMESTEPS_RANGE","WANVIDEOSAMPLER","*"}
CONTROL_MODES = {"fixed","increment","decrement","randomize"}

def widget_input_names(cls):
    """按 object_info 顺序列出 widget 型输入名（含被转成连线的），跳过 autogrow 容器"""
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
    """UI 格式 -> API 格式"""
    nodes = {n["id"]: n for n in d["nodes"] if n.get("mode", 0) == 0 and n["type"] != "MarkdownNote"}
    links = {l[0]: l for l in d["links"]}
    out = {}
    for nid, n in nodes.items():
        cls = n["type"]
        inp = {}
        fed = set()
        for i in n.get("inputs", []):
            if i.get("link") is not None:
                l = links[i["link"]]
                inp[i["name"]] = [str(l[1]), l[2]]
                fed.add(i["name"])
        wv = list(n.get("widgets_values") or [])
        names_all, defaults = widget_input_names(cls)
        # 丢掉前端专用尾部 widget（如 control_after_generate / 上传按钮占位）
        while len(wv) > len(names_all) and wv and (wv[-1] in CONTROL_MODES or not isinstance(wv[-1], (int, float))):
            wv.pop()
        if len(wv) > len(names_all):
            print(f"[{label}] !! {cls}({nid}) widget数({len(wv)}) > 输入数({len(names_all)}): {wv} / {names_all}")
            continue
        for k, v in zip(names_all, wv):
            if k not in fed:          # 已转成连线的 widget 用连线值，忽略旧缓存值
                inp[k] = v
        for k in names_all[len(wv):]:  # 新版多出的可选 widget 用默认值补齐
            if k not in fed and k in defaults:
                inp[k] = defaults[k]
                print(f"[{label}] {cls}({nid}) 补默认值 {k}={defaults[k]!r}")
        out[str(nid)] = {"class_type": cls, "inputs": inp}
    return out

def fix_seed_and_prefix(d, prefix):
    for n in d["nodes"]:
        if n["type"] == "RandomNoise":
            n["widgets_values"] = [SEED, "fixed"]
        if n["type"] == "SaveVideo":
            w = n["widgets_values"]; w[0] = prefix

def strip_flashvsr(d):
    """去掉 FlashVSR 尾巴：VAEDecode(122) 直连 CreateVideo(130)"""
    d["nodes"] = [n for n in d["nodes"] if n["id"] != 206]
    nodes = {n["id"]: n for n in d["nodes"]}
    # 找到 206 的输入 link（来自122）和输出 link（去往130）
    in_link = out_link = None
    for l in d["links"]:
        if l[3] == 206: in_link = l
        if l[1] == 206: out_link = l
    d["links"] = [l for l in d["links"] if l[1] != 206 and l[3] != 206]
    # 新 link: 122 -> 130
    new_id = max(l[0] for l in d["links"]) + 1
    d["links"].append([new_id, 122, 0, 130, out_link[4], "IMAGE"])
    for i in nodes[130]["inputs"]:
        if i.get("link") == out_link[0]:
            i["link"] = new_id
    return d

def add_hyperstep(d):
    """在 BlockCache(205) 后插 NBH3HyperStep(207)"""
    nodes = {n["id"]: n for n in d["nodes"]}
    d["nodes"].append({
        "id": 207, "type": "NBH3HyperStep",
        "pos": [-1140, 5420], "size": [320, 90], "flags": {}, "order": 25, "mode": 0,
        "inputs": [{"name": "model", "type": "MODEL", "link": None}],
        "outputs": [{"name": "model", "type": "MODEL", "links": []}],
        "properties": {"Node name for S&R": "NBH3HyperStep"},
        "widgets_values": ["Turbo - Skip 36 Blocks"],
    })
    nodes = {n["id"]: n for n in d["nodes"]}
    new_id = max(l[0] for l in d["links"]) + 1
    for l in list(d["links"]):
        if l[1] == 205:  # 205 -> X 改为 207 -> X
            l[1] = 207
            nodes[207]["outputs"][0]["links"].append(l[0])
    d["links"].append([new_id, 205, 0, 207, 0, "MODEL"])
    nodes[207]["inputs"][0]["link"] = new_id
    return d

# ---------- 构建三组 ----------
base_c = json.load(open(r"E:\sp\chaofen.json", encoding="utf-8"))
base_a = json.load(open(r"E:\sp\222.json", encoding="utf-8"))

# A: HyperStep 单独用（222 原样，固定种子，改输出前缀）
a = copy.deepcopy(base_a); fix_seed_and_prefix(a, "abtest/A_hyperstep_only")

# C: Sage + BlockCache（超分流去 FlashVSR 尾巴）
c = strip_flashvsr(copy.deepcopy(base_c)); fix_seed_and_prefix(c, "abtest/C_sage_cache")

# B: Sage + BlockCache + HyperStep
b = add_hyperstep(strip_flashvsr(copy.deepcopy(base_c))); fix_seed_and_prefix(b, "abtest/B_all_three")

for label, wf in [("A", a), ("B", b), ("C", c)]:
    api = to_api(wf, label)
    json.dump(api, open(rf"E:\sp\ab_{label}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[{label}] 节点数={len(api)} 已写出 ab_{label}.json")
