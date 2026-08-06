# -*- coding: utf-8 -*-
"""构建「超分·急速」工作流：Sage + HyperStep + FlashVSR 尾巴，上传到 pod"""
import json, copy, urllib.request, urllib.parse

base = json.load(open(r"E:\sp\chaofen.json", encoding="utf-8"))
nodes = {n["id"]: n for n in base["nodes"]}

HYPER_TEMPLATE = {
    "id": 207, "type": "NBH3HyperStep",
    "pos": [-1140, 5420], "size": [320, 90], "flags": {}, "order": 25, "mode": 0,
    "inputs": [{"name": "model", "type": "MODEL", "link": None}],
    "outputs": [{"name": "model", "type": "MODEL", "links": []}],
    "properties": {"Node name for S&R": "NBH3HyperStep"},
    "widgets_values": ["Turbo - Skip 36 Blocks"],
    "title": "NB H3 HyperStep (Turbo)",
}

# 1) 加 HyperStep 节点，删 BlockCache(205)
base["nodes"].append(copy.deepcopy(HYPER_TEMPLATE))
base["nodes"] = [n for n in base["nodes"] if n["id"] != 205]
nodes = {n["id"]: n for n in base["nodes"]}

# 2) 重连模型链：127 -> 204 -> 207 -> {124,126}
drop = [l[0] for l in base["links"]
        if l[5] == "MODEL" and l[1] in {127, 204, 205} and l[3] in {124, 126, 204, 205}]
base["links"] = [l for l in base["links"] if l[0] not in drop]
for n in base["nodes"]:
    for i in n.get("inputs", []):
        if i.get("link") in drop:
            i["link"] = None
    for o in n.get("outputs", []):
        if o.get("links"):
            o["links"] = [x for x in o["links"] if x not in drop]

nid = max(l[0] for l in base["links"]) + 1
def connect(src, dst):
    global nid
    base["links"].append([nid, src, 0, dst, 0, "MODEL"])
    for i in nodes[dst]["inputs"]:
        if i["name"] == "model":
            i["link"] = nid
    o = nodes[src]["outputs"][0]
    o["links"] = (o.get("links") or []) + [nid]
    nid += 1

connect(127, 204)   # UNETLoader -> Sage
connect(204, 207)   # Sage -> HyperStep
connect(207, 124)   # HyperStep -> BasicScheduler
connect(207, 126)   # HyperStep -> BasicGuider

# 3) 种子恢复随机
for n in base["nodes"]:
    if n["type"] == "RandomNoise":
        n["widgets_values"] = [154504777461846, "randomize"]

# 4) 校验：FlashVSR 尾巴还在，链路完整
cls = [n["type"] for n in base["nodes"]]
assert "FlashVSRNode" in cls and "MiniMaxH3BlockCacheT8" not in cls
model_links = [(l[1], l[3]) for l in base["links"] if l[5] == "MODEL"]
print("MODEL 链路:", model_links)
print("节点数:", len(base["nodes"]))

out = r"E:\sp\chaofen_jisu.json"
json.dump(base, open(out, "w", encoding="utf-8"), ensure_ascii=False)
print("已写出", out)

# 5) 上传到 pod
BASE = "https://8189-cpod-1sqfx2anig0i.pod.compshare.cn"
path = urllib.parse.quote("workflows/超分·急速.json")
req = urllib.request.Request(
    f"{BASE}/userdata/{path}?overwrite=true&full_info=true",
    data=open(out, "rb").read(),
    headers={"Content-Type": "application/json"}, method="POST")
print("上传状态:", urllib.request.urlopen(req, timeout=60).status)
lst = json.loads(urllib.request.urlopen(BASE + "/userdata?dir=workflows", timeout=30).read())
print("工作流列表:", [x.get("path") for x in lst] if isinstance(lst, list) else lst)
