#!/usr/bin/env python
"""给 MiniMax H3 工作流插入 SolAttnPatch 节点（kijai ComfyUI-SolAttn_triton）。

同时支持两种格式：
- API prompt 格式（h3_whitemask_api.json / h3_charswap_api.json）
- ComfyUI 0.30 UI 格式（h3_whitemask_r2v.json）

做法：UNETLoader 的 MODEL 输出原来直连 BasicScheduler / BasicGuider，
现在改为 UNETLoader → SolAttnPatch → BasicScheduler / BasicGuider。
幂等：已经插过 SolAttnPatch 的文件直接跳过。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SOLATTN_DEFAULTS = {
    "tau": 1.2,                # 阈值，越大越稀疏越快；1.0~1.5 质量稳
    "start_percent": 0.2,      # 前 20% 步数保持 dense 预热（论文设置）
    "end_percent": 0.9,
    "min_tokens": 4096,
    "int8_qk": False,
    "sink_conditioning": "exact_kv",  # H3 打包条件行精确可见，防提示词漂移
    "morton": False,           # 先保守不开；要更快可开并配合 2d_frame
    "morton_curve": "2d_frame",  # H3 帧间隔非均匀，官方建议 2d_frame
    "verbose": False,
}


def patch_api_format(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    if any(n.get("class_type") == "SolAttnPatch" for n in data.values()):
        print(f"[skip] {path.name} 已有 SolAttnPatch")
        return False
    # 找 UNETLoader 及其 model 消费者
    unet_id = next(k for k, n in data.items() if n.get("class_type") == "UNETLoader")
    new_id = str(max(int(k) for k in data) + 1)
    data[new_id] = {
        "class_type": "SolAttnPatch",
        "inputs": {"model": [unet_id, 0], **SOLATTN_DEFAULTS},
        "_meta": {"title": "Patch Sol-Attn"},
    }
    count = 0
    for nid, n in data.items():
        if nid == new_id:
            continue  # 别改接 SolAttnPatch 自己的 model 输入，否则成环
        for key, val in n.get("inputs", {}).items():
            if isinstance(val, list) and val[0] == unet_id:
                n["inputs"][key] = [new_id, 0]
                count += 1
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] {path.name}: 插入节点 {new_id}，改接 {count} 处 model 输入")
    return True


def patch_ui_format(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    if any(n.get("type") == "SolAttnPatch" for n in data["nodes"]):
        print(f"[skip] {path.name} 已有 SolAttnPatch")
        return False
    nodes = {n["id"]: n for n in data["nodes"]}
    links = data["links"]  # [link_id, src_node, src_slot, dst_node, dst_slot, type]
    unet = next(n for n in data["nodes"] if n.get("type") == "UNETLoader")
    # UNETLoader 输出 0 上现有的 model 连线
    model_links = [l for l in links if l[1] == unet["id"] and l[2] == 0]
    if not model_links:
        raise RuntimeError(f"{path.name}: UNETLoader 没有 model 输出连线")
    new_node_id = data["last_node_id"] + 1
    next_link = data["last_link_id"] + 1
    # 新节点：SolAttnPatch
    patch_node = {
        "id": new_node_id,
        "type": "SolAttnPatch",
        "pos": [unet["pos"][0] + 300, unet["pos"][1]],
        "size": [270, 250],
        "flags": {},
        "order": max(n.get("order", 0) for n in data["nodes"]) + 1,
        "mode": 0,
        "inputs": [
            {"name": "model", "type": "MODEL", "link": None},
            *[{"name": k, "type": "WIDGET", "widget": {"name": k}, "link": None}
              for k in SOLATTN_DEFAULTS],
        ],
        "outputs": [{"name": "MODEL", "type": "MODEL", "links": []}],
        "properties": {"Node name for S&R": "SolAttnPatch"},
        "widgets_values": list(SOLATTN_DEFAULTS.values()),
    }
    # UNETLoader → patch.model
    in_link_id = next_link
    next_link += 1
    links.append([in_link_id, unet["id"], 0, new_node_id, 0, "MODEL"])
    patch_node["inputs"][0]["link"] = in_link_id
    # patch.MODEL → 原来的消费者
    for old in model_links:
        out_link_id = next_link
        next_link += 1
        links.append([out_link_id, new_node_id, 0, old[3], old[4], "MODEL"])
        patch_node["outputs"][0]["links"].append(out_link_id)
        # 更新目标节点的 input link
        dst = nodes[old[3]]
        dst["inputs"][old[4]]["link"] = out_link_id
        links.remove(old)
    # UNETLoader 输出槽的 links 列表更新
    unet_out = unet["outputs"][0]
    unet_out["links"] = [in_link_id if l in {ol[0] for ol in model_links} else l
                         for l in (unet_out.get("links") or [])]
    data["nodes"].append(patch_node)
    data["last_node_id"] = new_node_id
    data["last_link_id"] = next_link - 1
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] {path.name}: 插入节点 {new_node_id}，改接 {len(model_links)} 条 model 连线")
    return True


def main() -> None:
    root = Path(__file__).resolve().parent.parent / "comfy_workflows"
    for name, fmt in [
        ("h3_whitemask_api.json", "api"),
        ("h3_charswap_api.json", "api"),
        ("h3_whitemask_r2v.json", "ui"),
    ]:
        path = root / name
        if fmt == "api":
            patch_api_format(path)
        else:
            patch_ui_format(path)


if __name__ == "__main__":
    sys.exit(main())
