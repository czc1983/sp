"""Convert ComfyUI GUI workflow JSON to API prompt format and queue it."""
import json
import urllib.request
import urllib.parse

BASE = "https://8189-cpod-1sqfx2anig0i.pod.compshare.cn"
WF_PATH = r"E:\sp\h3_sage_easycache_hyperstep.json"


def fetch(path):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=60).read())


def gui_to_api(wf, obj_info):
    # link_id -> (src_node_id, src_output_slot)
    links = {l[0]: (l[1], l[2]) for l in wf["links"]}
    nodes = {n["id"]: n for n in wf["nodes"]}
    prompt = {}

    for nid, node in nodes.items():
        ctype = node["type"]
        if ctype not in obj_info:
            continue  # frontend-only nodes like MarkdownNote
        info = obj_info[ctype]
        inputs = {}

        # 1) connected socket inputs
        connected_widget_inputs = set()
        for slot, inp in enumerate(node.get("inputs", [])):
            if inp.get("link") is not None:
                src = links[inp["link"]]
                inputs[inp["name"]] = [str(src[0]), src[1]]
                if inp.get("widget"):
                    connected_widget_inputs.add(inp["widget"]["name"])

        # 2) widget values, mapped in object_info order
        # order = required then optional, skipping pure socket inputs
        ordered = []
        for section in ("required", "optional"):
            for name, spec in info["input"].get(section, {}).items():
                ordered.append((name, spec))

        widget_names = []
        for name, spec in ordered:
            is_socket = isinstance(spec[0], str) and spec[0].isupper() and spec[0] not in ("STRING", "INT", "FLOAT", "BOOLEAN", "COMBO")
            # socket types are like MODEL/CLIP/IMAGE; COMBO/primitive are widgets
            if isinstance(spec[0], list) or spec[0] in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"):
                widget_names.append(name)

        wvalues = list(node.get("widgets_values", []))
        # handle dict-style trailing widget values (rare)
        extra = {}
        if wvalues and isinstance(wvalues[-1], dict):
            extra = wvalues.pop()
        for name, val in zip(widget_names, wvalues):
            if name in connected_widget_inputs:
                continue  # provided by link
            inputs[name] = val
        inputs.update(extra)

        prompt[str(nid)] = {"class_type": ctype, "inputs": inputs}
    return prompt


def main():
    wf = json.load(open(WF_PATH, encoding="utf-8"))
    obj_info = fetch("/object_info")
    prompt = gui_to_api(wf, obj_info)
    with open(r"E:\sp\api_prompt_debug.json", "w", encoding="utf-8") as f:
        json.dump(prompt, f, ensure_ascii=False, indent=1)

    body = json.dumps({"prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(
        BASE + "/prompt", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
        print("QUEUED:", json.dumps(resp))
    except urllib.error.HTTPError as e:
        print("REJECTED:", e.code)
        print(e.read().decode("utf-8", errors="replace")[:2000])


if __name__ == "__main__":
    main()
