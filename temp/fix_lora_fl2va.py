# -*- coding: utf-8 -*-
"""Fix lora-fl2va.json on pod: rewire image to first_frame, set task_type=I2VA, update prompt."""
import json
import shutil

PATH = "/root/ComfyUI-H3/user/default/workflows/lora-fl2va.json"
BACKUP = PATH + ".bak_kimi"

PROMPT = (
    "integrated_multimodal_description: <Picture 1> 中的人物在同一位置开始欢快地跳舞："
    "从静止站姿自然起步，手臂随节奏摆动，身体轻快律动，脚步踏出明快的舞步，"
    "动作连贯有节奏感；镜头保持固定，人物外观、服装、发型与首帧完全一致，"
    "背景、道具与光影保持首帧原样，不出现新人物、不切换场景、不加任何文字水印。"
    "\n\n"
    "overall_soundscape: 室内环境安静，舞步踩踏地板的声音与音乐节拍同步，衣料摩擦声轻微自然。"
    "\n\n"
    "non_diegetic_music: 活力四射的流行舞曲，节奏明快，贯穿全片。"
)

with open(PATH, encoding="utf-8") as f:
    d = json.load(f)

shutil.copyfile(PATH, BACKUP)

cond = None
for n in d["nodes"]:
    if "AudioConditioning" in str(n.get("type", "")):
        cond = n
        break
assert cond is not None, "conditioning node not found"

inputs = cond.get("inputs") or []
by_name = {inp.get("name"): inp for inp in inputs}
ref = by_name.get("ref_images.ref_image_0") or by_name.get("ref_image_0")
first = by_name.get("first_frame")
assert ref is not None and first is not None, "expected inputs not found"
assert ref.get("link") is not None, "ref_image_0 has no link to move"
assert first.get("link") is None, "first_frame already connected"

first["link"] = ref["link"]
ref["link"] = None

wv = cond.get("widgets_values")
assert isinstance(wv, list) and wv[4] in ("T2VA", "I2VA", "auto", "FL2VA", "L2VA", "Ref2VA"), wv[4]
print("task_type before:", wv[4])
wv[4] = "I2VA"
wv[0] = PROMPT

# fix link bookkeeping: last_link_id unchanged (we reuse the same link id)
with open(PATH, "w", encoding="utf-8") as f:
    json.dump(d, f, ensure_ascii=False, indent=2)

# verify
d2 = json.load(open(PATH, encoding="utf-8"))
for n in d2["nodes"]:
    if "AudioConditioning" in str(n.get("type", "")):
        for inp in n.get("inputs") or []:
            if inp.get("name") in ("first_frame", "ref_images.ref_image_0", "ref_image_0"):
                print(inp.get("name"), "->", inp.get("link"))
        print("task_type after:", n["widgets_values"][4])
        print("prompt head:", n["widgets_values"][0][:60])
print("OK, backup at", BACKUP)
