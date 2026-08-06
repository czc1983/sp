# -*- coding: utf-8 -*-
"""v4 工作流（288x512/6步）：换提示词 + 换参考视频 h3_test2.mp4，提交 8189"""
import json
import urllib.request

prompt_text = open("/root/prompt_v6.txt", encoding="utf-8").read().strip()
wf = json.load(open("/root/h3_prompt_v4.json"))
nodes = wf.get("prompt", wf)
assert nodes["136"]["inputs"]["width"] == 288 and nodes["124"]["inputs"]["steps"] == 6, "workflow config drifted"
nodes["138"]["inputs"]["value"] = prompt_text
nodes["200"]["inputs"]["file"] = "h3_test2.mp4"
json.dump(wf, open("/root/h3_prompt_v6.json", "w"), ensure_ascii=False, indent=1)

payload = json.dumps({"prompt": nodes}).encode()
req = urllib.request.Request("http://127.0.0.1:8189/prompt", data=payload,
                             headers={"Content-Type": "application/json"})
print(json.dumps(json.load(urllib.request.urlopen(req)), ensure_ascii=False))
