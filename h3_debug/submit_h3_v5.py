# -*- coding: utf-8 -*-
"""用已 patch 好的 v4 工作流（288x512/6步），替换提示词为 v5 并提交 8189"""
import json
import urllib.request

prompt_text = open("/root/prompt_v5.txt", encoding="utf-8").read().strip()
wf = json.load(open("/root/h3_prompt_v4.json"))
nodes = wf.get("prompt", wf)
assert nodes["136"]["inputs"]["width"] == 288 and nodes["124"]["inputs"]["steps"] == 6, "workflow config drifted"
nodes["138"]["inputs"]["value"] = prompt_text
json.dump(wf, open("/root/h3_prompt_v5.json", "w"), ensure_ascii=False, indent=1)

payload = json.dumps({"prompt": nodes}).encode()
req = urllib.request.Request("http://127.0.0.1:8189/prompt", data=payload,
                             headers={"Content-Type": "application/json"})
print(json.dumps(json.load(urllib.request.urlopen(req)), ensure_ascii=False))
