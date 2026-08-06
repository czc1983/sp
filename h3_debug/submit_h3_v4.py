# -*- coding: utf-8 -*-
"""patch H3 工作流：替换提示词 + 288x512/6步 测试配置，提交 8189"""
import json
import urllib.request

prompt_text = open("/root/prompt_v4.txt", encoding="utf-8").read().strip()
wf = json.load(open("/root/h3_prompt_v2.json"))
nodes = wf.get("prompt", wf)

nodes["138"]["inputs"]["value"] = prompt_text
nodes["136"]["inputs"]["width"] = 288
nodes["136"]["inputs"]["height"] = 512
nodes["124"]["inputs"]["steps"] = 6

json.dump(wf, open("/root/h3_prompt_v4.json", "w"), ensure_ascii=False, indent=1)

payload = json.dumps({"prompt": nodes}).encode()
req = urllib.request.Request("http://127.0.0.1:8189/prompt", data=payload,
                             headers={"Content-Type": "application/json"})
resp = json.load(urllib.request.urlopen(req))
print(json.dumps(resp, ensure_ascii=False))
