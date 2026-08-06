# -*- coding: utf-8 -*-
"""通用提交：python submit_h3_seg.py <prompt_txt路径> <input目录中的视频名> <输出workflow名>"""
import json
import sys
import urllib.request

prompt_path, video_file, wf_name = sys.argv[1], sys.argv[2], sys.argv[3]
prompt_text = open(prompt_path, encoding="utf-8").read().strip()
wf = json.load(open("/root/h3_prompt_v4.json"))
nodes = wf.get("prompt", wf)
assert nodes["136"]["inputs"]["width"] == 288 and nodes["124"]["inputs"]["steps"] == 6, "workflow config drifted"
nodes["138"]["inputs"]["value"] = prompt_text
nodes["200"]["inputs"]["file"] = video_file
json.dump(wf, open(f"/root/{wf_name}", "w"), ensure_ascii=False, indent=1)

payload = json.dumps({"prompt": nodes}).encode()
req = urllib.request.Request("http://127.0.0.1:8189/prompt", data=payload,
                             headers={"Content-Type": "application/json"})
print(json.dumps(json.load(urllib.request.urlopen(req)), ensure_ascii=False))
