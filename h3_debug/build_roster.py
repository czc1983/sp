# -*- coding: utf-8 -*-
"""
全片人物名册登记（kimi-k2.5，走百炼网关）
- 输入：h3_debug 下的全片抽帧（gf_*.jpg / gf2_*.jpg）
- 输出：global_roster.txt（角色数量 + 名册 + 配色）
- 预警：片中可能有两名或更多不同男性角色，必须按脸型/发型/痣/服装区分，宁可多列不可合并
用法: python build_roster.py [模型名]   默认 kimi-k2.5
"""
import base64
import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(r"E:\sp")
FRAMES_DIR = ROOT / "h3_debug"
SETTINGS = ROOT / ".dub_config" / "settings.json"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "kimi-k2.5"
OUT = FRAMES_DIR / "global_roster.txt"

# 全片抽帧：镜子房(gf 5-50) + 蓝窗房(gf2 40-73) + 卧室(gf 80-110)
FRAME_FILES = [
    "gf_5.0.jpg", "gf_15.0.jpg", "gf_25.0.jpg", "gf_35.0.jpg", "gf_50.0.jpg",
    "gf2_40.0.jpg", "gf2_45.0.jpg", "gf2_54.0.jpg", "gf2_62.0.jpg",
    "gf2_64.0.jpg", "gf2_68.0.jpg",
    "gf_80.0.jpg", "gf_95.0.jpg", "gf_110.0.jpg",
]

cfg = json.loads(SETTINGS.read_text(encoding="utf-8"))["foreign_dub"]
BASE_URL = cfg["asr"]["base_url"].rstrip("/")
API_KEY = cfg["asr"]["api_key"]

content = []
for name in FRAME_FILES:
    p = FRAMES_DIR / name
    if not p.exists():
        print(f"!! 缺帧 {name}，跳过", flush=True)
        continue
    b64 = base64.b64encode(p.read_bytes()).decode()
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

prompt = f"""这是一部 117 秒短剧的 {len(content)} 张全片抽帧（按时间顺序，覆盖全片前、中、后段）。请为全片做"人物名册"登记，供后续逐镜头反推时统一身份使用。严格遵守以下规则：

1. 镜子、玻璃等反光面中的影像是倒影，不是独立人物，数人数时必须排除。
2. 【重要预警】片中很可能有两名或更多不同的男性角色：他们年龄相仿、都是黑发，容易混淆。必须按脸型（瘦削/圆润）、发型（向上梳起/刘海垂落）、面部标志（痣、疤痕）、服装（颜色/花纹）逐一比对区分，宁可多列、绝不可把两个人合并成一个。女性角色同理核查。
3. 拿不准是否同一人时，分开列，并在条目里注明"可能与X是同一人，不确定"。
4. 角色名必须按"分配颜色+性别"命名（如"深灰蓝男""暖陶土女"），名字里严禁出现任何服装、面料词汇（下游视频模型会按角色名字面生成真实衣物）；禁止用"男主/反派/丈夫"等剧情称呼，禁止推测人物关系与剧情。
5. 每人的服装要描述可比对细节（颜色、花纹、领口），仅供身份比对，不会进入角色名。

请严格按以下格式输出：
【角色数量】（排除倒影后的数字）
【名册】
- 角色名：性别、发型、脸型与面部标志、典型服装（含花纹/颜色细节）、在哪些场景/行为中出现（只写看得见的）
【配色】
- 角色名=哑光颜色（配色规则：女性角色=哑光暖陶土；男性角色中穿深色服装者=哑光深灰蓝、穿白色服装者=哑光墨绿；若角色更多，自选对比强烈的哑光纯色并写清；每位角色颜色必须不同）"""

content.append({"type": "text", "text": prompt})

payload = {"model": MODEL, "messages": [{"role": "user", "content": content}],
           "max_tokens": 4096, "temperature": 0.1}
t0 = time.time()
r = requests.post(f"{BASE_URL}/chat/completions",
                  headers={"Authorization": f"Bearer {API_KEY}"},
                  json=payload, timeout=600)
el = time.time() - t0
if r.status_code != 200:
    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
d = r.json()
text = d["choices"][0]["message"]["content"]
usage = d.get("usage") or {}

# 截取正文（从【角色数量】起）
idx = text.find("【角色数量】")
if idx > 0:
    text = text[idx:].strip()

OUT.write_text(text + "\n", encoding="utf-8")
print(text, flush=True)
print(f"\n[名册已写入 {OUT}] 模型={MODEL} 耗时={el:.1f}s "
      f"in={usage.get('prompt_tokens', -1)} out={usage.get('completion_tokens', -1)}", flush=True)
