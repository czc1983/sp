# -*- coding: utf-8 -*-
"""
百炼云反推 v4：混合管线（快速模型干事实层 + 思考模型只做最终复核）
  Pass A  人物与场景档案（快速模型，全部帧）
  Pass B  逐帧窄问题（快速模型，含"背影必须服装比对身份"铁律）
  Pass B2 不确定重审（快速模型，仅解决身份类问题，禁止叙事脑补）
  Pass C  镜头与光影（快速模型）
  Pass D  组装 H3 提示词草稿（快速模型，纯文本）
  Pass E  整体复核（思考模型 qwen3-vl-235b-a22b-thinking，跨模型纠错）
记录每次调用的耗时与 token 消耗，结尾输出成本报表。
用法: python bailian_reverse.py [快速模型名]   默认 kimi-k2.5
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

FAST_MODEL = sys.argv[1] if len(sys.argv) > 1 else "kimi-k2.5"
REVIEW_MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3-vl-30b-a3b-instruct"
# 第3、4个参数：抽帧时间点（逗号分隔）、帧文件前缀（区分不同视频，隔离缓存）
FRAME_TIMES = [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [0.2, 1.0, 2.0, 3.0, 3.5, 4.4]
FRAME_PREFIX = sys.argv[4] if len(sys.argv) > 4 else "qf"
VIDEO_DURATION = FRAME_TIMES[-1] + 0.2
# 第5个参数（可选）：全片人物名册文件，注入后跨段身份与配色统一
ROSTER = Path(sys.argv[5]).read_text(encoding="utf-8") if len(sys.argv) > 5 else None
REVIEW_SLUG = REVIEW_MODEL.replace("/", "_").replace(".", "_")
SLUG = FAST_MODEL.replace("/", "_").replace(".", "_") + f"_{FRAME_PREFIX}_hybrid" + ("_roster" if ROSTER else "")
CACHE_PATH = FRAMES_DIR / f"bailian_cache_{SLUG}.json"
RESULT_MD = FRAMES_DIR / f"bailian_reverse_result_{SLUG}.md"
PROMPT_TXT = FRAMES_DIR / f"prompt_v4_{SLUG}.txt"
DRAFT_TXT = FRAMES_DIR / f"prompt_v4_{SLUG}_draft.txt"

cfg = json.loads(SETTINGS.read_text(encoding="utf-8"))["foreign_dub"]
BASE_URL = cfg["asr"]["base_url"].rstrip("/")
API_KEY = cfg["asr"]["api_key"]

CALL_LOG = []  # {pass_name, model, seconds, prompt_tokens, completion_tokens, total_tokens}


def img_url(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/jpeg;base64,{b64}"


def chat(messages, model, pass_name, max_tokens=4096, temperature=0.1):
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    if model.startswith("qwen"):
        payload["vl_high_resolution_images"] = True
    t0 = time.time()
    r = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json=payload,
        timeout=600,
    )
    el = time.time() - t0
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    d = r.json()
    usage = d.get("usage") or {}
    CALL_LOG.append({
        "pass": pass_name,
        "model": model,
        "seconds": round(el, 1),
        "prompt_tokens": usage.get("prompt_tokens", -1),
        "completion_tokens": usage.get("completion_tokens", -1),
        "total_tokens": usage.get("total_tokens", -1),
    })
    return d["choices"][0]["message"]["content"]


def frames_content(times):
    return [{"type": "image_url", "image_url": {"url": img_url(FRAMES_DIR / f"{FRAME_PREFIX}_{t}.jpg")}} for t in times]


def load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache):
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    t_start = time.time()
    cache = load_cache()

    # ---------- Pass A: 人物档案 ----------
    if "arch" in cache:
        print("[Pass A] 命中缓存，跳过", flush=True)
        arch = cache["arch"]
    else:
        if ROSTER:
            pass_a_prompt = f"""这是一段约{VIDEO_DURATION}秒短剧片段的 {len(FRAME_TIMES)} 张抽帧（时间点：{FRAME_TIMES} 秒）。

全片人物名册（跨镜头统一身份，后续必须使用名册中的称呼，不得新编编号）：
{ROSTER}

请只做"本段档案"登记，遵守以下规则：
1. 镜子、玻璃等反光面中的影像是倒影，不是独立人物。
2. 拿不准的项目必须写"不确定"，禁止猜测；禁止推测剧情与人物关系。
3. 登记服装时要具体描述花纹、颜色、领口等可比对特征。

请严格按以下格式输出：
【真实人数】（排除倒影后的数字）
【出场角色】名册中哪些角色在本段出现（用名册称呼），各自本段的服装细节（与名册典型服装不同需注明）
【反光面】画面中是否有镜子/玻璃等反光面；在什么位置；镜中映出的是哪位角色的倒影（没有就写"无"）
【肢体接触】各帧中是否可见角色之间的肢体接触，是谁的手放在谁的什么部位（用名册称呼）
【场景】室内/室外、可见的家具或结构（只列看得见的）
【可见文字】任何一帧画面里出现的字幕/文字内容及其出现的时间点（没有就写"无"）"""
        else:
            pass_a_prompt = f"""这是一段约{VIDEO_DURATION}秒短剧的 {len(FRAME_TIMES)} 张抽帧（时间点：{FRAME_TIMES} 秒）。请只做"人物与场景档案"登记，遵守以下规则：
1. 镜子、玻璃等反光面中的影像是倒影，不是独立人物。数真实人数时必须排除倒影。
2. 拿不准的项目必须写"不确定"，禁止猜测。
3. 不要描述剧情、不要推测人物关系与情绪走向，只登记看得到的事实。
4. 登记每位人物的服装时，要具体描述花纹、颜色、领口等可比对特征，供后续帧做身份比对用。

请严格按以下格式输出：
【真实人数】（排除倒影后的数字）
【人物清单】按画面中出现顺序编号（人物A、人物B……），每人登记：性别、发型、上衣（含花纹/颜色细节）、下装（看不清就写不确定）
【反光面】画面中是否有镜子/玻璃等反光面；在什么位置；镜中映出的是哪位人物的倒影；镜中哪位人物的面部最清晰
【肢体接触】各帧中是否可见人物之间的肢体接触（如手搭肩、拥抱、牵手），是谁的手放在谁的什么部位
【场景】室内/室外、可见的家具或结构（只列看得见的）
【可见文字】任何一帧画面里出现的字幕/文字内容及其出现的时间点（没有就写"无"）"""
        print("[Pass A] 人物档案 ...", flush=True)
        arch = chat([{"role": "user", "content": frames_content(FRAME_TIMES) + [{"type": "text", "text": pass_a_prompt}]}],
                    FAST_MODEL, "A")
        cache["arch"] = arch
        save_cache(cache)
        print(arch, flush=True)

    # ---------- Pass B: 逐帧窄问题 ----------
    frame_notes = cache.get("frame_notes", {})
    for t in FRAME_TIMES:
        if str(t) in frame_notes:
            print(f"[Pass B] 第 {t}s 帧命中缓存，跳过", flush=True)
            continue
        pass_b_prompt = f"""背景档案（已确认的事实，请沿用其中的人物编号，不得新增人物）：
{arch}

现在只看这一张帧（第 {t} 秒）。逐条回答以下窄问题，每条一两句话，拿不准写"不确定"，禁止推测剧情。
铁律：遇到背影、侧脸或被遮挡的人物，禁止凭构图习惯猜身份——必须先把该人物可见的服装/发型与档案中每个人的登记特征逐一比对，并指出比对的画面区域（如"右侧前景人物的外套花纹与镜中人物A一致"），再报身份；比对不上就写"不确定"。
1. 这一帧画面里有哪几位真实人物（用档案中的称呼），各自在画面什么位置（左/右/中、前景/背景）？
2. 各自的身体姿态与头部朝向（正脸/侧脸/背影，面向谁）？
3. 各自的视线方向（看向谁/看向哪里）？眼睑是睁开、半垂还是闭上？
4. 嘴部状态：闭合、微张、还是明显在说话？
5. 手部：可见的手分别在什么位置？是否有肢体接触（谁的手放在谁的什么部位）？
6. 谁遮挡谁？镜中（如有）映出了什么？镜中谁的脸最清晰？
7. 与前后相邻帧相比，这一帧最明显的一个动作变化是什么？（只根据本帧能确定的写，不确定就写不确定）"""
        print(f"[Pass B] 第 {t}s 帧 ...", flush=True)
        note = chat([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": img_url(FRAMES_DIR / f"{FRAME_PREFIX}_{t}.jpg")}},
            {"type": "text", "text": pass_b_prompt},
        ]}], FAST_MODEL, f"B-{t}")
        frame_notes[str(t)] = note
        cache["frame_notes"] = frame_notes
        save_cache(cache)
        print(note, flush=True)

    # ---------- Pass B2: 不确定重审（仅身份类，禁止叙事脑补） ----------
    frame_notes2 = cache.get("frame_notes2", {})
    for i, t in enumerate(FRAME_TIMES):
        key = str(t)
        if key in frame_notes2:
            continue
        note = frame_notes[key]
        if "不确定" not in note:
            frame_notes2[key] = note
            continue
        neighbors = [FRAME_TIMES[j] for j in (i - 1, i, i + 1) if 0 <= j < len(FRAME_TIMES)]
        recheck_prompt = f"""背景档案（已确认的事实，请沿用人物编号）：
{arch}

下面是与第 {t} 秒相邻的抽帧（依次是 {neighbors} 秒），以及之前只凭第 {t} 秒单帧做出的观察：
{note}

任务：利用相邻帧的服装与位置连续性，重新审视上面标记"不确定"的条目。严格限制：
1. 只允许解决"身份/位置/嘴部开合/眼睑"这类可直接观察的事实问题。
2. 禁止推测动作意图与剧情（例如不许从"两人距离近"推断亲吻、拥抱、亲密关系等）——只有画面中直接可见的接触才能写。
3. 每个改判都必须写出画面依据（哪个区域、什么特征）；没有直接依据的保留"不确定"。
然后输出该帧的完整修正版观察（格式与上面一致，条目1-7），不要输出分析过程。"""
        print(f"[Pass B2] 第 {t}s 帧不确定项重审（相邻帧 {neighbors}）...", flush=True)
        note2 = chat([{"role": "user", "content": frames_content(neighbors) + [{"type": "text", "text": recheck_prompt}]}],
                     FAST_MODEL, f"B2-{t}")
        frame_notes2[key] = note2
        cache["frame_notes2"] = frame_notes2
        save_cache(cache)
        print(note2, flush=True)
    cache["frame_notes2"] = frame_notes2
    save_cache(cache)

    # ---------- Pass C: 镜头与光影 ----------
    if "camera" in cache:
        print("[Pass C] 命中缓存，跳过", flush=True)
        camera = cache["camera"]
    else:
        pass_c_prompt = f"""这是一段约{VIDEO_DURATION}秒短剧的 {len(FRAME_TIMES)} 张抽帧（时间点：{FRAME_TIMES} 秒）。已确认的人物档案：
{arch}

只回答镜头与光影问题，拿不准写"不确定"：
【景别】整体景别（特写/近景/中景/全景）
【机位运动】对比各帧，机位是固定还是在运动？如果运动，方向与方式是什么（推/拉/摇/移/环绕/切换视角）？
【景深】前景与背景的虚化情况
【主光源】光线方向、色温（暖/冷）、明暗对比强度
【氛围词】用三个以内的词概括影调氛围"""
        print("[Pass C] 镜头光影 ...", flush=True)
        camera = chat([{"role": "user", "content": frames_content(FRAME_TIMES) + [{"type": "text", "text": pass_c_prompt}]}],
                      FAST_MODEL, "C")
        cache["camera"] = camera
        save_cache(cache)
        print(camera, flush=True)

    # ---------- 汇总反推结果 ----------
    md = [
        "# 反推结果（混合管线）",
        "",
        f"- 视频时长约 {VIDEO_DURATION}s，抽帧时间点：{FRAME_TIMES}",
        f"- 事实层模型：{FAST_MODEL}；复核模型：{REVIEW_MODEL}",
        "",
        "## Pass A · 人物与场景档案",
        arch,
        "",
        "## Pass B · 逐帧观察（经 B2 身份重审）",
    ]
    for t in FRAME_TIMES:
        md += [f"### 第 {t}s", frame_notes2[str(t)], ""]
    md += ["## Pass C · 镜头与光影", camera, ""]
    RESULT_MD.write_text("\n".join(md), encoding="utf-8")

    # ---------- Pass D: 组装 H3 提示词草稿 ----------
    if "draft" in cache:
        print("[Pass D] 命中缓存，跳过", flush=True)
        draft = cache["draft"]
    else:
        if ROSTER:
            # 配色从名册文件的【配色】段动态读取，支持任意人数
            m = re.search(r"【配色】\s*\n((?:\s*[-—*].*(?:\n|$))+)", ROSTER)
            color_lines = m.group(1).strip() if m else ""
            color_rule = f"（固定配色，禁止更改或交换，严格按名册配色执行：\n{color_lines}\n称呼必须使用名册角色名）"
        else:
            color_rule = "（你指定对比强烈的哑光纯色，每位人物不同色）"
        assemble_prompt = f"""你是视频生成提示词工程师。下面是对一段{VIDEO_DURATION}秒参考视频的反推结果，以及一个"风格转换"需求。请把二者合成为一段发给图生视频模型的中文提示词。

【反推结果】
{chr(10).join(md[2:])}

【风格转换需求（必须原样体现的核心要求）】
- 所有人物替换为彩色树脂关节人偶素体（BJD 娃娃素体）：光头无发、身体带球形关节、极简人偶面部；衣服消融进身体，通体同色光滑哑光树脂。不同人物不同颜色，同一人物（含其镜中倒影）严格同色。
- 场景空间结构完整保留（含镜子与镜中倒影），仅将背景与物体材质替换为哑光浅色，保留原片明暗对比与光影方向。
- 严格按参考视频的构图、人物数量、位置、姿态、遮挡关系与运镜演变，时间轴不变。
- 口型与参考音频对齐，眼神与表情必须准确保留（眼睑开合程度、视线方向、情绪状态不变）。
- 音频：完整保留参考视频原声（语言、声线、语速、情绪均不变），不重新配音，不加配乐。音频区块只允许写"台词与参考音频完全一致"，禁止在提示词里引述台词文字内容（画面中的字幕文字可能与原声语言不一致，引述会误导配音）。
- 全程不出现任何文字、字幕、水印。

【输出格式要求】
1. 开头一段"构图与人物"：写清真实人数、每位人偶的哑光颜色分配{color_rule}、各自位置，以及镜子/倒影与左右布局的关系，明确"倒影与本体是同一人，不计入人数"。
2. 中间按反推的时间轴分拍（用 [0s-1s] 这样的时间戳），每拍写人物动作、姿态、视线、嘴部状态、手部接触、镜中同步情况；反推中标记"不确定"的内容不要编造，直接省略。相邻拍之间内容有变化才写，不要重复堆砌。
3. 然后一段"镜头"、一段"光影"、一段"音频"。
4. 最后一段"负面清单"：不要真实皮肤/头发丝/睫毛/布料质感；不要删除镜子或倒影；不要把倒影计为独立人物；不要任何文字字幕水印；不要人偶颜色漂移；不要改变台词语言、不要根据画面文字生成语音。
5. 直接输出提示词全文，不要任何解释。总长度参考 400-600 字。"""
        print("[Pass D] 组装 H3 提示词草稿 ...", flush=True)
        draft = chat([{"role": "user", "content": [{"type": "text", "text": assemble_prompt}]}],
                     FAST_MODEL, "D", max_tokens=3000, temperature=0.2)
        cache["draft"] = draft
        save_cache(cache)
    DRAFT_TXT.write_text(draft, encoding="utf-8")

    # ---------- Pass E: 跨模型整体复核 ----------
    final_cache_key = f"final_{REVIEW_SLUG}"
    if final_cache_key in cache:
        print("[Pass E] 命中缓存，跳过", flush=True)
        final_prompt = cache[final_cache_key]
    else:
        review_prompt = f"""下面是参考视频的 {len(FRAME_TIMES)} 张抽帧（时间点：{FRAME_TIMES} 秒）、已确认的人物档案，以及一份根据反推结果组装的视频生成提示词草稿。

【人物档案】
{arch}

【提示词草稿】
{draft}

你是苛刻的校对员。逐条核对草稿中关于参考视频本身的每一个事实性描述（人物身份、位置、姿态、视线、嘴部、手部、遮挡、镜中内容、镜头运动、光影），与抽帧画面比对：
1. 特别注意人物身份是否与档案矛盾（背影人物的服装必须与档案逐人比对，指出画面依据）。
2. 特别警惕"叙事化脑补"：草稿中任何动作/情绪描述，若抽帧画面里没有直接可见的依据（例如从"距离近"推断出的亲吻、拥抱升级），必须删除。
3. 风格转换部分（树脂人偶、颜色分配、负面清单）属于需求指令，不核对，但若与时间轴描述矛盾也需理顺。
4. 拿不准的细节宁可删去，不要保留可疑描述。

输出：修正后的提示词全文（格式与草稿一致），不要输出校对过程或任何解释。"""
        print(f"[Pass E] 跨模型整体复核（{REVIEW_MODEL}）...", flush=True)
        final_prompt = chat([{"role": "user", "content": frames_content(FRAME_TIMES) + [{"type": "text", "text": review_prompt}]}],
                            REVIEW_MODEL, "E", max_tokens=3000, temperature=0.1)
        # 剥离可能的回显（从正文开头标记起截取）
        for marker in ("构图与人物", "构图与场景"):
            idx = final_prompt.find(marker)
            if idx > 0:
                final_prompt = final_prompt[idx:].strip()
                break
        cache[final_cache_key] = final_prompt
        save_cache(cache)
    # ---------- 固定风格合约头（程序化焊死，不经 LLM） ----------
    FIXED_HEADER = """视频 1 是构图、动作、口型、运镜与光影的唯一参考。

风格合约：所有人物替换为彩色树脂关节人偶素体（BJD 娃娃素体）：光头无发、身体带球形关节、极简人偶面部；衣服消融进身体，通体同色光滑哑光树脂。每位人偶分配下文指定的哑光纯色，同一人物与其镜中倒影（如有镜子）严格同色。场景的空间结构与关键道具完整保留——若画面中存在镜子则含镜框与镜中倒影，墙面、房间布局、光源方向全部不变，仅将背景与物体材质替换为哑光浅色，保留原片的明暗对比与光影方向。

画面文字：全程不出现任何文字、字母、符号、字幕与水印；即使参考视频中有烧录字幕，也必须在画面中完全去除，不得复刻、不得残留任何类似文字的图形。"""
    final_with_header = FIXED_HEADER + "\n\n" + final_prompt
    PROMPT_TXT.write_text(final_with_header, encoding="utf-8")
    (FRAMES_DIR / f"prompt_v4_{SLUG}_{REVIEW_SLUG}.txt").write_text(final_with_header, encoding="utf-8")

    # ---------- 成本报表 ----------
    total_el = time.time() - t_start
    report = ["", "=" * 60, "【成本报表】"]
    sum_in = sum_out = 0
    for c in CALL_LOG:
        report.append(f"  {c['pass']:<8} {c['model']:<35} {c['seconds']:>6}s  "
                      f"in={c['prompt_tokens']:>7} out={c['completion_tokens']:>6} total={c['total_tokens']:>7}")
        if c['prompt_tokens'] > 0:
            sum_in += c['prompt_tokens']
            sum_out += c['completion_tokens']
    report.append(f"  合计: {len(CALL_LOG)} 次调用, 输入 {sum_in} tokens, 输出 {sum_out} tokens, 墙钟 {total_el:.0f}s（含缓存跳过）")
    text = "\n".join(report)
    print(text, flush=True)
    (FRAMES_DIR / f"cost_report_{SLUG}.txt").write_text(text, encoding="utf-8")

    print("=" * 60, flush=True)
    print(final_prompt, flush=True)


if __name__ == "__main__":
    main()
