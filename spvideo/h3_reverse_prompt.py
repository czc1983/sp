"""MiniMax H3 白膜提示词自动反推模块（Mode 2 编排页路线）。

移植自夜班验证管线 h3_debug/bailian_reverse.py（2026-08-05 定稿的混合方案）：
  Pass A  人物与场景档案（快速模型，全部抽帧）
  Pass B  逐帧窄问题（快速模型，含"背影必须服装比对身份"铁律）
  Pass B2 不确定重审（快速模型，仅解决身份类问题，禁止叙事脑补）
  Pass C  镜头与光影（快速模型）
  Pass D  组装 H3 提示词草稿（快速模型，纯文本）
  Pass E  整体复核（复核模型，跨模型纠错）

接口配置复用 .dub_config/settings.json 的 foreign_dub.asr（阿里兼容端点 + key），
kimi-k2.5 / qwen3-vl 系列都走这一个端点（夜班实测路线）。

用法：
    from spvideo.h3_reverse_prompt import reverse_prompt_for_clip
    result = reverse_prompt_for_clip("path/to/clip.mp4", roster_text=None)
    prompt = result["prompt"]   # 含焊死风格合约头的完整提示词
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import requests

SP_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = SP_ROOT / ".dub_config" / "settings.json"
CACHE_ROOT = SP_ROOT / ".h3_reverse_cache"
# 反推规则版本：规则改动后必须 bump，避免旧缓存把已废弃的结论带回来。
PROMPT_CACHE_VERSION = "v6_scene_context_20260806"
PROMPT_FORMATS = {"sp_v4", "official_v1"}
DEFAULT_PROMPT_FORMAT = "sp_v4"

DEFAULT_FAST_MODEL = "kimi-k2.5"
DEFAULT_REVIEW_MODEL = "qwen3-vl-30b-a3b-instruct"

FRAME_COUNT = 6  # 每段抽帧数（与夜班管线一致）

FIXED_HEADER = """视频 1 是构图、动作、口型、运镜与光影的唯一参考。

风格合约：所有人物替换为彩色树脂关节人偶素体（BJD 娃娃素体）：光头无发、身体带球形关节、极简人偶面部；衣服消融进身体，通体同色光滑哑光树脂。每位人偶分配下文指定的哑光纯色，同一人物与其镜中倒影（如有镜子）严格同色。场景的空间结构与关键道具完整保留——若画面中存在镜子则含镜框与镜中倒影，墙面、房间布局、光源方向全部不变，仅将背景与物体材质替换为哑光浅色，保留原片的明暗对比与光影方向。

画面文字：全程不出现任何文字、字母、符号、字幕与水印；即使参考视频中有烧录字幕，也必须在画面中完全去除，不得复刻、不得残留任何类似文字的图形。"""


class ReversePromptError(RuntimeError):
    """反推管线失败（接口、抽帧、配置等）。"""


def _context_block(scene_context: str) -> str:
    """把全片预分析的剧情背景包装成各 Pass 通用的消歧块。"""
    return f"""全片剧情背景（来自整片预分析，是已确认的事实，不是推测）：
{scene_context.strip()}

背景使用规则：
1. 以上背景只用于消歧——判断人物身份、动作含义和可见部件状态（例如"痛苦挣扎"背景下张嘴皱眉是痛苦呼喊而非大笑）。
2. 各条规则中的"禁止推测剧情"指禁止编造背景之外的新剧情、新人物、新关系；背景本身可以直接作为判读依据。
3. 落笔仍只写画面中可见的事实；背景解释了"为什么"，但不能写出画面里看不到的东西。"""


def _context_hash(scene_context: str | None, roster_text: str | None) -> str:
    raw = f"{(scene_context or '').strip()}|{(roster_text or '').strip()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _load_api_config() -> tuple[str, str]:
    try:
        cfg = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))["foreign_dub"]["asr"]
        base_url = str(cfg["base_url"]).rstrip("/")
        api_key = str(cfg["api_key"]).strip()
    except Exception as exc:  # noqa: BLE001
        raise ReversePromptError(f"h3_reverse_config: 读取 {SETTINGS_PATH} 失败: {exc}") from exc
    if not base_url or not api_key:
        raise ReversePromptError("h3_reverse_config: foreign_dub.asr.base_url/api_key 为空")
    return base_url, api_key


def clip_fingerprint(clip_path: str | Path) -> str:
    """按路径+大小+修改时间+反推规则版本算指纹，变了就重新反推。"""
    clip = Path(clip_path).resolve()
    stat = clip.stat()
    raw = f"{clip}|{stat.st_size}|{int(stat.st_mtime)}|{PROMPT_CACHE_VERSION}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def _normalize_prompt_format(prompt_format: str | None) -> str:
    fmt = str(prompt_format or DEFAULT_PROMPT_FORMAT).strip() or DEFAULT_PROMPT_FORMAT
    if fmt not in PROMPT_FORMATS:
        raise ReversePromptError(f"h3_reverse_format: 未知 prompt_format={fmt!r}，可选 {sorted(PROMPT_FORMATS)}")
    return fmt


def cache_path_for(clip_path: str | Path, prompt_format: str | None = None) -> Path:
    fmt = _normalize_prompt_format(prompt_format)
    return CACHE_ROOT / f"{clip_fingerprint(clip_path)}_{fmt}.json"


def read_cached_prompt(clip_path: str | Path, prompt_format: str | None = None) -> str | None:
    """不改变文件的前提下，读取已缓存的最终提示词（供 GET 查询用）。"""
    path = cache_path_for(clip_path, prompt_format)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    final = str(data.get("final_prompt") or "").strip()
    return final or None


def _frame_times(duration: float) -> list[float]:
    """按时长自适应取抽帧点：开头 0.2s + 均布 + 结尾前 0.3s。

    超短片段（< 0.6s）不能用 0.6s 下限硬套：抽帧点会超出真实时长，
    ffmpeg 在最后一帧之后抽不到帧直接报"抽帧失败"（0.28s 镜头的教训）。
    短片段改为在 [d*0.2, d-0.05] 内按内容量取 1~N 个点。
    """
    d = float(duration)
    if d <= 0:
        d = 0.6
    if d < 0.6:
        lo = max(0.0, min(0.1, d * 0.2))
        hi = max(lo, d - 0.05)
        count = max(1, min(FRAME_COUNT, int(round(d * 4))))
        if count <= 1:
            return [round((lo + hi) / 2, 2)]
        step = (hi - lo) / (count - 1)
        return sorted({round(lo + step * i, 2) for i in range(count)})
    times = {0.2, d * 0.25, d * 0.45, d * 0.65, d * 0.82, max(d - 0.3, 0.2)}
    return sorted(round(min(max(t, 0.1), d - 0.05), 2) for t in times)[:FRAME_COUNT]


def _extract_frames(clip: Path, times: list[float], frames_dir: Path, log: Callable[[str], None]) -> dict[float, Path]:
    from spvideo import ffmpeg_tools

    ffmpeg = ffmpeg_tools.find_binary(ffmpeg_tools.FFMPEG_CANDIDATES)
    frames_dir.mkdir(parents=True, exist_ok=True)
    out: dict[float, Path] = {}
    for t in times:
        target = frames_dir / f"f_{t:.2f}.jpg"
        if not target.is_file() or target.stat().st_size <= 0:
            subprocess.run(
                [ffmpeg, "-y", "-ss", f"{t:.2f}", "-i", str(clip), "-vframes", "1",
                 "-q:v", "3", str(target)],
                capture_output=True, check=False,
            )
        if not target.is_file() or target.stat().st_size <= 0:
            # 兜底：seek 越过最后一帧时（短片段尾部），往前退 0.1s 再抽一次，仍不行就抽首帧
            for fallback in (max(0.0, t - 0.1), 0.0):
                if fallback == t:
                    continue
                log(f"> 反推抽帧: t={t}s 抽不到，退回 t={fallback:.2f}s")
                subprocess.run(
                    [ffmpeg, "-y", "-ss", f"{fallback:.2f}", "-i", str(clip), "-vframes", "1",
                     "-q:v", "3", str(target)],
                    capture_output=True, check=False,
                )
                if target.is_file() and target.stat().st_size > 0:
                    break
        if not target.is_file() or target.stat().st_size <= 0:
            raise ReversePromptError(f"h3_reverse_frame: 抽帧失败 t={t}s clip={json.dumps(str(clip))}")
        out[t] = target
    log(f"> 反推抽帧: {len(out)} 帧 @ {[t for t in out]}")
    return out


class _ReverseSession:
    """一次反推的上下文：接口、缓存、调用日志。"""

    def __init__(self, clip: Path, *, fast_model: str, review_model: str,
                 prompt_format: str, log: Callable[[str], None],
                 context_hash: str = ""):
        self.clip = clip
        self.fast_model = fast_model
        self.review_model = review_model
        self.prompt_format = _normalize_prompt_format(prompt_format)
        self.log = log
        self.base_url, self.api_key = _load_api_config()
        self.cache_path = cache_path_for(clip, self.prompt_format)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, Any] = {}
        if self.cache_path.is_file():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self.cache = {}
        # 剧情背景或名册变化时，旧的分 Pass 缓存全部作废重推。
        if self.cache.get("context_hash") != context_hash:
            self.cache = {}
        self.cache["context_hash"] = context_hash
        self.call_log: list[dict[str, Any]] = []

    def save_cache(self) -> None:
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _img_url(path: Path) -> str:
        b64 = base64.b64encode(path.read_bytes()).decode()
        return f"data:image/jpeg;base64,{b64}"

    def frames_content(self, frames: dict[float, Path], times: list[float]) -> list[dict[str, Any]]:
        return [{"type": "image_url", "image_url": {"url": self._img_url(frames[t])}} for t in times]

    def chat(self, messages: list[dict[str, Any]], model: str, pass_name: str,
             max_tokens: int = 4096, temperature: float = 0.1) -> str:
        payload: dict[str, Any] = {"model": model, "messages": messages,
                                   "max_tokens": max_tokens, "temperature": temperature}
        if model.startswith("qwen"):
            payload["vl_high_resolution_images"] = True
        t0 = time.time()
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=600,
        )
        elapsed = time.time() - t0
        if resp.status_code != 200:
            raise ReversePromptError(f"h3_reverse_api: HTTP {resp.status_code} {resp.text[:300]}")
        data = resp.json()
        usage = data.get("usage") or {}
        self.call_log.append({
            "pass": pass_name, "model": model, "seconds": round(elapsed, 1),
            "prompt_tokens": usage.get("prompt_tokens", -1),
            "completion_tokens": usage.get("completion_tokens", -1),
        })
        self.log(f"> 反推 {pass_name}（{model}）{elapsed:.0f}s")
        return str(data["choices"][0]["message"]["content"])


def _build_assemble_prompt(md_text: str, video_duration: float, color_rule: str, prompt_format: str) -> str:
    """Pass D 组装提示词。sp_v4 为当前稳定版；official_v1 套 MiniMax H3 Ref2VA 六字段壳。"""
    if prompt_format == "official_v1":
        return f"""你是视频生成提示词工程师。下面是对一段{video_duration}秒参考视频的反推结果，以及一个"风格转换"需求。请把二者合成为发给 MiniMax H3 Ref2VA 的中文提示词，描述性文字用简体中文，官方字段名、标签、标记保持英文。

【反推结果】
{md_text}

【风格转换需求（必须体现）】
- 所有人物替换为彩色树脂关节人偶素体（BJD 娃娃素体）：光头无发、身体带球形关节、极简人偶面部；衣服消融进身体，通体同色光滑哑光树脂。不同人物不同颜色，同一人物（含其镜中倒影）严格同色。
- 场景空间结构完整保留（含镜子与镜中倒影），仅将背景与物体材质替换为哑光浅色，保留原片明暗对比与光影方向。
- 严格按参考视频的构图、人物数量、位置、姿态、遮挡关系与运镜演变，时间轴不变；默认整段只是一个连续镜头，优先运镜，不要发明切镜。
- 口型与参考音频对齐，眼神与面部表情只按可观察部件保留：眼睑开合程度、视线方向、眉形、嘴角、嘴部开合逐拍一致；禁止把表情概括成任何情绪结论词。
- 档案里的发型/发色/服装质地只用于身份比对，不得写进最终提示词；服装一律表述为消融进身体的同色哑光树脂。禁止出现头发、头发丝、睫毛、真实皮肤纹理、布料质感等与树脂素体冲突的源片质感词。
- 音频：完整保留参考视频原声，不重新配音，不加配乐；禁止引述台词文字内容。全程不出现任何文字、字幕、水印。

【输出格式要求：严格输出且只输出以下六个字段，按此顺序，字段名英文，字段之间空一行】
subject_definitions:
定义 <Video 1> 为唯一参考视频；定义每位人偶为 <Subject N>，写清哑光纯色分配{color_rule}、位置、镜子/倒影关系，明确倒影与本体是同一人不计入人数；写清"画面全程无文字/字幕/水印"。
summary:
以方括号组合开头，例如 [video editing, audio reuse]，用一小段说明本段是把参考视频人物替换为彩色树脂关节人偶素体并复用原音频。
retention_analysis:
每个被跟踪标签一行，关系只用 fully_preserved / partially_preserved / attribute_transfer / weak_reference；至少覆盖构图、真实人数、位置、姿态、遮挡、镜子/倒影、眼睑/视线/眉形/嘴角/嘴部、音频、光影。
detailed_description:
先用一两句确立树脂人偶风格与材质替换规则，再从 [Shot 1] 开始按播放顺序写；[Shot 1] 不加 At 时间戳。默认只写 [Shot 1] 一个连续镜头，内部可按反推时间轴写"0.2-1.2s"这类节拍；只有反推结果里有明确切镜证据时才增加 [Shot N] At MM:SS.mmm，且时间戳严格递增、小于 {video_duration:.2f} 秒。每拍写人物动作、姿态、视线、眼睑开合、眉形/嘴角、嘴部状态、手部接触、镜中同步；只允许写可见部件状态，禁止情绪结论词；反推中标记"不确定"的内容直接省略。
overall_soundscape:
只写"台词与参考音频完全一致"，可加一句环境声/动作声保持参考视频；不要重复或引述任何台词文字。
non_diegetic_music:
没有观众视角配乐时写 N/A。

直接输出六字段全文，不要 Markdown 代码围栏，不要解释，不要校对过程。"""

    return f"""你是视频生成提示词工程师。下面是对一段{video_duration}秒参考视频的反推结果，以及一个"风格转换"需求。请把二者合成为一段发给图生视频模型的中文提示词。

【反推结果】
{md_text}

【风格转换需求（必须原样体现的核心要求）】
- 所有人物替换为彩色树脂关节人偶素体（BJD 娃娃素体）：光头无发、身体带球形关节、极简人偶面部；衣服消融进身体，通体同色光滑哑光树脂。不同人物不同颜色，同一人物（含其镜中倒影）严格同色。
- 场景空间结构完整保留（含镜子与镜中倒影），仅将背景与物体材质替换为哑光浅色，保留原片明暗对比与光影方向。
- 严格按参考视频的构图、人物数量、位置、姿态、遮挡关系与运镜演变，时间轴不变。
- 口型与参考音频对齐，眼神与面部表情必须按可观察特征保留：眼睑开合程度、视线方向、眉形、嘴角、嘴部开合逐拍一致；禁止把表情概括成任何情绪结论词。
- 音频：完整保留参考视频原声（语言、声线、语速、情绪均不变），不重新配音，不加配乐。音频区块只允许写"台词与参考音频完全一致"，禁止在提示词里引述台词文字内容（画面中的字幕文字可能与原声语言不一致，引述会误导配音）。
- 全程不出现任何文字、字幕、水印。

【输出格式要求】
1. 开头一段"构图与人物"：写清真实人数、每位人偶的哑光颜色分配{color_rule}、各自位置，以及镜子/倒影与左右布局的关系，明确"倒影与本体是同一人，不计入人数"。档案里的发型/发色/服装质地只用于身份比对，不得写进最终提示词；服装一律表述为消融进身体的同色哑光树脂。
2. 中间按反推的时间轴分拍（用 [0s-1s] 这样的时间戳），每拍写人物动作、姿态、视线、眼睑开合、眉形/嘴角、嘴部状态、手部接触、镜中同步情况；只允许写可见部件状态，禁止情绪结论词；禁止出现头发、头发丝、睫毛、真实皮肤纹理、布料质感等与树脂素体冲突的源片质感词；反推中标记"不确定"的内容不要编造，直接省略。相邻拍之间内容有变化才写，不要重复堆砌。
3. 然后一段"镜头"、一段"光影"、一段"音频"。
4. 最后一段"负面清单"：不要真实皮肤/头发丝/睫毛/布料质感；不要删除镜子或倒影；不要把倒影计为独立人物；不要任何文字字幕水印；不要人偶颜色漂移；不要用情绪结论词概括表情，表情只用眼睑/视线/眉形/嘴角/嘴部开合描述；不要改变台词语言、不要根据画面文字生成语音。
5. 直接输出提示词全文，不要任何解释。总长度参考 400-600 字。"""


def reverse_prompt_for_clip(
    clip_path: str | Path,
    *,
    roster_text: str | None = None,
    scene_context: str | None = None,
    fast_model: str = DEFAULT_FAST_MODEL,
    review_model: str = DEFAULT_REVIEW_MODEL,
    prompt_format: str = DEFAULT_PROMPT_FORMAT,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """对单个片段反推 H3 白膜提示词。返回 {prompt, draft, arch, camera, cost, cached}。"""
    logger = log or (lambda _m: None)
    clip = Path(clip_path).resolve()
    if not clip.is_file():
        raise ReversePromptError(f"h3_reverse_clip_missing: {json.dumps(str(clip))}")

    context = (scene_context or "").strip()
    ctx_block = _context_block(context) if context else ""
    ctx_hash = _context_hash(scene_context, roster_text)
    s = _ReverseSession(clip, fast_model=fast_model, review_model=review_model,
                        prompt_format=prompt_format, log=logger,
                        context_hash=ctx_hash)
    if context:
        logger("> 反推: 已注入全片剧情背景（消歧用）")
    if s.cache.get("final_prompt"):
        logger("> 反推: 命中缓存，直接复用")
        return {**s.cache.get("result_meta", {}), "prompt": s.cache["final_prompt"], "cached": True}

    from spvideo import ffmpeg_tools

    meta = ffmpeg_tools.probe_video(clip)
    duration = float(meta.duration or 0)
    times = _frame_times(duration)
    frames = _extract_frames(clip, times, CACHE_ROOT / f"frames_{clip_fingerprint(clip)}", logger)
    video_duration = round(duration, 1)
    roster = (roster_text or "").strip() or None
    t_start = time.time()

    # ---------- Pass A: 人物与场景档案 ----------
    if s.cache.get("arch"):
        arch = s.cache["arch"]
    else:
        if roster:
            pass_a_prompt = f"""这是一段约{video_duration}秒短剧片段的 {len(times)} 张抽帧（时间点：{times} 秒）。

全片人物名册（跨镜头统一身份，后续必须使用名册中的称呼，不得新编编号）：
{roster}

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
            pass_a_prompt = f"""这是一段约{video_duration}秒短剧的 {len(times)} 张抽帧（时间点：{times} 秒）。请只做"人物与场景档案"登记，遵守以下规则：
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
        if ctx_block:
            pass_a_prompt = ctx_block + "\n\n" + pass_a_prompt
        arch = s.chat([{"role": "user", "content": s.frames_content(frames, times) + [{"type": "text", "text": pass_a_prompt}]}],
                      s.fast_model, "A")
        s.cache["arch"] = arch
        s.save_cache()

    # ---------- Pass B: 逐帧窄问题 ----------
    frame_notes = s.cache.get("frame_notes", {})
    for t in times:
        if str(t) in frame_notes:
            continue
        pass_b_prompt = (ctx_block + "\n\n" if ctx_block else "") + f"""背景档案（已确认的事实，请沿用其中的人物编号，不得新增人物）：
{arch}

现在只看这一张帧（第 {t} 秒）。逐条回答以下窄问题，每条一两句话，拿不准写"不确定"，禁止推测剧情。
铁律：遇到背影、侧脸或被遮挡的人物，禁止凭构图习惯猜身份——必须先把该人物可见的服装/发型与档案中每个人的登记特征逐一比对，并指出比对的画面区域（如"右侧前景人物的外套花纹与镜中人物A一致"），再报身份；比对不上就写"不确定"。
表情铁律：只准登记可观察部件状态（眼睑开合、视线方向、眉形、嘴角、嘴部开合）；禁止写大笑、微笑、痛苦、愤怒、悲伤、撕心裂肺等任何情绪结论词；只能写"眼睑从闭合转为睁开""嘴角上扬"这类可见变化。
1. 这一帧画面里有哪几位真实人物（用档案中的称呼），各自在画面什么位置（左/右/中、前景/背景）？
2. 各自的身体姿态与头部朝向（正脸/侧脸/背影，面向谁）？
3. 各自的视线方向（看向谁/看向哪里）？眼睑是睁开、半垂还是闭上？眉形与嘴角可见状态是什么？
4. 嘴部状态：闭合、微张、还是明显在说话？
5. 手部：可见的手分别在什么位置？是否有肢体接触（谁的手放在谁的什么部位）？
6. 谁遮挡谁？镜中（如有）映出了什么？镜中谁的脸最清晰？
7. 与前后相邻帧相比，这一帧最明显的一个动作变化是什么？（只根据本帧能确定的写，不确定就写不确定）"""
        note = s.chat([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": s._img_url(frames[t])}},
            {"type": "text", "text": pass_b_prompt},
        ]}], s.fast_model, f"B-{t}")
        frame_notes[str(t)] = note
        s.cache["frame_notes"] = frame_notes
        s.save_cache()

    # ---------- Pass B2: 不确定重审 ----------
    frame_notes2 = s.cache.get("frame_notes2", {})
    for i, t in enumerate(times):
        key = str(t)
        if key in frame_notes2:
            continue
        note = frame_notes.get(key, "")
        if "不确定" not in note:
            frame_notes2[key] = note
            continue
        neighbors = [times[j] for j in (i - 1, i, i + 1) if 0 <= j < len(times)]
        recheck_prompt = (ctx_block + "\n\n" if ctx_block else "") + f"""背景档案（已确认的事实，请沿用人物编号）：
{arch}

下面是与第 {t} 秒相邻的抽帧（依次是 {neighbors} 秒），以及之前只凭第 {t} 秒单帧做出的观察：
{note}

任务：利用相邻帧的服装与位置连续性，重新审视上面标记"不确定"的条目。严格限制：
1. 只允许解决"身份/位置/嘴部开合/眼睑"这类可直接观察的事实问题。
2. 禁止推测动作意图与剧情（例如不许从"两人距离近"推断亲吻、拥抱、亲密关系等）——只有画面中直接可见的接触才能写。
3. 每个改判都必须写出画面依据（哪个区域、什么特征）；没有直接依据的保留"不确定"。
然后输出该帧的完整修正版观察（格式与上面一致，条目1-7），不要输出分析过程。"""
        note2 = s.chat([{"role": "user", "content": s.frames_content(frames, neighbors) + [{"type": "text", "text": recheck_prompt}]}],
                       s.fast_model, f"B2-{t}")
        frame_notes2[key] = note2
        s.cache["frame_notes2"] = frame_notes2
        s.save_cache()
    s.cache["frame_notes2"] = frame_notes2
    s.save_cache()

    # ---------- Pass C: 镜头与光影 ----------
    if s.cache.get("camera"):
        camera = s.cache["camera"]
    else:
        pass_c_prompt = f"""这是一段约{video_duration}秒短剧的 {len(times)} 张抽帧（时间点：{times} 秒）。已确认的人物档案：
{arch}

只回答镜头与光影问题，拿不准写"不确定"：
【景别】整体景别（特写/近景/中景/全景）
【机位运动】对比各帧，机位是固定还是在运动？如果运动，方向与方式是什么（推/拉/摇/移/环绕/切换视角）？
【景深】前景与背景的虚化情况
【主光源】光线方向、色温（暖/冷）、明暗对比强度
【氛围词】用三个以内的词概括影调氛围"""
        camera = s.chat([{"role": "user", "content": s.frames_content(frames, times) + [{"type": "text", "text": pass_c_prompt}]}],
                        s.fast_model, "C")
        s.cache["camera"] = camera
        s.save_cache()

    # ---------- 汇总 ----------
    md = [f"- 视频时长约 {video_duration}s，抽帧时间点：{times}",
          f"- 事实层模型：{s.fast_model}；复核模型：{s.review_model}", ""]
    if context:
        md += ["## 全片剧情背景（整片预分析已确认的事实，供消歧，禁止在此之外脑补）", context, ""]
    md += ["## Pass A · 人物与场景档案", arch, "", "## Pass B · 逐帧观察（经 B2 身份重审）"]
    for t in times:
        md += [f"### 第 {t}s", frame_notes2.get(str(t), ""), ""]
    md += ["## Pass C · 镜头与光影", camera, ""]

    # ---------- Pass D: 组装草稿 ----------
    if s.cache.get("draft"):
        draft = s.cache["draft"]
    else:
        if roster:
            m = re.search(r"【配色】\s*\n((?:\s*[-—*].*(?:\n|$))+)", roster)
            color_lines = m.group(1).strip() if m else ""
            color_rule = f"（固定配色，禁止更改或交换，严格按名册配色执行：\n{color_lines}\n称呼必须使用名册角色名）" if color_lines else "（使用名册角色名，配色对比强烈且每人不同）"
        else:
            color_rule = "（你指定对比强烈的哑光纯色，每位人物不同色）"
        assemble_prompt = _build_assemble_prompt(chr(10).join(md), video_duration, color_rule, s.prompt_format)
        draft = s.chat([{"role": "user", "content": [{"type": "text", "text": assemble_prompt}]}],
                       s.fast_model, "D", max_tokens=3000, temperature=0.2)
        s.cache["draft"] = draft
        s.save_cache()

    # ---------- Pass E: 跨模型整体复核 ----------
    review_key = f"final_{s.review_model.replace('/', '_').replace('.', '_')}"
    if s.cache.get(review_key):
        final_prompt = s.cache[review_key]
    else:
        review_context = f"【剧情背景（整片预分析已确认的事实）】\n{context}\n\n" if context else ""
        review_prompt = f"""下面是参考视频的 {len(times)} 张抽帧（时间点：{times} 秒）、已确认的人物档案，以及一份根据反推结果组装的视频生成提示词草稿。

【人物档案】
{arch}

{review_context}【提示词草稿】
{draft}

你是苛刻的校对员。逐条核对草稿中关于参考视频本身的每一个事实性描述（人物身份、位置、姿态、视线、眼睑、眉形/嘴角、嘴部、手部、遮挡、镜中内容、镜头运动、光影），与抽帧画面比对：
1. 特别注意人物身份是否与档案矛盾（背影人物的服装必须与档案逐人比对，指出画面依据）。
2. 特别警惕"叙事化脑补"：草稿中任何动作/情绪描述，若抽帧画面里没有直接可见的依据（例如从"距离近"推断出的亲吻、拥抱升级），必须删除。但与给定剧情背景一致的动作判读不算脑补（例如背景为"痛苦挣扎"时，把张嘴皱眉写作痛苦呼喊而非大笑），可以保留。
3. 表情描述硬性规则：删除大笑、微笑、痛苦、悲伤、愤怒、撕心裂肺等一切情绪结论词；只能改写成抽帧里直接可见的部件状态（眼睑从闭合转为睁开、视线方向、眉形、嘴角上扬/下撇、嘴部开合）。无法直接确认的表情细节一律删除，不许保留情绪标签。
4. 树脂素体冲突词硬性规则：删除头发、头发丝、睫毛、真实皮肤纹理、布料质感等源片质感词；档案中的发型/服装只用于身份比对，最终提示词里服装只能写成消融进身体的同色哑光树脂。
5. 风格转换部分（树脂人偶、颜色分配、负面清单）属于需求指令，不核对，但若与时间轴描述矛盾也需理顺。
6. 拿不准的细节宁可删去，不要保留可疑描述。

输出：修正后的提示词全文（格式与草稿一致），不要输出校对过程或任何解释。"""
        final_prompt = s.chat([{"role": "user", "content": s.frames_content(frames, times) + [{"type": "text", "text": review_prompt}]}],
                              s.review_model, "E", max_tokens=3000, temperature=0.1)
        markers = ("subject_definitions:",) if s.prompt_format == "official_v1" else ("构图与人物", "构图与场景")
        for marker in markers:
            idx = final_prompt.find(marker)
            if idx > 0:
                final_prompt = final_prompt[idx:].strip()
                break
        s.cache[review_key] = final_prompt
        s.save_cache()

    # ---------- 固定风格合约头（sp_v4 程序化焊死；official_v1 已进入六字段壳） ----------
    if s.prompt_format == "official_v1":
        final_with_header = final_prompt.strip()
    else:
        final_with_header = FIXED_HEADER + "\n\n" + final_prompt
    s.cache["final_prompt"] = final_with_header
    cost = {
        "calls": len(s.call_log),
        "prompt_tokens": sum(c["prompt_tokens"] for c in s.call_log if c["prompt_tokens"] > 0),
        "completion_tokens": sum(c["completion_tokens"] for c in s.call_log if c["completion_tokens"] > 0),
        "seconds": round(time.time() - t_start, 1),
        "fast_model": s.fast_model,
        "review_model": s.review_model,
        "prompt_format": s.prompt_format,
    }
    s.cache["result_meta"] = {"cost": cost, "arch": arch, "camera": camera, "draft": draft,
                              "prompt_format": s.prompt_format}
    s.save_cache()
    logger(f"> 反推完成[{s.prompt_format}]: {cost['calls']} 次调用 {cost['seconds']}s，输入 {cost['prompt_tokens']} / 输出 {cost['completion_tokens']} tokens")
    return {"prompt": final_with_header, "draft": draft, "arch": arch,
            "camera": camera, "cost": cost, "cached": False, "prompt_format": s.prompt_format}
