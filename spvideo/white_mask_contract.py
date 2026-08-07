# 本模块是白膜合约单一事实源，comfy_workflows 下 3 个 JSON 与 web_ui/story_generate_dashboard.html 中的静态副本须与本模块文案保持一致。
"""SP × MiniMax-H3 白膜合约（WhiteMaskContract）单一事实源。"""

WHITE_MASK_CONTRACT_VERSION = "wm_v1_20260807"

# 官方上限是 Unicode 字符数（len()），不是 utf-8 字节数——中文按字节会虚报 3 倍。
MAX_PROMPT_CHARS = 7000


def validate_final_prompt(prompt: str) -> str:
    """校验最终提示词长度：超过 MAX_PROMPT_CHARS 抛 ValueError，否则原样返回。"""
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(
            f"white_mask_contract: 最终提示词超长，实际 {len(prompt)} 字符，上限 {MAX_PROMPT_CHARS} 字符"
        )
    return prompt


def subject_block() -> str:
    return "把视频里的人物全部转换成彩色树脂关节人偶素体：衣服完全消融进身体，通体同色光滑哑光树脂材质，像BJD娃娃素体；光头，没有头发和睫毛，但保留眉形；身体带球形关节，极简人偶面部；不复制原人物的脸型、五官特征和肤色。每位人物分配一种固定纯色（第1人红色、第2人蓝色、第3人黄色，以此类推），同一人物全片颜色不变，不同人偶颜色要有明显区别。"


def face_retention_block() -> str:
    return "表情、眼神与动作只按可观察部件保留：眼睑开合程度、视线方向、眉形、嘴角、嘴部开合与参考视频逐拍一致，姿态与动作节奏完全跟随原片；禁止用任何情绪结论词概括表情。"


def background_block(background_replace: bool = False) -> str:
    if background_replace:
        return "背景替换为上传的背景图：人物与镜中倒影仍变为对应颜色的人偶，场景结构、道具与镜子位置跟随上传的背景图，人偶光影与新背景协调一致。"
    return "背景、场景道具、家具、镜子及其镜中倒影必须完整保留原样，不得抹白、不得改变、不得简化，只有人物被替换成人偶，镜中的人影倒影同步变成对应颜色的人偶。"


def audio_block() -> str:
    return "声音与原视频完全一致：完整保留原视频原声（语言、声线、语速、情绪均不变），不重新配音、不加配乐；人偶口型与视频声音逐拍对齐。"


def negative_block() -> str:
    return "绝对不要真实皮肤质感、不要头发丝、不要睫毛、不要布料质感，画面中不要出现任何文字、字幕、水印。"


def build_white_mask_prompt(background_replace: bool = False) -> str:
    """按 subject → background → face_retention → audio → negative 顺序拼接完整 prompt。"""
    return "".join([
        subject_block(),
        background_block(background_replace),
        face_retention_block(),
        audio_block(),
        negative_block(),
    ])
