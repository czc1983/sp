from __future__ import annotations

from pathlib import Path

# 优先用专用人声模型；失败时按顺序回退
VOCAL_MODEL_CANDIDATES = [
    "UVR-MDX-NET-Voc_FT.onnx",
    "UVR-MDX-NET-Inst_HQ_3.onnx",  # 伴奏模型：人声 = 原曲 - 伴奏（separate 会自动给两轨）
]


def separate_vocals(
    audio_path: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "",
    model_file_dir: str | Path = "",
    log=None,
) -> Path:
    """用 UVR MDX-NET 模型把人声从背景音乐中分离出来，返回人声音频路径。

    输出文件由 audio-separator 命名，选取文件名里含 Vocals 的那一条。
    """
    from audio_separator.separator import Separator

    _ensure_ffmpeg_on_path()

    source = Path(audio_path)
    if not source.is_file():
        raise FileNotFoundError(f"audio_not_found: {source}")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(model_file_dir) if model_file_dir else out_dir / "_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    candidates = [model_name] if model_name else list(VOCAL_MODEL_CANDIDATES)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            if log:
                log(f"加载人声分离模型 {candidate}")
            separator = Separator(
                output_dir=str(out_dir),
                output_format="WAV",
                model_file_dir=str(model_dir),
                log_level=40,  # ERROR，避免刷进度
            )
            separator.load_model(model_filename=candidate)
            outputs = separator.separate(str(source))
            vocals = _pick_vocals(outputs)
            if vocals is None:
                raise RuntimeError(f"separate_output_missing_vocals: {outputs}")
            return Path(out_dir / vocals) if not Path(vocals).is_absolute() else Path(vocals)
        except Exception as exc:  # noqa: BLE001 - 逐个模型回退
            last_error = exc
            if log:
                log(f"模型 {candidate} 失败：{exc}，尝试下一个")
    raise RuntimeError(f"vocal_separation_failed: {last_error}")


def _pick_vocals(outputs) -> str | None:
    names = [str(item) for item in (outputs or [])]
    for name in names:
        if "vocal" in Path(name).stem.lower() and "inst" not in Path(name).stem.lower():
            return name
    return None


def _pick_instrumental(outputs) -> str | None:
    names = [str(item) for item in (outputs or [])]
    for name in names:
        stem = Path(name).stem.lower()
        if "instrumental" in stem or "accompaniment" in stem:
            return name
    return None


def ensure_stems(
    audio_path: str | Path,
    output_dir: str | Path,
    *,
    log=None,
) -> tuple[Path, Path | None]:
    """确保人声/伴奏两条分离轨都存在，返回 (vocals, instrumental)。

    已分离过（目录里能找到同名 *_Vocals_* 文件）就直接复用，不重复跑模型。
    分离失败时 vocals 回落为原始音频、instrumental 为 None。
    """
    source = Path(audio_path)
    out_dir = Path(output_dir)
    existing_vocals = sorted(out_dir.glob(f"{source.stem}_*(Vocals)*.wav"))
    if existing_vocals:
        vocals = existing_vocals[0]
        inst_matches = sorted(out_dir.glob(f"{source.stem}_*(Instrumental)*.wav"))
        instrumental = inst_matches[0] if inst_matches else None
        return vocals, instrumental
    try:
        from audio_separator.separator import Separator  # noqa: F401
    except Exception:
        return source, None
    vocals = separate_vocals(source, out_dir, log=log)
    inst_matches = sorted(out_dir.glob(f"{source.stem}_*(Instrumental)*.wav"))
    instrumental = inst_matches[0] if inst_matches else None
    return vocals, instrumental


def _ensure_ffmpeg_on_path() -> None:
    """audio-separator 依赖 PATH 上的 ffmpeg；本机用的是 remotion 自带版，手动注入。"""
    import os
    import shutil

    if shutil.which("ffmpeg"):
        return
    try:
        from .ffmpeg_tools import FFMPEG_CANDIDATES, find_binary

        ffmpeg_path = find_binary("", FFMPEG_CANDIDATES)
    except Exception:
        return
    ffmpeg_dir = str(Path(ffmpeg_path).parent)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
