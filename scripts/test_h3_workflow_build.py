"""离线构造两个 MiniMax H3 工作流并断言关键字段被正确覆盖（不提交 ComfyUI）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spvideo.minimax_h3_client import (  # noqa: E402
    build_charswap_workflow,
    build_white_mask_workflow,
)


def main() -> None:
    # ---- 白膜生成工作流 ----
    wf = build_white_mask_workflow(
        "my_clip.mp4",
        prompt="自定义提示词",
        width=512,
        height=896,
        length=100,
        steps=30,
        seed=12345,
    )
    assert wf["140"]["inputs"]["file"] == "my_clip.mp4", wf["140"]
    h3 = wf["136"]["inputs"]
    assert h3["prompt"] == "自定义提示词"
    assert h3["width"] == 512 and h3["height"] == 896 and h3["length"] == 100
    assert h3["ref_videos.ref_video_0"] == ["141", 0]
    assert not any(k.startswith("ref_images.") for k in h3), "白膜工作流不应有 ref_images"
    assert wf["124"]["inputs"]["steps"] == 30
    assert wf["129"]["inputs"]["noise_seed"] == 12345
    assert wf["92"]["inputs"]["filename_prefix"].startswith("h3_whitemask")
    print("[ok] white-mask: 140 file / 136 prompt+size / 124 steps / 129 seed 覆盖正确")

    # 默认提示词保留模板现值
    wf_default = build_white_mask_workflow("a.mp4")
    assert "彩色树脂关节人偶" in wf_default["136"]["inputs"]["prompt"]
    assert wf_default["124"]["inputs"]["steps"] == 20
    assert isinstance(wf_default["129"]["inputs"]["noise_seed"], int)
    print("[ok] white-mask: 默认 prompt/steps 来自模板，seed 为 None 时随机")

    # ---- 白膜换人工作流（2 张图，与模板一致）----
    wf2 = build_charswap_workflow(
        "mask.mp4",
        ["char1.png", "char2.png"],
        width=512,
        height=896,
        length=97,
        steps=25,
        seed=999,
    )
    assert wf2["140"]["inputs"]["file"] == "mask.mp4"
    h3b = wf2["136"]["inputs"]
    assert h3b["width"] == 512 and h3b["height"] == 896 and h3b["length"] == 97
    assert h3b["ref_images.ref_image_0"] == ["150", 0]
    assert h3b["ref_images.ref_image_1"] == ["151", 0]
    assert "ref_images.ref_image_2" not in h3b
    assert wf2["150"]["inputs"]["image"] == "char1.png"
    assert wf2["151"]["inputs"]["image"] == "char2.png"
    assert wf2["124"]["inputs"]["steps"] == 25
    assert wf2["129"]["inputs"]["noise_seed"] == 999
    assert wf2["92"]["inputs"]["filename_prefix"].startswith("h3_wmswap")
    print("[ok] charswap(2 图): ref_images 数量=2，LoadImage 150/151 正确")

    # ---- 1 张图：应删掉 151 节点和 ref_image_1 引用 ----
    wf1 = build_charswap_workflow("mask.mp4", ["only.png"])
    h3c = wf1["136"]["inputs"]
    assert h3c["ref_images.ref_image_0"] == ["150", 0]
    assert "ref_images.ref_image_1" not in h3c
    assert "151" not in wf1
    assert wf1["150"]["inputs"]["image"] == "only.png"
    print("[ok] charswap(1 图): 多余的 LoadImage 节点和 ref_image_1 引用已移除")

    # ---- 3 张图：应新增 152 节点 ----
    wf3 = build_charswap_workflow("mask.mp4", ["a.png", "b.png", "c.png"])
    h3d = wf3["136"]["inputs"]
    assert h3d["ref_images.ref_image_2"] == ["152", 0]
    assert wf3["152"]["inputs"]["image"] == "c.png"
    print("[ok] charswap(3 图): 新增 LoadImage 152 和 ref_image_2 引用")

    # ---- 超界应报错 ----
    for bad in ([], ["a", "b", "c", "d"]):
        try:
            build_charswap_workflow("m.mp4", bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {len(bad)} images")
    print("[ok] charswap: 0 张 / 4 张参考图正确抛 ValueError")

    print("\n全部断言通过。")


if __name__ == "__main__":
    main()
