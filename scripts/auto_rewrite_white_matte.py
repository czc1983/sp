from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spvideo.scail2_client import COMFY_URL, Scail2Client


DEFAULT_INPUT = Path(r"C:\Users\Administrator\Desktop\短剧\ai剪辑\177\77.mp4")
DEFAULT_PROMPT = """
Transform the input video into a high-resolution de-identified 3D white clay mannequin control video.
Preserve the exact number of people, original body motion, timing, head direction, gaze direction,
contact relationship, camera movement, framing, lens perspective, foreground/background occlusion,
and spatial layout from the source video.

All visible humans must become smooth white clay mannequins with simplified neutral clothing.
Use pure white and light gray clay render, soft studio lighting, clear sculpted 3D form,
clear body volume, readable facial direction, subtle mouth opening and expression structure only.

Do not keep realistic identity, face texture, skin color, hair texture, original hairstyle,
original clothing details, fabric pattern, jewelry, makeup, subtitles, watermark, text, color, or realistic human pixels.
Do not add people, remove people, change positions, change actions, change camera angle, change scene layout,
or create new choreography.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit one source video to the scail2 Wan2.2/Bernini rv2v path and save a de-identified white-clay matte video.",
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help=f"Source video path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="",
        help="Output mp4 path. Default: <source folder>/自动重绘白膜.mp4",
    )
    parser.add_argument(
        "--comfy",
        default=COMFY_URL,
        help=f"ComfyUI server URL. Default from SCAIL2_COMFY_URL/COMFY_URL or {COMFY_URL}",
    )
    parser.add_argument(
        "--backend",
        choices=["scail2", "runninghub"],
        default="scail2",
        help="Generation backend. scail2 uses the ComfyUI pod; runninghub uses the exported Bernini workflow.",
    )
    parser.add_argument(
        "--preset",
        choices=["fast", "balanced", "quality"],
        default="balanced",
        help="Sampler preset. fast=49 frames, balanced/quality=81 frames.",
    )
    parser.add_argument("--rate", type=int, default=24, help="Forced source frame rate.")
    parser.add_argument(
        "--frames",
        type=int,
        default=81,
        help="Max frames to render before sampler cap. 81 frames at 24fps is about 3.4s.",
    )
    parser.add_argument("--width", type=int, default=512, help="Output width before aspect normalization.")
    parser.add_argument("--height", type=int, default=896, help="Output height before aspect normalization.")
    parser.add_argument(
        "--no-normalize-size",
        action="store_true",
        help="Keep the explicit width/height instead of matching the source aspect.",
    )
    parser.add_argument(
        "--runninghub-workflow",
        default="",
        help="Optional RunningHub Bernini workflow JSON path.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Positive prompt for clay/white-matte redraw.",
    )
    parser.add_argument(
        "--reference",
        default="",
        help="Optional white-clay mannequin reference image. If omitted, a simple neutral reference is generated.",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Run prompt-only rv2v with no reference image.",
    )
    return parser.parse_args()


def make_default_reference(path: Path, width: int = 768, height: int = 1344) -> Path:
    import cv2
    import numpy as np

    canvas = np.full((height, width, 3), 226, dtype=np.uint8)

    def ellipse(center, axes, angle=0, color=(246, 246, 246), thickness=-1):
        cv2.ellipse(canvas, center, axes, angle, 0, 360, color, thickness, cv2.LINE_AA)

    def capsule(p1, p2, radius, color=(244, 244, 244)):
        cv2.line(canvas, p1, p2, color, radius * 2, cv2.LINE_AA)
        cv2.circle(canvas, p1, radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, radius, color, -1, cv2.LINE_AA)

    cx = width // 2
    # Soft shadow and clay form.
    ellipse((cx, int(height * 0.92)), (190, 38), 0, (196, 196, 196))
    ellipse((cx, int(height * 0.16)), (88, 112), 0)
    ellipse((cx, int(height * 0.34)), (132, 178), 0)
    ellipse((cx, int(height * 0.51)), (150, 112), 0)
    capsule((cx - 160, int(height * 0.32)), (cx - 220, int(height * 0.56)), 42)
    capsule((cx + 160, int(height * 0.32)), (cx + 220, int(height * 0.56)), 42)
    capsule((cx - 70, int(height * 0.58)), (cx - 98, int(height * 0.84)), 50)
    capsule((cx + 70, int(height * 0.58)), (cx + 98, int(height * 0.84)), 50)

    # Minimal facial direction only, still identity-free.
    ellipse((cx - 32, int(height * 0.15)), (13, 5), 0, (205, 205, 205))
    ellipse((cx + 32, int(height * 0.15)), (13, 5), 0, (205, 205, 205))
    capsule((cx - 22, int(height * 0.22)), (cx + 22, int(height * 0.22)), 4, (198, 198, 198))

    # Studio lighting: brighten upper-left, vignette lower-right.
    yy, xx = np.mgrid[0:height, 0:width]
    light = 1.08 - 0.22 * ((xx / width) * 0.7 + (yy / height) * 0.9)
    canvas = np.clip(canvas.astype(np.float32) * light[..., None], 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), canvas)
    return path


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.is_file():
        raise FileNotFoundError(f"Source video not found: {video_path}")

    output_path = Path(args.output) if args.output else video_path.with_name("自动重绘白膜.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[input] {video_path}")
    print(f"[backend] {args.backend}")
    if args.backend == "scail2":
        print(f"[server] {args.comfy}")
    print(f"[output] {output_path}")
    print("[mode] WanAnimatePlus Bernini rv2v, clay redraw")

    ref_images: list[str] = []
    if not args.no_reference:
        if args.reference:
            reference_path = Path(args.reference)
        else:
            reference_path = ROOT / ".tmp" / "auto_white_clay_mannequin_reference.png"
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            make_default_reference(reference_path)
        if not reference_path.is_file():
            raise FileNotFoundError(f"Reference image not found: {reference_path}")
        ref_images = [str(reference_path)]
        print(f"[reference] {reference_path}")

    video_window = {
        "force_rate": int(args.rate),
        "frame_load_cap": int(args.frames),
        "skip_first_frames": 0,
        "select_every_nth": 1,
    }
    if args.backend == "runninghub":
        from spvideo.runninghub_client import RunningHubClient

        client = RunningHubClient(workflow_path=args.runninghub_workflow or None)
        result = client.transfer_bernini(
            video_path=str(video_path),
            ref_images=ref_images,
            role_names=["white clay mannequin"] if ref_images else [],
            positive_prompt=args.prompt,
            width=int(args.width),
            height=int(args.height),
            video_window=video_window,
            normalize_size=not args.no_normalize_size,
            on_progress=lambda message: print(f"> {message}", flush=True),
        )
    else:
        client = Scail2Client(args.comfy)
        result = client.transfer_bernini_test(
            video_path=str(video_path),
            ref_images=ref_images,
            role_names=["white clay mannequin"] if ref_images else [],
            positive_prompt=args.prompt,
            width=int(args.width),
            height=int(args.height),
            video_window=video_window,
            sampler_preset=args.preset,
            normalize_size=not args.no_normalize_size,
            on_progress=lambda message: print(f"> {message}", flush=True),
        )

    generated_path = Path(str(result["output_path"]))
    if generated_path.resolve() != output_path.resolve():
        shutil.copy2(generated_path, output_path)

    print(f"[done] {output_path}")
    print(f"[raw] {generated_path}")
    print(f"[workflow] {result.get('workflow_path')}")
    print(f"[task] {result.get('prompt_id')}")


if __name__ == "__main__":
    main()
