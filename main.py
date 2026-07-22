from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from spvideo.pipeline import run_segmentation


def main() -> int:
    parser = argparse.ArgumentParser(description="SP video planner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split", help="Analyze and split a video into AI workflow buckets.")
    split_parser.add_argument("--input", required=True, help="Input mp4 path.")
    split_parser.add_argument("--project-dir", required=True, help="Output project directory.")
    split_parser.add_argument("--sample-interval", type=float, default=1.0, help="Frame sampling interval in seconds.")
    split_parser.add_argument("--min-segment", type=float, default=2.0, help="Minimum segment duration in seconds.")
    split_parser.add_argument("--max-segment", type=float, default=6.0, help="Maximum AI-friendly segment duration in seconds.")
    split_parser.add_argument("--no-export-video", action="store_true", help="Only analyze; do not cut mp4 clips.")
    split_parser.add_argument("--extract-backgrounds", action="store_true", help="Extract background reference images into 03_ai_inputs/background_refs.")
    split_parser.add_argument("--two-pass", action="store_true", help="Use PySceneDetect + YOLO two-pass detection (more accurate shot boundaries).")
    split_parser.add_argument("--omnishotcut", action="store_true", help="Add OmniShotCut as the first shot-candidate layer; implies --two-pass (requires CUDA).")
    split_parser.add_argument("--sam3-finalize", action="store_true", help="Enable layer-6 SAM3 subject-track finalization when a local executor is available.")
    split_parser.add_argument("--yolo-conf", type=float, default=0.35, help="YOLO person detection confidence threshold (used with --two-pass).")
    split_parser.add_argument("--gemini-concurrency", type=int, default=10, help="Concurrent visual-model identity checks.")
    split_parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="YOLO inference device: auto(autodetect) / cuda(GPU) / cpu.")

    gui_parser = subparsers.add_parser("gui", help="Start PyQt5 desktop interface.")
    gui_parser.add_argument("--project-dir", default="", help="Optional project directory to preload.")

    args = parser.parse_args()

    if args.command == "split":
        device = {"auto": None, "cuda": "cuda", "cpu": "cpu"}[getattr(args, "device", "auto")]
        result = run_segmentation(
            video_path=args.input,
            project_dir=args.project_dir,
            sample_interval=args.sample_interval,
            min_segment_duration=args.min_segment,
            max_segment_duration=args.max_segment,
            export_video=not args.no_export_video,
            extract_backgrounds=getattr(args, "extract_backgrounds", False),
            use_two_pass=getattr(args, "two_pass", False),
            use_omnishotcut=getattr(args, "omnishotcut", False),
            use_sam3_finalize=getattr(args, "sam3_finalize", False),
            yolo_conf_threshold=getattr(args, "yolo_conf", 0.35),
            device=device,
            gemini_identity_concurrency=max(1, args.gemini_concurrency),
        )
        print(json.dumps({"project_dir": result["project_dir"], "segment_count": result["segment_count"]}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "gui":
        try:
            from ui.main_window import run_app
        except ModuleNotFoundError as exc:
            if exc.name == "PyQt5":
                print("PyQt5 is not installed. Install it with: pip install PyQt5", file=sys.stderr)
                return 2
            raise
        return run_app(Path(args.project_dir) if args.project_dir else None)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
