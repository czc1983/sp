"""
SAM3 视频人物追踪模块
用法：
    python sam3_tracker.py --video 片段.mp4 --text "a person" --output masks/
"""
import sys
import os
import argparse
import cv2
import numpy as np
import torch

# patch SAM3 checkpoint loading（去掉 detector. 前缀）
import sam3.model_builder as mb
_orig_load = mb._load_checkpoint


def _patched_load(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    new_ckpt = {k.replace("detector.", "", 1): v for k, v in ckpt.items()}
    model.load_state_dict(new_ckpt, strict=False)


mb._load_checkpoint = _patched_load

from sam3.model.sam3_video_predictor import Sam3VideoPredictor

# 默认路径
MODEL_DIR = r"C:\Users\Administrator\.cache\sam3\models\facebook--sam3\snapshots\master"
BPE_PATH = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python312\Lib\site-packages\sam3\assets\bpe_simple_vocab_16e6.txt.gz"


class SAM3Tracker:
    """SAM3 视频人物追踪器"""

    def __init__(self, checkpoint_path=None, bpe_path=None):
        self.checkpoint = checkpoint_path or os.path.join(MODEL_DIR, "sam3.pt")
        self.bpe = bpe_path or BPE_PATH
        self._vp = None
        self._session_count = 0

    @property
    def predictor(self):
        if self._vp is None:
            self._vp = Sam3VideoPredictor(
                checkpoint_path=self.checkpoint,
                bpe_path=self.bpe,
            )
        return self._vp

    def track(
        self,
        video_path: str,
        text_prompt: str,
        output_dir: str | None = None,
    ) -> dict:
        """
        跟踪视频中文本描述的人物

        Args:
            video_path: 视频文件路径
            text_prompt: 要跟踪的目标描述
            output_dir: 可选，保存每帧 mask PNG 的目录

        Returns:
            {num_frames, tracked_frames, masks, boxes, scores}
        """
        self._session_count += 1
        sid = f"track_{self._session_count}"

        predictor = self.predictor
        predictor.start_session(video_path, session_id=sid)
        predictor.add_prompt(sid, frame_idx=0, text=text_prompt)

        return self._propagate_and_collect(sid, output_dir)

    def track_text_objects(
        self,
        video_path: str,
        text_prompt: str,
        frame_idx: int = 0,
        max_frames: int | None = None,
        output_dir: str | None = None,
        propagation_direction: str = "both",
    ) -> dict:
        """Track every SAM3 object returned by a text prompt without collapsing IDs."""
        self._session_count += 1
        sid = f"track_{self._session_count}"
        predictor = self.predictor
        predictor.start_session(video_path, session_id=sid)
        prompt_result = predictor.add_prompt(
            sid,
            frame_idx=frame_idx,
            text=text_prompt,
        )
        result = self._propagate_and_collect_objects(
            sid,
            output_dir=output_dir,
            max_frames=max_frames,
            start_frame_idx=frame_idx,
            initial_results=[prompt_result],
            propagation_direction=propagation_direction,
        )
        result["prompt_mode"] = "text_objects"
        result["propagation_direction"] = propagation_direction
        return result

    def track_by_point(
        self,
        video_path: str,
        point: list,
        frame_idx: int = 0,
        max_frames: int | None = None,
        output_dir: str | None = None,
        propagation_direction: str = "both",
    ) -> dict:
        """用坐标点指定跟踪目标（配合 InsightFace 人脸匹配使用）

        Args:
            video_path: 视频文件路径
            point: 归一化坐标 [x, y]，范围 0-1，指向要跟踪的人脸中心
            output_dir: 可选，保存 mask 目录

        Returns:
            {num_frames, tracked_frames, masks, boxes, scores}
        """
        self._session_count += 1
        sid = f"track_{self._session_count}"

        predictor = self.predictor
        predictor.start_session(video_path, session_id=sid)
        points = _coerce_prompt_points(point)
        prompt_result = predictor.add_prompt(
            sid, frame_idx=frame_idx,
            points=points,
            point_labels=[1] * len(points),  # 1=正样本（追这个人）
            obj_id=1,
        )

        result = self._propagate_and_collect(
            sid,
            output_dir,
            max_frames=max_frames,
            start_frame_idx=frame_idx,
            initial_results=[prompt_result],
            propagation_direction=propagation_direction,
        )
        result["prompt_mode"] = "points"
        result["propagation_direction"] = propagation_direction
        return result

    def track_by_box(
        self,
        video_path: str,
        box_xywh: list,
        point: list | None = None,
        frame_idx: int = 0,
        max_frames: int | None = None,
        output_dir: str | None = None,
        propagation_direction: str = "both",
    ) -> dict:
        self._session_count += 1
        sid = f"track_{self._session_count}"

        predictor = self.predictor
        predictor.start_session(video_path, session_id=sid)
        prompt_result = predictor.add_prompt(
            session_id=sid,
            frame_idx=frame_idx,
            bounding_boxes=[box_xywh],
            bounding_box_labels=[1],
        )

        if not _result_has_mask(prompt_result):
            predictor.close_session(sid)
            _clear_mask_outputs(output_dir)
            result = self.track_by_point(
                video_path=video_path,
                point=_person_points_from_box(box_xywh, point),
                frame_idx=frame_idx,
                max_frames=max_frames,
                output_dir=output_dir,
                propagation_direction=propagation_direction,
            )
            result["prompt_mode"] = "points_after_empty_box"
            result["box_tracked_frames"] = 0
            result["propagation_direction"] = propagation_direction
            return result

        result = self._propagate_and_collect(
            sid,
            output_dir,
            max_frames=max_frames,
            start_frame_idx=frame_idx,
            initial_results=[prompt_result],
            propagation_direction=propagation_direction,
        )
        result["prompt_mode"] = "box"
        result["propagation_direction"] = propagation_direction

        expected_frames = max_frames or result.get("num_frames") or 0
        minimum_coverage = max(1, int(np.ceil(float(expected_frames) * 0.75)))
        if point is not None and int(result.get("tracked_frames") or 0) < minimum_coverage:
            box_tracked_frames = int(result.get("tracked_frames") or 0)
            _clear_mask_outputs(output_dir)
            result = self.track_by_point(
                video_path=video_path,
                point=_person_points_from_box(box_xywh, point),
                frame_idx=frame_idx,
                max_frames=max_frames,
                output_dir=output_dir,
                propagation_direction=propagation_direction,
            )
            result["prompt_mode"] = "points_after_weak_box"
            result["box_tracked_frames"] = box_tracked_frames
            result["propagation_direction"] = propagation_direction
        return result

    def _propagate_and_collect(
        self,
        sid: str,
        output_dir: str | None = None,
        max_frames: int | None = None,
        start_frame_idx: int | None = None,
        initial_results: list[dict] | None = None,
        propagation_direction: str = "both",
    ) -> dict:
        """传播追踪并收集结果"""
        predictor = self.predictor

        request = {
            "type": "propagate_in_video",
            "session_id": sid,
            "propagation_direction": propagation_direction,
        }
        if start_frame_idx is not None:
            request["start_frame_index"] = start_frame_idx
        if max_frames is not None:
            request["max_frame_num_to_track"] = max_frames
        frame_data = {}

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        def collect_result(r: dict) -> None:
            frame_idx = int(r["frame_index"])
            if max_frames is not None and not (0 <= frame_idx < max_frames):
                return
            out = r["outputs"]
            obj_ids = out.get("out_obj_ids")

            raw_masks = out.get("out_binary_masks")
            raw_boxes = out.get("out_boxes_xywh")
            raw_probs = out.get("out_probs")
            mask_count = _sequence_len(raw_masks)

            if obj_ids is not None and len(obj_ids) > 0 and mask_count > 0:
                # 取第一个跟踪目标
                m = raw_masks[0]
                mask = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
                b = raw_boxes[0] if _sequence_len(raw_boxes) > 0 else None
                box = b.cpu().numpy() if hasattr(b, "cpu") else (np.asarray(b) if b is not None else None)
                score = float(raw_probs[0]) if _sequence_len(raw_probs) > 0 else 0.0
            else:
                mask = None
                box = None
                score = 0.0

            if mask is not None and not np.any(mask):
                mask = None
                box = None
                score = 0.0

            previous = frame_data.get(frame_idx)
            if previous is None or (mask is not None and previous["mask"] is None) or score > previous["score"]:
                frame_data[frame_idx] = {"mask": mask, "box": box, "score": score}

        try:
            for item in initial_results or []:
                collect_result(item)
            for item in predictor.handle_stream_request(request):
                collect_result(item)
        finally:
            predictor.close_session(sid)

        if max_frames is not None:
            num_frames = max_frames
        elif frame_data:
            num_frames = max(frame_data) + 1
        else:
            num_frames = 0

        masks = []
        boxes = []
        scores = []
        tracked = 0
        for frame_idx in range(num_frames):
            item = frame_data.get(frame_idx, {"mask": None, "box": None, "score": 0.0})
            mask = item["mask"]
            masks.append(mask)
            boxes.append(item["box"])
            scores.append(item["score"])
            if mask is not None:
                tracked += 1
                if output_dir:
                    mask_img = (mask.astype(np.uint8) * 255)
                    ok, encoded = cv2.imencode(".png", mask_img)
                    if ok:
                        with open(os.path.join(output_dir, f"mask_{frame_idx:04d}.png"), "wb") as file:
                            file.write(encoded.tobytes())

        return {
            "num_frames": num_frames,
            "tracked_frames": tracked,
            "masks": masks,
            "boxes": boxes,
            "scores": scores,
        }

    def _propagate_and_collect_objects(
        self,
        sid: str,
        output_dir: str | None = None,
        max_frames: int | None = None,
        start_frame_idx: int | None = None,
        initial_results: list[dict] | None = None,
        propagation_direction: str = "both",
    ) -> dict:
        predictor = self.predictor
        request = {
            "type": "propagate_in_video",
            "session_id": sid,
            "propagation_direction": propagation_direction,
        }
        if start_frame_idx is not None:
            request["start_frame_index"] = start_frame_idx
        if max_frames is not None:
            request["max_frame_num_to_track"] = max_frames

        frame_objects: dict[int, dict[int, dict]] = {}
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        def collect_result(result: dict) -> None:
            frame_idx = int(result["frame_index"])
            if max_frames is not None and not (0 <= frame_idx < max_frames):
                return
            outputs = result["outputs"]
            obj_ids = outputs.get("out_obj_ids")
            raw_masks = outputs.get("out_binary_masks")
            raw_boxes = outputs.get("out_boxes_xywh")
            raw_probs = outputs.get("out_probs")
            count = min(_sequence_len(obj_ids), _sequence_len(raw_masks))
            if count <= 0:
                return
            objects = frame_objects.setdefault(frame_idx, {})
            for index in range(count):
                obj_id = int(obj_ids[index])
                raw_mask = raw_masks[index]
                mask = raw_mask.cpu().numpy() if hasattr(raw_mask, "cpu") else np.asarray(raw_mask)
                mask = np.squeeze(mask).astype(bool)
                if not np.any(mask):
                    continue
                raw_box = raw_boxes[index] if index < _sequence_len(raw_boxes) else None
                box = (
                    raw_box.cpu().numpy()
                    if hasattr(raw_box, "cpu")
                    else (np.asarray(raw_box) if raw_box is not None else None)
                )
                score = float(raw_probs[index]) if index < _sequence_len(raw_probs) else 0.0
                previous = objects.get(obj_id)
                if previous is None or score > previous["score"]:
                    objects[obj_id] = {"mask": mask, "box": box, "score": score}

        try:
            for item in initial_results or []:
                collect_result(item)
            for item in predictor.handle_stream_request(request):
                collect_result(item)
        finally:
            predictor.close_session(sid)

        if max_frames is not None:
            num_frames = max_frames
        elif frame_objects:
            num_frames = max(frame_objects) + 1
        else:
            num_frames = 0
        object_ids = sorted({obj_id for objects in frame_objects.values() for obj_id in objects})
        objects_result: dict[int, dict] = {}
        for obj_id in object_ids:
            masks = []
            boxes = []
            scores = []
            tracked_frames = 0
            object_dir = os.path.join(output_dir, f"object_{obj_id}") if output_dir else ""
            if object_dir:
                os.makedirs(object_dir, exist_ok=True)
            for frame_idx in range(num_frames):
                item = frame_objects.get(frame_idx, {}).get(obj_id)
                mask = item["mask"] if item is not None else None
                box = item["box"] if item is not None else None
                score = item["score"] if item is not None else 0.0
                masks.append(mask)
                boxes.append(box)
                scores.append(score)
                if mask is None:
                    continue
                tracked_frames += 1
                if object_dir:
                    ok, encoded = cv2.imencode(".png", mask.astype(np.uint8) * 255)
                    if ok:
                        with open(os.path.join(object_dir, f"mask_{frame_idx:04d}.png"), "wb") as file:
                            file.write(encoded.tobytes())
            objects_result[obj_id] = {
                "tracked_frames": tracked_frames,
                "masks": masks,
                "boxes": boxes,
                "scores": scores,
            }
        return {
            "num_frames": num_frames,
            "object_count": len(object_ids),
            "object_ids": object_ids,
            "objects": objects_result,
        }

    def close(self):
        if self._vp is not None:
            del self._vp
            self._vp = None
            torch.cuda.empty_cache()


def _coerce_prompt_points(value: list) -> list[list[float]]:
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    ):
        return [[float(value[0]), float(value[1])]]
    if isinstance(value, (list, tuple)):
        points = [
            [float(item[0]), float(item[1])]
            for item in value
            if isinstance(item, (list, tuple)) and len(item) == 2
        ]
        if points:
            return points
    raise ValueError("point_prompt_required")


def _person_points_from_box(box_xywh: list, preferred: list | None) -> list[list[float]]:
    x, y, width, height = [float(item) for item in box_xywh]
    center_x = max(0.0, min(1.0, x + width * 0.5))
    points = [
        [center_x, max(0.0, min(1.0, y + height * ratio))]
        for ratio in (0.35, 0.52, 0.70)
    ]
    if isinstance(preferred, (list, tuple)) and len(preferred) == 2:
        preferred_point = [float(preferred[0]), float(preferred[1])]
        if all(0.0 <= item <= 1.0 for item in preferred_point):
            points.insert(0, preferred_point)
    return points


def _result_has_mask(result: dict | None) -> bool:
    if not isinstance(result, dict):
        return False
    outputs = result.get("outputs") or {}
    masks = outputs.get("out_binary_masks")
    if _sequence_len(masks) <= 0:
        return False
    for index in range(_sequence_len(masks)):
        mask = masks[index]
        array = mask.cpu().numpy() if hasattr(mask, "cpu") else np.asarray(mask)
        if np.any(array):
            return True
    return False


def _clear_mask_outputs(output_dir: str | None) -> None:
    if not output_dir or not os.path.isdir(output_dir):
        return
    for name in os.listdir(output_dir):
        if name.startswith("mask_") and name.endswith(".png"):
            try:
                os.remove(os.path.join(output_dir, name))
            except FileNotFoundError:
                pass


def _sequence_len(value) -> int:
    if value is None:
        return 0
    if hasattr(value, "shape"):
        try:
            return int(value.shape[0])
        except (TypeError, ValueError, IndexError):
            return 0
    try:
        return len(value)
    except TypeError:
        return 0


def main():
    parser = argparse.ArgumentParser(description="SAM3 视频人物追踪")
    parser.add_argument("--video", "-v", required=True, help="输入视频路径")
    parser.add_argument("--text", "-t", default="a person", help="跟踪目标文本描述")
    parser.add_argument("--output", "-o", default=None, help="输出 mask 图片目录（可选）")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"视频不存在: {args.video}")
        sys.exit(1)

    print(f"视频: {args.video}")
    print(f"目标: {args.text}")

    tracker = SAM3Tracker()
    result = tracker.track(
        video_path=args.video,
        text_prompt=args.text,
        output_dir=args.output,
    )

    print(f"\n总帧数: {result['num_frames']}")
    print(f"跟踪到人: {result['tracked_frames']}/{result['num_frames']}")
    print(f"跟踪率: {result['tracked_frames'] / max(1, result['num_frames']) * 100:.1f}%")

    if result["tracked_frames"] > 0 and args.output:
        print(f"Mask 图片已保存到: {args.output}")

    tracker.close()


if __name__ == "__main__":
    main()
