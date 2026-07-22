import tempfile
import unittest
from pathlib import Path

import numpy as np

from spvideo.sam3_tracker import SAM3Tracker


class _FakePredictor:
    def __init__(self):
        self.closed = False

    def start_session(self, video_path, session_id):
        return None

    def add_prompt(self, session_id, frame_idx, text):
        return self._result(frame_idx)

    def handle_stream_request(self, request):
        yield self._result(1)

    def close_session(self, session_id):
        self.closed = True

    @staticmethod
    def _result(frame_idx):
        mask_a = np.zeros((8, 8), dtype=bool)
        mask_b = np.zeros((8, 8), dtype=bool)
        mask_a[1:5, 1:3] = True
        mask_b[2:7, 5:7] = True
        return {
            "frame_index": frame_idx,
            "outputs": {
                "out_obj_ids": [3, 7],
                "out_binary_masks": [mask_a, mask_b],
                "out_boxes_xywh": [[0.1, 0.1, 0.2, 0.5], [0.6, 0.2, 0.2, 0.6]],
                "out_probs": [0.9, 0.8],
            },
        }


class Sam3TextObjectsTests(unittest.TestCase):
    def test_preserves_every_text_prompt_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracker = SAM3Tracker.__new__(SAM3Tracker)
            tracker._session_count = 0
            tracker._vp = _FakePredictor()

            result = tracker.track_text_objects(
                "video.mp4",
                "person",
                frame_idx=0,
                max_frames=2,
                output_dir=tmp,
            )

            self.assertEqual(result["object_ids"], [3, 7])
            self.assertEqual(result["object_count"], 2)
            self.assertEqual(result["objects"][3]["tracked_frames"], 2)
            self.assertEqual(result["objects"][7]["tracked_frames"], 2)
            self.assertTrue((Path(tmp) / "object_3" / "mask_0001.png").exists())
            self.assertTrue((Path(tmp) / "object_7" / "mask_0001.png").exists())
            self.assertTrue(tracker._vp.closed)


if __name__ == "__main__":
    unittest.main()
