from __future__ import annotations

import unittest

from spvideo.pipeline import _needs_visual_identity_review


class CascadeGatingTests(unittest.TestCase):
    def test_face_confirmed_single_person_skips_remote_visual_review(self) -> None:
        self.assertFalse(
            _needs_visual_identity_review(
                {"person_count": 1, "face_identity_checked": True}
            )
        )

    def test_unconfirmed_or_conflicting_segments_require_visual_review(self) -> None:
        self.assertTrue(_needs_visual_identity_review({"person_count": 1}))
        self.assertTrue(
            _needs_visual_identity_review(
                {
                    "person_count": 1,
                    "face_identity_checked": True,
                    "end_sources": ["sam3_track_lost"],
                }
            )
        )
        self.assertTrue(
            _needs_visual_identity_review(
                {"person_count": 2, "face_identity_checked": True}
            )
        )


if __name__ == "__main__":
    unittest.main()
