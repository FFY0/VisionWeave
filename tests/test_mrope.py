import unittest

import numpy as np

from visionweave.mrope import VisualRun, positions_for_frame, routed_mrope_positions
from visionweave.route import hard_route_layout


class RoutedPositionTests(unittest.TestCase):
    def test_compression_preserves_spatial_span_and_following_text_positions(self):
        native = hard_route_layout((1, 4, 4), np.array([False]))
        compressed = hard_route_layout((1, 4, 4), np.array([True]))
        native_pos, native_delta = routed_mrope_positions(7, [VisualRun(2, native.positions_x2, 2)])
        compressed_pos, compressed_delta = routed_mrope_positions(
            4, [VisualRun(2, compressed.positions_x2, 2)]
        )
        np.testing.assert_array_equal(native_pos[:, :2], [[0, 2]] * 3)
        np.testing.assert_array_equal(compressed_pos[:, 2], [4, 5, 5])
        np.testing.assert_array_equal(native_pos[:, -1], [8, 8, 8])
        np.testing.assert_array_equal(compressed_pos[:, -1], native_pos[:, -1])
        # Decode positions advance by two from either prefill's last position.
        self.assertEqual(7 * 2 + native_delta, 10)
        self.assertEqual(4 * 2 + compressed_delta, 10)

    def test_video_frames_reset_temporal_coordinate_without_mutating_payload(self):
        layout = hard_route_layout((2, 4, 4), np.array([False, True]))
        original = layout.positions_x2.copy()
        frame = positions_for_frame(layout.positions_x2, 4, 1)
        np.testing.assert_array_equal(frame, [[0, 1, 1]])
        np.testing.assert_array_equal(layout.positions_x2, original)


if __name__ == "__main__":
    unittest.main()
