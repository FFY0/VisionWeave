import os
import unittest
from unittest.mock import patch

import numpy as np

from visionweave import ROUTE_THRESHOLD_ENV, route_threshold
from visionweave.route import hard_route_layout, native_token_count


class RouteThresholdTests(unittest.TestCase):
    def test_missing_threshold_is_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "not set"):
                route_threshold()

    def test_finite_nonnegative_thresholds(self):
        for value in (0.0, 0.5, 1.0, 1.1, 2.0):
            with self.subTest(value=value):
                with patch.dict(os.environ, {ROUTE_THRESHOLD_ENV: str(value)}):
                    self.assertEqual(route_threshold(), value)

    def test_invalid_thresholds_are_rejected(self):
        for value in ("", " ", "invalid", "nan", "inf", "-inf", "-0.1"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {ROUTE_THRESHOLD_ENV: value}):
                    with self.assertRaises(RuntimeError):
                        route_threshold()

    def test_threshold_above_one_keeps_saturated_probabilities_native(self):
        grid = (1, 8, 8)
        probabilities = np.array([0.0, 0.49, 0.5, 1.0], dtype=np.float32)
        with patch.dict(os.environ, {ROUTE_THRESHOLD_ENV: "1.1"}):
            mask = probabilities >= route_threshold()
        layout = hard_route_layout(grid, mask)
        self.assertFalse(mask.any())
        self.assertFalse(layout.is_compressed.any())
        self.assertEqual(layout.output_length, native_token_count(grid))
        self.assertEqual(int((probabilities >= 1.0).sum()), 1)

    def test_trained_threshold_keeps_its_inclusive_boundary(self):
        probabilities = np.array([0.0, 0.49, 0.5, 1.0], dtype=np.float32)
        with patch.dict(os.environ, {ROUTE_THRESHOLD_ENV: "0.5"}):
            mask = probabilities >= route_threshold()
        np.testing.assert_array_equal(mask, [False, False, True, True])


if __name__ == "__main__":
    unittest.main()
