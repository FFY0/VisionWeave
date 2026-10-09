"""Unit tests for hard route layouts and wire payload validation."""

import itertools
import unittest

import numpy as np

from visionweave import POSITION_SCALE
from visionweave.route import (
    RouteItem,
    RoutePayload,
    block_count,
    hard_route_layout,
    native_token_count,
)

GRIDS = [(1, 4, 4), (1, 4, 8), (1, 8, 4), (1, 8, 12), (2, 8, 8), (3, 12, 8), (1, 16, 16)]
MASK_SEEDS = ["none", "all", 0, 1, 2]


def reference_hard_layout(grid, compressed_mask):
    """(gather, is_compressed, positions_unscaled, frame_lengths) by literal transcription."""
    t, h, w = grid
    native_h, native_w = (h // 2, w // 2)
    block_h, block_w = (h // 4, w // 4)
    native_positions = []
    native_block_indices = []
    for frame in range(t):
        for row in range(native_h):
            for col in range(native_w):
                native_positions.append((float(frame), float(row), float(col)))
                native_block_indices.append(
                    frame * block_h * block_w + row // 2 * block_w + col // 2
                )
    block_centers = []
    top_left_by_block = []
    for frame in range(t):
        for block_row in range(block_h):
            for block_col in range(block_w):
                block_centers.append((float(frame), block_row * 2 + 0.5, block_col * 2 + 0.5))
                top_left_by_block.append(
                    frame * native_h * native_w + block_row * 2 * native_w + block_col * 2
                )
    gather, flags, positions = ([], [], [])
    frame_lengths = [0] * t
    for native_index, block_index in enumerate(native_block_indices):
        frame = native_index // (native_h * native_w)
        if compressed_mask[block_index]:
            if native_index != top_left_by_block[block_index]:
                continue
            gather.append(block_index)
            flags.append(True)
            positions.append(block_centers[block_index])
        else:
            gather.append(native_index)
            flags.append(False)
            positions.append(native_positions[native_index])
        frame_lengths[frame] += 1
    return (gather, flags, positions, frame_lengths)


def mask_for(grid, seed):
    blocks = block_count(grid)
    if seed == "none":
        return np.zeros(blocks, dtype=bool)
    if seed == "all":
        return np.ones(blocks, dtype=bool)
    return np.random.default_rng(seed).random(blocks) < 0.5


def payload_for(items, fingerprint="fp", threshold=0.5):
    built = [
        RouteItem.from_layout(modality, grid, mask, hard_route_layout(grid, mask))
        for modality, grid, mask in items
    ]
    return RoutePayload(fingerprint=fingerprint, threshold=threshold, items=tuple(built))


class TestLayoutParity(unittest.TestCase):
    def test_matches_reference(self):
        for grid, seed in itertools.product(GRIDS, MASK_SEEDS):
            with self.subTest(grid=grid, seed=seed):
                mask = mask_for(grid, seed)
                layout = hard_route_layout(grid, mask)
                gather, flags, positions, frame_lengths = reference_hard_layout(grid, mask.tolist())
                self.assertEqual(layout.gather_index.tolist(), gather)
                self.assertEqual(layout.is_compressed.tolist(), flags)
                self.assertEqual(layout.frame_lengths, tuple(frame_lengths))
                scaled = [[POSITION_SCALE * axis for axis in row] for row in positions]
                self.assertTrue(
                    all((float(v).is_integer() for row in scaled for v in row)),
                    "POSITION_SCALE must make every routed position integral",
                )
                self.assertEqual(
                    layout.positions_x2.tolist(), [[int(v) for v in row] for row in scaled]
                )

    def test_routed_length_identity(self):
        for grid, seed in itertools.product(GRIDS, ["none", "all", 3]):
            with self.subTest(grid=grid, seed=seed):
                mask = mask_for(grid, seed)
                layout = hard_route_layout(grid, mask)
                compressed = int(mask.sum())
                self.assertEqual(layout.output_length, native_token_count(grid) - 3 * compressed)
                self.assertEqual(layout.output_length, 4 * block_count(grid) - 3 * compressed)

    def test_threshold_extremes(self):
        grid = (1, 8, 8)
        kept = hard_route_layout(grid, np.zeros(block_count(grid), dtype=bool))
        squeezed = hard_route_layout(grid, np.ones(block_count(grid), dtype=bool))
        self.assertEqual(kept.output_length, native_token_count(grid))
        self.assertEqual(squeezed.output_length * 4, native_token_count(grid))
        self.assertFalse(kept.is_compressed.any())
        self.assertTrue(squeezed.is_compressed.all())

    def test_positions_unique_per_item(self):
        for grid, seed in itertools.product(GRIDS, ["none", "all", 4]):
            with self.subTest(grid=grid, seed=seed):
                layout = hard_route_layout(grid, mask_for(grid, seed))
                rows = {tuple(row) for row in layout.positions_x2.tolist()}
                self.assertEqual(len(rows), layout.output_length)

    def test_gather_index_stays_in_range(self):
        for grid, seed in itertools.product(GRIDS, ["none", "all", 5]):
            with self.subTest(grid=grid, seed=seed):
                layout = hard_route_layout(grid, mask_for(grid, seed))
                native = layout.gather_index[~layout.is_compressed]
                compressed = layout.gather_index[layout.is_compressed]
                if native.size:
                    self.assertGreaterEqual(int(native.min()), 0)
                    self.assertLess(int(native.max()), native_token_count(grid))
                if compressed.size:
                    self.assertGreaterEqual(int(compressed.min()), 0)
                    self.assertLess(int(compressed.max()), block_count(grid))

    def test_unaligned_grid_rejected(self):
        for grid in [(1, 4, 6), (1, 6, 4), (2, 8, 10)]:
            with self.subTest(grid=grid):
                with self.assertRaisesRegex(ValueError, "divisible by 4"):
                    hard_route_layout(grid, np.zeros(max(block_count(grid), 1), dtype=bool))

    def test_mask_length_checked(self):
        with self.assertRaisesRegex(ValueError, "routing blocks"):
            hard_route_layout((1, 8, 8), np.zeros(3, dtype=bool))


class TestPayload(unittest.TestCase):
    def test_roundtrip(self):
        payload = payload_for(
            [
                ("image", (1, 8, 12), mask_for((1, 8, 12), 5)),
                ("video", (3, 8, 8), mask_for((3, 8, 8), 6)),
            ]
        )
        restored = RoutePayload.from_dict(payload.to_dict())
        self.assertEqual(restored.output_length, payload.output_length)
        self.assertEqual(restored.native_tokens, payload.native_tokens)
        self.assertEqual(restored.compressed_count, payload.compressed_count)
        self.assertEqual(restored.total_blocks, payload.total_blocks)
        for original, copy in zip(payload.items, restored.items):
            self.assertEqual(copy.modality, original.modality)
            self.assertEqual(copy.grid, original.grid)
            self.assertEqual(copy.frame_lengths, original.frame_lengths)
            self.assertTrue(np.array_equal(copy.compressed_mask, original.compressed_mask))
            self.assertTrue(np.array_equal(copy.positions_x2, original.positions_x2))
            self.assertTrue(np.array_equal(copy.is_compressed, original.is_compressed))
        restored.validate(fingerprint="fp", threshold=0.5, embedding_rows=payload.output_length)

    def test_rejects_skew(self):
        payload = payload_for([("image", (1, 8, 8), mask_for((1, 8, 8), 7))])
        rows = payload.output_length
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            payload.validate(fingerprint="other", threshold=0.5, embedding_rows=rows)
        with self.assertRaisesRegex(ValueError, "threshold"):
            payload.validate(fingerprint="fp", threshold=0.4, embedding_rows=rows)
        with self.assertRaisesRegex(ValueError, "embedding rows"):
            payload.validate(fingerprint="fp", threshold=0.5, embedding_rows=rows + 1)

    def test_rejects_schema_and_comparison_drift(self):
        payload = payload_for([("image", (1, 8, 8), mask_for((1, 8, 8), 10))])
        rows = payload.output_length
        raw = payload.to_dict()
        raw["schema_version"] = raw["schema_version"] + 1
        with self.assertRaisesRegex(ValueError, "schema"):
            RoutePayload.from_dict(raw).validate(
                fingerprint="fp", threshold=0.5, embedding_rows=rows
            )
        raw = payload.to_dict()
        raw["comparison"] = ">"
        with self.assertRaisesRegex(ValueError, "comparison"):
            RoutePayload.from_dict(raw).validate(
                fingerprint="fp", threshold=0.5, embedding_rows=rows
            )

    def test_rejects_tampered_positions(self):
        payload = payload_for([("image", (1, 8, 8), mask_for((1, 8, 8), 8))])
        raw = payload.to_dict()
        positions = np.asarray(raw["items"][0]["positions_x2"]).copy()
        positions[0, 1] += 2
        raw["items"][0]["positions_x2"] = positions
        with self.assertRaisesRegex(ValueError, "positions disagree"):
            RoutePayload.from_dict(raw).validate(
                fingerprint="fp", threshold=0.5, embedding_rows=payload.output_length
            )

    def test_rejects_tampered_flags(self):
        payload = payload_for([("image", (1, 8, 8), mask_for((1, 8, 8), 11))])
        raw = payload.to_dict()
        flags = np.unpackbits(
            np.frombuffer(raw["items"][0]["is_compressed_packed"], dtype=np.uint8),
            count=payload.items[0].output_length,
        ).astype(bool)
        flags[0] = ~flags[0]
        raw["items"][0]["is_compressed_packed"] = np.packbits(flags).tobytes()
        with self.assertRaises(ValueError):
            RoutePayload.from_dict(raw).validate(
                fingerprint="fp", threshold=0.5, embedding_rows=payload.output_length
            )

    def test_rejects_mask_length_mismatch(self):
        payload = payload_for([("video", (2, 8, 8), mask_for((2, 8, 8), 9))])
        raw = payload.to_dict()
        raw["items"][0]["compressed_mask_bits"] = int(raw["items"][0]["compressed_mask_bits"]) - 1
        with self.assertRaises(ValueError):
            RoutePayload.from_dict(raw).validate(
                fingerprint="fp", threshold=0.5, embedding_rows=payload.output_length
            )

    def test_rejects_empty_payload(self):
        with self.assertRaisesRegex(ValueError, "no media items"):
            RoutePayload(fingerprint="fp", threshold=0.5, items=()).validate(
                fingerprint="fp", threshold=0.5, embedding_rows=0
            )


if __name__ == "__main__":
    unittest.main()
