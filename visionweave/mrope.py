"""Construct integer-scaled M-RoPE from routed token coordinates."""

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from . import POSITION_SCALE

__all__ = ["VisualRun", "positions_for_frame", "routed_mrope_positions"]


@dataclass(frozen=True)
class VisualRun:
    """One `<|vision_start|>`-delimited placeholder run: an image, or one frame of a video."""

    start: int
    "Index into `input_ids` of the run's FIRST placeholder token."
    positions_x2: np.ndarray
    "`(n, 3)` scaled `(t, h, w)`, relative to the run: `t` is 0 for every token."
    native_span: int
    "`max(h, w) // fine` for this run's patch grid, UNSCALED."


def positions_for_frame(item_positions_x2: np.ndarray, start: int, length: int) -> np.ndarray:
    """The slice of an item's positions belonging to one frame, with temporal zeroed."""
    frame = item_positions_x2[start : start + length].copy()
    frame[:, 0] = 0
    return frame


def routed_mrope_positions(total_length: int, runs: Sequence[VisualRun]) -> Tuple[np.ndarray, int]:
    """`(positions (3, total_length) int64, mrope_position_delta)`.

    `runs` must be sorted by `start` and non-overlapping; the gaps between them are text.
    """
    ps = POSITION_SCALE
    positions = np.zeros((3, total_length), dtype=np.int64)
    cur = 0
    cursor = 0
    for run in runs:
        if run.start < cursor:
            raise ValueError(
                f"visual runs overlap: run at {run.start} starts before the previous one ended at {cursor}."
            )
        text_len = run.start - cursor
        if text_len:
            positions[:, cursor : run.start] = np.arange(text_len, dtype=np.int64) * ps + cur
            cur += ps * text_len
        n = int(run.positions_x2.shape[0])
        if n == 0:
            raise ValueError(f"visual run at {run.start} is empty.")
        positions[:, run.start : run.start + n] = run.positions_x2.T.astype(np.int64) + cur
        cur += ps * run.native_span
        cursor = run.start + n
    if cursor > total_length:
        raise ValueError(f"visual runs cover {cursor} tokens but input_ids has {total_length}.")
    if cursor < total_length:
        text_len = total_length - cursor
        positions[:, cursor:] = np.arange(text_len, dtype=np.int64) * ps + cur
    delta = int(positions.max()) + ps * (1 - total_length)
    return (positions, delta)
