"""The routed layout: one implementation of "which token comes out where", shared by both sides."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np

from . import POSITION_SCALE, ROUTE_COMPARISON

SCHEMA_VERSION = 1
_MODALITIES = ("image", "video")
__all__ = [
    "SCHEMA_VERSION",
    "RouteItem",
    "RouteLayout",
    "RoutePayload",
    "block_count",
    "hard_route_layout",
    "native_token_count",
]


def native_token_count(grid: Sequence[int]) -> int:
    """Native (post-merger) visual tokens for one pre-merger `(t, h, w)` grid."""
    t, h, w = (int(x) for x in grid)
    return t * (h // 2) * (w // 2)


def block_count(grid: Sequence[int]) -> int:
    """Routing blocks (`4 x 4` pre-merger patches each) for one `(t, h, w)` grid."""
    t, h, w = (int(x) for x in grid)
    return t * (h // 4) * (w // 4)


def _check_grid(grid: Sequence[int]) -> Tuple[int, int, int]:
    t, h, w = (int(x) for x in grid)
    if t <= 0 or h <= 0 or w <= 0:
        raise ValueError(f"grid dimensions must be positive, got {(t, h, w)}.")
    if h % 4 or w % 4:
        raise ValueError(
            f"routing needs h and w divisible by 4 (one routing block is 4x4 pre-merger patches), got h={h} w={w}. The media processor must resize on patch_size*4 = 64; check that the resize-factor patches ran in this process."
        )
    return (t, h, w)


@dataclass(frozen=True)
class RouteLayout:
    """Where every routed token comes from, and where it sits."""

    gather_index: np.ndarray
    is_compressed: np.ndarray
    positions_x2: np.ndarray
    frame_lengths: Tuple[int, ...]

    @property
    def output_length(self) -> int:
        return int(self.gather_index.shape[0])


def hard_route_layout(grid: Sequence[int], compressed_mask: Iterable[bool]) -> RouteLayout:
    """The routed layout for one media item, from its grid and the router's per-block decision."""
    t, h, w = _check_grid(grid)
    nh, nw = (h // 2, w // 2)
    bh, bw = (h // 4, w // 4)
    blocks_per_frame = bh * bw
    cells_per_frame = nh * nw
    mask = np.asarray(compressed_mask, dtype=bool).reshape(-1)
    if mask.size != t * blocks_per_frame:
        raise ValueError(
            f"compressed_mask has {mask.size} entries but grid {(t, h, w)} has {t * blocks_per_frame} routing blocks."
        )
    mask = mask.reshape(t, bh, bw)
    cell_rows = np.arange(nh, dtype=np.int64)[:, None]
    cell_cols = np.arange(nw, dtype=np.int64)[None, :]
    top_left = (cell_rows % 2 == 0) & (cell_cols % 2 == 0)
    gather_parts = []
    compressed_parts = []
    position_parts = []
    frame_lengths = []
    for frame in range(t):
        cell_compressed = mask[frame][cell_rows // 2, cell_cols // 2]
        order = np.flatnonzero((~cell_compressed | top_left).reshape(-1))
        rows = order // nw
        cols = order % nw
        compressed = cell_compressed.reshape(-1)[order]
        native_index = frame * cells_per_frame + order
        block_index = frame * blocks_per_frame + rows // 2 * bw + cols // 2
        gather_parts.append(np.where(compressed, block_index, native_index))
        compressed_parts.append(compressed)
        position_parts.append(
            np.stack(
                (
                    np.full(order.shape, POSITION_SCALE * frame, dtype=np.int64),
                    np.where(compressed, 4 * (rows // 2) + 1, POSITION_SCALE * rows),
                    np.where(compressed, 4 * (cols // 2) + 1, POSITION_SCALE * cols),
                ),
                axis=1,
            )
        )
        expected = 4 * blocks_per_frame - 3 * int(compressed.sum())
        if order.size != expected:
            raise AssertionError(
                f"frame {frame} of grid {(t, h, w)} emitted {order.size} tokens, expected {expected} for {int(compressed.sum())} compressed blocks."
            )
        frame_lengths.append(int(order.size))
    return RouteLayout(
        gather_index=np.concatenate(gather_parts).astype(np.int64, copy=False),
        is_compressed=np.concatenate(compressed_parts).astype(bool, copy=False),
        positions_x2=np.concatenate(position_parts).astype(np.int32, copy=False),
        frame_lengths=tuple(frame_lengths),
    )


@dataclass(frozen=True)
class RouteItem:
    """One media item's routing decision, as it travels from encoder to tokenizer."""

    modality: str
    grid: Tuple[int, int, int]
    compressed_mask: np.ndarray
    frame_lengths: Tuple[int, ...]
    positions_x2: np.ndarray
    is_compressed: np.ndarray

    @property
    def native_tokens(self) -> int:
        return native_token_count(self.grid)

    @property
    def block_count(self) -> int:
        return block_count(self.grid)

    @property
    def compressed_count(self) -> int:
        return int(np.count_nonzero(self.compressed_mask))

    @property
    def output_length(self) -> int:
        return int(self.positions_x2.shape[0])

    @classmethod
    def from_layout(
        cls,
        modality: str,
        grid: Sequence[int],
        compressed_mask: Iterable[bool],
        layout: RouteLayout,
    ) -> "RouteItem":
        if modality not in _MODALITIES:
            raise ValueError(f"modality must be one of {_MODALITIES}, got {modality!r}.")
        return cls(
            modality=modality,
            grid=tuple((int(x) for x in grid)),
            compressed_mask=np.asarray(compressed_mask, dtype=bool).reshape(-1),
            frame_lengths=layout.frame_lengths,
            positions_x2=layout.positions_x2,
            is_compressed=layout.is_compressed,
        )

    def validate(self) -> None:
        """Recompute the layout from the mask and require the travelled fields to match it."""
        t, _, _ = _check_grid(self.grid)
        if self.compressed_mask.size != self.block_count:
            raise ValueError(
                f"{self.modality} item {self.grid}: mask has {self.compressed_mask.size} entries, expected {self.block_count} blocks."
            )
        expected_length = 4 * self.block_count - 3 * self.compressed_count
        if self.output_length != expected_length:
            raise ValueError(
                f"{self.modality} item {self.grid}: {self.output_length} routed tokens but 4*{self.block_count} - 3*{self.compressed_count} = {expected_length}."
            )
        if self.native_tokens - 3 * self.compressed_count != expected_length:
            raise ValueError(
                f"{self.modality} item {self.grid}: native {self.native_tokens} and compressed {self.compressed_count} do not reconcile with {expected_length} routed tokens."
            )
        if len(self.frame_lengths) != t or sum(self.frame_lengths) != expected_length:
            raise ValueError(
                f"{self.modality} item {self.grid}: frame_lengths {self.frame_lengths} do not cover {t} frames / {expected_length} tokens."
            )
        if self.is_compressed.shape != (expected_length,):
            raise ValueError(
                f"{self.modality} item {self.grid}: is_compressed has shape {self.is_compressed.shape}, expected ({expected_length},)."
            )
        if self.positions_x2.shape != (expected_length, 3):
            raise ValueError(
                f"{self.modality} item {self.grid}: positions have shape {self.positions_x2.shape}, expected ({expected_length}, 3)."
            )
        layout = hard_route_layout(self.grid, self.compressed_mask)
        if not np.array_equal(layout.is_compressed, self.is_compressed):
            raise ValueError(
                f"{self.modality} item {self.grid}: compressed flags disagree with the layout recomputed from the mask -- encoder and language server are running different versions of route.py."
            )
        if not np.array_equal(layout.positions_x2, self.positions_x2):
            raise ValueError(
                f"{self.modality} item {self.grid}: positions disagree with the layout recomputed from the mask -- encoder and language server are running different versions of route.py."
            )

    def to_dict(self) -> Dict[str, Any]:
        import torch

        return {
            "modality": self.modality,
            "grid": [int(x) for x in self.grid],
            "compressed_mask_packed": np.packbits(self.compressed_mask).tobytes(),
            "compressed_mask_bits": int(self.compressed_mask.size),
            "frame_lengths": [int(x) for x in self.frame_lengths],
            "positions_x2": torch.from_numpy(
                np.ascontiguousarray(self.positions_x2, dtype=np.int32)
            ),
            "is_compressed_packed": np.packbits(self.is_compressed).tobytes(),
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RouteItem":
        bits = int(raw["compressed_mask_bits"])
        mask = np.unpackbits(
            np.frombuffer(raw["compressed_mask_packed"], dtype=np.uint8), count=bits
        ).astype(bool)
        positions = np.asarray(raw["positions_x2"], dtype=np.int32).reshape(-1, 3)
        flags = np.unpackbits(
            np.frombuffer(raw["is_compressed_packed"], dtype=np.uint8), count=positions.shape[0]
        ).astype(bool)
        return cls(
            modality=str(raw["modality"]),
            grid=tuple((int(x) for x in raw["grid"])),
            compressed_mask=mask,
            frame_lengths=tuple((int(x) for x in raw["frame_lengths"])),
            positions_x2=positions,
            is_compressed=flags,
        )


@dataclass(frozen=True)
class RoutePayload:
    """Everything the tokenizer needs about one encoder part, plus who produced it."""

    fingerprint: str
    threshold: float
    items: Tuple[RouteItem, ...]
    schema_version: int = SCHEMA_VERSION
    comparison: str = ROUTE_COMPARISON

    @property
    def output_length(self) -> int:
        return sum((item.output_length for item in self.items))

    @property
    def native_tokens(self) -> int:
        return sum((item.native_tokens for item in self.items))

    @property
    def compressed_count(self) -> int:
        return sum((item.compressed_count for item in self.items))

    @property
    def total_blocks(self) -> int:
        return sum((item.block_count for item in self.items))

    def validate(self, *, fingerprint: str, threshold: float, embedding_rows: int) -> None:
        """Reject a payload this server must not act on."""
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"route payload schema {self.schema_version} != {SCHEMA_VERSION} expected by this process; encoder and language server are not the same build."
            )
        if self.comparison != ROUTE_COMPARISON:
            raise ValueError(
                f"route payload threshold comparison {self.comparison!r} != {ROUTE_COMPARISON!r}."
            )
        if self.fingerprint != fingerprint:
            raise ValueError(
                f"route payload fingerprint {self.fingerprint} != {fingerprint}; the encoder and this server are serving different checkpoints or geometries."
            )
        if abs(self.threshold - threshold) > 1e-09:
            raise ValueError(
                f"route payload threshold {self.threshold} != {threshold} configured here; the encoder and language server must use the same threshold."
            )
        if not self.items:
            raise ValueError("route payload carries no media items.")
        for item in self.items:
            item.validate()
        if self.output_length != embedding_rows:
            raise ValueError(
                f"route payload describes {self.output_length} routed tokens but the encoder sent {embedding_rows} embedding rows."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "fingerprint": str(self.fingerprint),
            "threshold": float(self.threshold),
            "comparison": str(self.comparison),
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RoutePayload":
        return cls(
            schema_version=int(raw["schema_version"]),
            fingerprint=str(raw["fingerprint"]),
            threshold=float(raw["threshold"]),
            comparison=str(raw["comparison"]),
            items=tuple((RouteItem.from_dict(item) for item in raw["items"])),
        )
