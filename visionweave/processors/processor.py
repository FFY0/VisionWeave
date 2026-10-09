# Portions derived from SGLang.
# Copyright 2025 Qwen Team
# Copyright 2025 SGLang Team
# Licensed under Apache-2.0; see LICENSES/Apache-2.0.txt.
# Modified for the VisionWeave routed serving integration.
"""Build placeholders and M-RoPE from encoder routing metadata."""

import inspect
import logging
from typing import Dict, List, Tuple

import torch
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.multimodal.processors.base_processor import MultimodalProcessorOutput
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

from .. import ARCHITECTURE, config_fingerprint, route_threshold, validate_geometry
from ..core_patches import ROUTE_KWARG
from ..mrope import VisualRun, positions_for_frame, routed_mrope_positions
from ..patches import set_geometry
from ..route import RouteItem, RoutePayload

logger = logging.getLogger(__name__)
_UPSTREAM_BUILDER_MARKERS = (
    "mm_token_num = img_grid_thw[img_idx].prod() // (spatial_merge_size**2)",
    "frame_seqlen = video_grid_thw[video_idx][1:].prod().item() // (",
    'timestamp_text = f"<{curr_time:.1f} seconds>"',
    "video_tokens.extend([video_token_id] * frame_seqlen)",
    "cur_idx = mm_start_idx + 2  # jump to vision_end_id",
)


def _assert_upstream_builder_unchanged() -> None:
    src = inspect.getsource(QwenVLImageProcessor.build_input_ids_with_timestamps)
    missing = [m for m in _UPSTREAM_BUILDER_MARKERS if m not in src]
    if missing:
        raise RuntimeError(
            f"visionweave copied QwenVLImageProcessor.build_input_ids_with_timestamps from sglang df2f34cca, but the installed sglang's version no longer contains: {missing!r}. Re-derive VisionWeaveImageProcessor._build_routed_input_ids before serving."
        )


_assert_upstream_builder_unchanged()


class Qwen3_5VisionWeaveForConditionalGeneration:
    """Name-only stand-in for the entry class, so this module never imports the model."""


assert Qwen3_5VisionWeaveForConditionalGeneration.__name__ == ARCHITECTURE, (
    f"the processor's arch stub is named {Qwen3_5VisionWeaveForConditionalGeneration.__name__!r} but the package's canonical arch name is {ARCHITECTURE!r}; sglang matches PROCESSOR_MAPPING by __name__ against hf_config.architectures, so a mismatch means this processor is never selected and the generic Qwen processor serves native token counts instead."
)


class VisionWeaveImageProcessor(QwenVLImageProcessor):
    models = [Qwen3_5VisionWeaveForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        self.fine, self.pool, self.coarse = validate_geometry(hf_config.vision_config)
        set_geometry(hf_config.vision_config)
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.threshold = route_threshold()
        self.fingerprint = config_fingerprint(hf_config)
        logger.info(
            "[visionweave] processor ready: fine=%d pool=%d coarse=%d threshold=%.6g fingerprint=%s",
            self.fine,
            self.pool,
            self.coarse,
            self.threshold,
            self.fingerprint,
        )

    def get_mm_data(self, prompt, embeddings, **kwargs):
        img_grid_thw = kwargs.get("img_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        video_timestamps = kwargs.get("video_timestamps", None)
        routes = kwargs.get(ROUTE_KWARG, None)
        if routes is None:
            raise ValueError(
                f"the encoder returned embeddings without a VisionWeave route payload. Without it the routed length is unknown, and building placeholders from the grid would give the native count. Check that the encoder process loaded {ARCHITECTURE} (not the base architecture)."
            )
        if kwargs.get("audio_feature_lens") is not None:
            raise ValueError("VisionWeave does not support audio input.")
        routes = _group_routes_by_modality(routes)
        _check_route_arity(routes, img_grid_thw, video_grid_thw)
        image_items = self._payload_items(routes, Modality.IMAGE, embeddings, img_grid_thw)
        video_items = self._payload_items(routes, Modality.VIDEO, embeddings, video_grid_thw)
        input_ids, offsets, modality_list, runs = self._build_routed_input_ids(
            prompt, image_items, video_items, video_timestamps
        )
        positions, delta = routed_mrope_positions(len(input_ids), runs)
        mm_items = []
        consumed: Dict[Modality, int] = {}
        for modality, offset in zip(modality_list, offsets):
            num_tokens = offset[1] - offset[0] + 1
            start = consumed.get(modality, 0)
            mm_items.append(
                MultimodalDataItem(
                    modality=modality,
                    offsets=[offset],
                    precomputed_embeddings=embeddings[modality][start : start + num_tokens],
                )
            )
            consumed[modality] = start + num_tokens
        for modality, used in consumed.items():
            available = int(embeddings[modality].shape[0])
            if used != available:
                raise ValueError(
                    f"{modality} placeholders consume {used} embedding rows but the encoder sent {available}; the route payload and the embeddings disagree."
                )
        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=mm_items,
            im_start_id=self.IM_START_TOKEN_ID,
            im_end_id=self.IM_END_TOKEN_ID,
            im_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
            audio_token_id=self.mm_tokens.audio_token_id,
            mrope_positions=torch.from_numpy(positions),
            mrope_position_delta=torch.tensor([[delta]], dtype=torch.long),
        )

    def _payload_items(self, routes, modality, embeddings, grid_thw) -> List[RouteItem]:
        """Validate this modality's payloads and flatten them into one list, in part order."""
        payloads = routes.get(modality, [])
        if not payloads:
            if grid_thw is not None and len(grid_thw):
                raise ValueError(
                    f"the encoder sent {len(grid_thw)} {modality} grid(s) but no route payload."
                )
            return []
        rows = int(embeddings[modality].shape[0])
        items: List[RouteItem] = []
        seen = 0
        for raw in payloads:
            payload = RoutePayload.from_dict(raw)
            payload.validate(
                fingerprint=self.fingerprint,
                threshold=self.threshold,
                embedding_rows=payload.output_length,
            )
            for item in payload.items:
                if item.modality != _MODALITY_NAMES[modality]:
                    raise ValueError(
                        f"route payload arrived on the {modality} channel but its item says {item.modality!r}."
                    )
            items.extend(payload.items)
            seen += payload.output_length
        if seen != rows:
            raise ValueError(
                f"{modality} route payloads describe {seen} tokens but the encoder sent {rows} embedding rows."
            )
        if grid_thw is not None:
            grids = [tuple((int(x) for x in row)) for row in grid_thw.tolist()]
            claimed = [item.grid for item in items]
            if grids != claimed:
                raise ValueError(
                    f"{modality} grids {grids} do not match the route payload's {claimed}; the encoder's media order and the payload's disagree."
                )
        return items

    def _build_routed_input_ids(
        self, prompt, image_items, video_items, video_timestamps
    ) -> Tuple[List[int], List[Tuple[int, int]], List[Modality], List[VisualRun]]:
        """`QwenVLImageProcessor.build_input_ids_with_timestamps` with routed lengths."""
        if not isinstance(prompt, list):
            prompt = self._processor.tokenizer.encode(prompt)
        img_token_id = self.IM_TOKEN_ID
        video_token_id = self.VIDEO_TOKEN_ID
        vision_start_token_id = self.vision_start_token_id
        vision_end_token_id = self.vision_end_token_id
        input_ids: List[int] = []
        offsets: List[Tuple[int, int]] = []
        modality_list: List[Modality] = []
        runs: List[VisualRun] = []
        cur_idx = 0
        vision_start_indices = []
        for i in range(len(prompt) - 1):
            if prompt[i + 1] == img_token_id:
                vision_start_indices.append((i, Modality.IMAGE))
            elif prompt[i + 1] == video_token_id:
                vision_start_indices.append((i, Modality.VIDEO))
        img_idx = 0
        video_idx = 0
        for mm_start_idx, modality in vision_start_indices:
            modality_list.append(modality)
            video_tokens = None
            if modality == Modality.IMAGE:
                item = _nth(image_items, img_idx, "image")
                mm_token_num = item.output_length
                img_idx += 1
            else:
                item = _nth(video_items, video_idx, "video")
                if video_timestamps is None or video_idx >= len(video_timestamps):
                    raise ValueError(
                        f"VisionWeave needs per-frame timestamps for every video (the trained prompt format includes them), but none arrived for video {video_idx}."
                    )
                curr_timestamps = video_timestamps[video_idx]
                num_frames = item.grid[0]
                if len(curr_timestamps) < num_frames:
                    raise ValueError(
                        f"video {video_idx} has {num_frames} frames but only {len(curr_timestamps)} timestamps."
                    )
                video_tokens = []
                _current_offset = len(input_ids) + mm_start_idx + 1 - cur_idx
                frame_start = 0
                for frame_idx in range(num_frames):
                    if frame_idx > 0:
                        modality_list.append(Modality.VIDEO)
                    frame_seqlen = item.frame_lengths[frame_idx]
                    curr_time = curr_timestamps[frame_idx]
                    timestamp_text = f"<{curr_time:.1f} seconds>"
                    timestamp_tokens = self._processor.tokenizer.encode(
                        timestamp_text, add_special_tokens=False
                    )
                    video_tokens.extend(timestamp_tokens)
                    _current_offset += len(timestamp_tokens)
                    if vision_start_token_id is not None:
                        video_tokens.append(vision_start_token_id)
                        _current_offset += 1
                    video_tokens.extend([video_token_id] * frame_seqlen)
                    if vision_end_token_id is not None:
                        video_tokens.append(vision_end_token_id)
                    offsets.append((_current_offset, _current_offset + frame_seqlen - 1))
                    runs.append(
                        VisualRun(
                            start=_current_offset,
                            positions_x2=positions_for_frame(
                                item.positions_x2, frame_start, frame_seqlen
                            ),
                            native_span=_native_span(item.grid, self.fine),
                        )
                    )
                    frame_start += frame_seqlen
                    _current_offset += (
                        frame_seqlen + 1 if vision_end_token_id is not None else frame_seqlen
                    )
                mm_token_num = len(video_tokens)
                video_idx += 1
            assert cur_idx <= mm_start_idx
            input_ids.extend(prompt[cur_idx : mm_start_idx + 1])
            if modality == Modality.VIDEO:
                input_ids.extend(video_tokens)
            else:
                mm_offset_start = len(input_ids)
                input_ids.extend([img_token_id] * mm_token_num)
                offsets.append((mm_offset_start, len(input_ids) - 1))
                runs.append(
                    VisualRun(
                        start=mm_offset_start,
                        positions_x2=positions_for_frame(item.positions_x2, 0, mm_token_num),
                        native_span=_native_span(item.grid, self.fine),
                    )
                )
            cur_idx = mm_start_idx + 2
        else:
            input_ids.extend(prompt[cur_idx:])
        if img_idx != len(image_items) or video_idx != len(video_items):
            raise ValueError(
                f"the prompt has {img_idx} image and {video_idx} video placeholders but the route payload describes {len(image_items)} and {len(video_items)}."
            )
        return (input_ids, offsets, modality_list, runs)

    async def process_mm_data_async(self, *args, **kwargs):
        """Refused: VisionWeave's placeholder count only exists after the encoder has routed."""
        raise RuntimeError(
            f"{ARCHITECTURE} cannot process multimodal data locally: the routed token count is only known after the encoder has run. This server must be started with --language-only and --encoder-urls."
        )


_MODALITY_NAMES = {Modality.IMAGE: "image", Modality.VIDEO: "video"}
_MODALITY_BY_NAME = {name: modality for modality, name in _MODALITY_NAMES.items()}


def _group_routes_by_modality(routes) -> Dict[Modality, List[dict]]:
    """Turn the wire's FLAT list of per-part payloads into `{Modality: [payload, ...]}`."""
    if isinstance(routes, dict):
        raise ValueError(
            "the VisionWeave route payload arrived as a dict. This build expects the flat, part-ordered list that upstream's per-part metadata registry produces; a dict means something is still aggregating the payloads itself (an old install_receiver_metadata, most likely) and the two aggregations would disagree about order."
        )
    grouped: Dict[Modality, List[dict]] = {}
    for part_idx, raw in enumerate(routes):
        names = {item["modality"] for item in raw["items"]}
        if len(names) != 1:
            raise ValueError(
                f"route payload for part {part_idx} covers modalities {sorted(names)}; one part is one request's run of ONE modality (several items of it are normal), so this payload was sliced wrong on the encoder."
            )
        name = names.pop()
        modality = _MODALITY_BY_NAME.get(name)
        if modality is None:
            raise ValueError(
                f"route payload for part {part_idx} claims modality {name!r}, which VisionWeave does not serve (known: {sorted(_MODALITY_BY_NAME)})."
            )
        grouped.setdefault(modality, []).append(raw)
    return grouped


def _check_route_arity(grouped: Dict[Modality, List[dict]], img_grid_thw, video_grid_thw) -> None:
    """Every grid the encoder sent must be described by exactly one route ITEM."""
    n_items = sum((len(raw["items"]) for payloads in grouped.values() for raw in payloads))
    n_grids = sum((0 if g is None else len(g) for g in (img_grid_thw, video_grid_thw)))
    if n_items != n_grids:
        n_payloads = sum((len(payloads) for payloads in grouped.values()))
        raise ValueError(
            f"the encoder sent {n_grids} grid(s) but the route payloads describe {n_items} media item(s) ({n_payloads} payload(s); one payload is one part, and one part is one request's run of one modality, so a payload may carry several items). Upstream's aggregation drops parts whose metadata is None (`receiver.py:707-715`), so a part that reached the wire without its route disappears here rather than raising -- which is exactly the silent grid-length fallback this package exists to prevent."
        )


def _native_span(grid, fine: int) -> int:
    """`max(h, w) // fine` -- the span the sequence advances by, whatever the router decided."""
    return max(int(grid[1]), int(grid[2])) // fine


def _nth(items: List[RouteItem], index: int, label: str) -> RouteItem:
    if index >= len(items):
        raise ValueError(
            f"the prompt has more {label} placeholders than the route payload describes ({len(items)})."
        )
    return items[index]
