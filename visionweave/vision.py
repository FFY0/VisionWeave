# Portions derived from SGLang.
# Copyright 2025 Qwen Team
# Copyright 2025 SGLang Team
# Licensed under Apache-2.0; see LICENSES/Apache-2.0.txt.
# Modified for the VisionWeave routed serving integration.
"""Vision tower with sampled cross-attention routing and hard token selection."""

import inspect
import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sglang.srt.environ import envs
from sglang.srt.layers.attention.vision import prepare_vision_attention_metadata
from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel, _is_npu
from sglang.srt.server_args import get_global_server_args

from visionweave.compressor import VisionWeaveCompressionProjector, compression_block_index

from . import route_threshold, validate_geometry
from .core_patches import publish_route
from .route import RouteItem, RoutePayload, block_count, hard_route_layout, native_token_count
from .router import VisionWeaveHierCrossAttnRouter, build_mask_net, router_probabilities

logger = logging.getLogger(__name__)
__all__ = ["Qwen3_5VisionWeaveVisionModel"]
_UPSTREAM_FORWARD_MARKERS = (
    "if self._use_vectorized_pos_embed(len(grid_thw_list)):",
    "pos_embeds = self.fast_pos_embed_interpolate_from_list(grid_thw_list)",
    "rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw_list)",
    'if get_mm().mm_attention_backend == "flashinfer_cudnn":',
    "forward_metadata = prepare_vision_attention_metadata(",
    "forward_metadata=forward_metadata,",
    "x = x.unsqueeze(1)",
    "for layer_num, blk in enumerate(self.blocks):",
    "if layer_num in self.deepstack_visual_indexes:",
    "x = self.merger(x)",
)
_INDEX_CACHE_MAX = 512


def _assert_upstream_forward_unchanged() -> None:
    src = inspect.getsource(Qwen3VLMoeVisionModel.forward)
    missing = [m for m in _UPSTREAM_FORWARD_MARKERS if m not in src]
    if missing:
        raise RuntimeError(
            f"visionweave.vision copied Qwen3VLMoeVisionModel.forward from sglang df2f34cca, but the installed sglang's version no longer contains: {missing!r}. Re-derive Qwen3_5VisionWeaveVisionModel.forward from the new upstream body before serving -- a stale copy would silently drop whatever changed."
        )


_assert_upstream_forward_unchanged()


def raster_permutation(t: int, h: int, w: int, fine: int, device) -> torch.Tensor:
    """Index that un-interleaves the patchify order into a plain `(t, h, w)` raster."""
    m = fine
    row = torch.arange(h, device=device)
    col = torch.arange(w, device=device)
    frame = (row // m * (w // m * m * m) + row % m * m).view(h, 1) + (
        col // m * (m * m) + col % m
    ).view(1, w)
    frames = (torch.arange(t, device=device) * (h * w)).view(t, 1, 1)
    return (frame.view(1, h, w) + frames).reshape(-1)


class Qwen3_5VisionWeaveVisionModel(Qwen3VLMoeVisionModel):
    """ViT whose output is the hard-routed mixture of native and compressed tokens."""

    def __init__(self, vision_config, *args, **kwargs):
        self.fine, self.pool, self.coarse = validate_geometry(vision_config)
        self.threshold = route_threshold()
        if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
            raise ValueError(
                "SGLANG_VIT_ENABLE_CUDA_GRAPH is incompatible with the VisionWeave router: the captured graph replays the base vision forward, which neither routes nor compresses. Unset it (python -m visionweave.serve does)."
            )
        if get_global_server_args().mm_attention_backend == "flashinfer_cudnn":
            raise ValueError(
                "--mm-attention-backend flashinfer_cudnn is untested with the VisionWeave router (it pads the vision batch and sequence dimensions, which would change what the router's global layers attend to). Use the default backend."
            )
        super().__init__(vision_config, *args, **kwargs)
        if self.deepstack_visual_indexes:
            raise ValueError(
                f"VisionWeave does not support deepstack; vision_config.deepstack_visual_indexes is {self.deepstack_visual_indexes!r}, expected []."
            )
        sample_depths = [int(d) for d in vision_config.router_sample_depths]
        self.router_layer_depths: List[int] = [d for s in sample_depths for d in (s, s)]
        self.router_sample_depths = sample_depths
        self.router_cross_attn = VisionWeaveHierCrossAttnRouter(
            hidden_size=vision_config.hidden_size,
            depth=int(vision_config.router_cross_attn_depth),
            num_heads=int(
                getattr(vision_config, "router_cross_attn_num_heads", vision_config.num_heads)
            ),
            mlp_ratio=int(getattr(vision_config, "router_cross_attn_mlp_ratio", 2)),
            use_kv_norm=True,
        )
        self.mask_net = build_mask_net(vision_config.hidden_size).float()
        self.last_layer_bias = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.register_buffer("router_bias", torch.zeros(1, dtype=torch.float32), persistent=True)
        self.compression_projector = VisionWeaveCompressionProjector(
            hidden_size=self.hidden_size, fine=self.fine, pool=self.pool
        )
        self._block_index_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}
        self._raster_index_cache: Dict[Tuple[int, int, int], torch.Tensor] = {}
        self._modality: Optional[str] = None
        self.fingerprint: Optional[str] = None
        logger.info(
            "[visionweave] router ready: fine=%d pool=%d coarse=%d sample_depths=%s threshold=%.6g",
            self.fine,
            self.pool,
            self.coarse,
            sample_depths,
            self.threshold,
        )

    @contextmanager
    def modality_ctx(self, modality: str):
        previous = self._modality
        self._modality = modality
        try:
            yield
        finally:
            self._modality = previous

    def _gather_index(self, cache, builder, t: int, h: int, w: int) -> torch.Tensor:
        key = (t, h, w)
        cached = cache.get(key)
        if cached is not None:
            return cached
        if h % self.coarse or w % self.coarse:
            raise ValueError(
                f"VisionWeave routing needs a grid divisible by coarse={self.coarse} on both spatial axes, got h={h} w={w}. The image processor should have resized on factor patch_size*coarse=64; check that patches.install_resize_factor_patch ran in the encoder's TokenizerManager process."
            )
        if len(cache) >= _INDEX_CACHE_MAX:
            cache.clear()
        index = builder(t, h, w)
        cache[key] = index
        return index

    def _to_raster(self, x: torch.Tensor, grid_thw_list) -> torch.Tensor:
        """`(tokens, hidden)` patchify order -> the same tokens in `(t, h, w)` raster order."""
        parts = []
        offset = 0
        for t, h, w in grid_thw_list:
            t, h, w = (int(t), int(h), int(w))
            index = self._gather_index(
                self._raster_index_cache,
                lambda t, h, w: raster_permutation(t, h, w, self.fine, self.device),
                t,
                h,
                w,
            )
            parts.append(index + offset)
            offset += t * h * w
        index = parts[0] if len(parts) == 1 else torch.cat(parts)
        return x.index_select(0, index)

    def _compress(self, x: torch.Tensor, grid_thw_list) -> torch.Tensor:
        """`(tokens, hidden)` native -> `(tokens // pool**2, hidden)` in merger order."""
        spatial = self.pool * self.pool
        expected = 0
        gather = []
        for t, h, w in grid_thw_list:
            index = self._gather_index(
                self._block_index_cache,
                lambda t, h, w: compression_block_index(t, h, w, self.fine, self.pool, self.device),
                int(t),
                int(h),
                int(w),
            )
            gather.append(index + expected)
            expected += int(t) * int(h) * int(w)
        if x.shape[0] != expected:
            raise ValueError(
                f"vision tower produced {x.shape[0]} tokens but grid_thw {grid_thw_list} accounts for {expected}"
            )
        index = gather[0] if len(gather) == 1 else torch.cat(gather)
        blocks = x.index_select(0, index).view(-1, spatial, x.shape[-1])
        return self.compression_projector(blocks)

    def _hard_select(
        self,
        native_values: torch.Tensor,
        compressed_values: torch.Tensor,
        probabilities: torch.Tensor,
        grid_thw_list,
    ) -> torch.Tensor:
        """Interleave the two token streams per the router's decision, and publish the route."""
        if self._modality is None:
            raise RuntimeError(
                "the VisionWeave vision tower was called without a modality; get_image_feature / get_video_feature must wrap the call in modality_ctx."
            )
        if self.fingerprint is None:
            raise RuntimeError(
                "the VisionWeave vision tower has no checkpoint fingerprint; the model class must set it before serving, or the language server cannot detect a checkpoint skew."
            )
        compressed_mask = (probabilities >= self.threshold).to("cpu").numpy()
        total_native = sum((native_token_count(g) for g in grid_thw_list))
        if native_values.shape[0] != total_native:
            raise RuntimeError(
                f"merger produced {native_values.shape[0]} native tokens, grid says {total_native}"
            )
        total_blocks = sum((block_count(g) for g in grid_thw_list))
        if compressed_values.shape[0] != total_blocks or probabilities.shape[0] != total_blocks:
            raise RuntimeError(
                f"{compressed_values.shape[0]} compressed tokens and {probabilities.shape[0]} router decisions for {total_blocks} blocks"
            )
        items = []
        index_parts = []
        native_offset = 0
        block_offset = 0
        for grid in grid_thw_list:
            grid = tuple((int(x) for x in grid))
            blocks = block_count(grid)
            mask = compressed_mask[block_offset : block_offset + blocks]
            layout = hard_route_layout(grid, mask)
            items.append(RouteItem.from_layout(self._modality, grid, mask, layout))
            index = layout.gather_index.astype(np.int64, copy=True)
            comp = layout.is_compressed
            index[comp] += total_native + block_offset
            index[~comp] += native_offset
            index_parts.append(index)
            native_offset += native_token_count(grid)
            block_offset += blocks
        payload = RoutePayload(
            fingerprint=self.fingerprint, threshold=self.threshold, items=tuple(items)
        )
        publish_route(payload.to_dict())
        flat = np.concatenate(index_parts) if len(index_parts) > 1 else index_parts[0]
        index = torch.from_numpy(flat).to(native_values.device)
        combined = torch.cat((native_values, compressed_values))
        routed = combined.index_select(0, index)
        return routed

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
            raise RuntimeError(
                "SGLANG_VIT_ENABLE_CUDA_GRAPH became set after model construction; the ViT graph path skips the VisionWeave router."
            )
        x = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
        x = self.patch_embed(x)
        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
            grid_thw = np.array(grid_thw, dtype=np.int32)
        else:
            grid_thw_list = grid_thw.tolist()
            grid_thw = grid_thw.cpu().numpy()
        if self._use_vectorized_pos_embed(len(grid_thw_list)):
            pos_embeds = self.fast_pos_embed_interpolate_vectorized(grid_thw_list)
        else:
            pos_embeds = self.fast_pos_embed_interpolate_from_list(grid_thw_list)
        x += pos_embeds
        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw_list)
        token_cu_seqlens = np.repeat(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            axis=0, dtype=np.int32
        )
        token_cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), token_cu_seqlens])
        packed_indptrs = None
        flashinfer_sequence_lengths = None
        flashinfer_max_seqlen = 0
        cu_seqlens = torch.from_numpy(token_cu_seqlens)
        if not _is_npu:
            cu_seqlens = cu_seqlens.to(self.device, non_blocking=True)
        else:
            cu_seqlens = cu_seqlens.to("cpu")
        forward_metadata = prepare_vision_attention_metadata(
            cu_seqlens,
            device=self.device,
            packed_indptrs=packed_indptrs,
            sequence_lengths=flashinfer_sequence_lengths,
            flashinfer_max_seqlen=flashinfer_max_seqlen,
        )
        x = x.unsqueeze(1)
        query, router_context = self.router_cross_attn.init_state(
            grid_thw_list, self.device, self.dtype
        )
        layer_index = 0
        while (
            layer_index < len(self.router_layer_depths)
            and self.router_layer_depths[layer_index] == 0
        ):
            query, layer_index = self._run_tap(
                query, router_context, x.squeeze(1), grid_thw_list, layer_index
            )
        for block_depth, blk in enumerate(self.blocks, start=1):
            x = blk(
                x,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                forward_metadata=forward_metadata,
            )
            while (
                layer_index < len(self.router_layer_depths)
                and self.router_layer_depths[layer_index] == block_depth
            ):
                query, layer_index = self._run_tap(
                    query, router_context, x.squeeze(1), grid_thw_list, layer_index
                )
        if layer_index != len(self.router_layer_depths):
            raise RuntimeError(
                f"only {layer_index} of {len(self.router_layer_depths)} router layers ran; router_sample_depths={self.router_sample_depths} does not fit {len(self.blocks)} ViT blocks."
            )
        x = x.squeeze(1)
        native_values = self.merger(x)
        probabilities = router_probabilities(
            self.mask_net, self.last_layer_bias, self.router_bias, query
        )
        compressed_values = self.merger(self._compress(x, grid_thw_list))
        return self._hard_select(native_values, compressed_values, probabilities, grid_thw_list)

    def _run_tap(self, query, context, states, grid_thw_list, layer_index: int):
        """Run the Local+Global pair that shares one ViT snapshot."""
        raster = self._to_raster(states, grid_thw_list)
        query = self.router_cross_attn.run_layer(query, layer_index, raster, grid_thw_list, context)
        query = self.router_cross_attn.run_layer(
            query, layer_index + 1, raster, grid_thw_list, context
        )
        return (query, layer_index + 2)
