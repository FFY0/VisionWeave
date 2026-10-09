# Portions derived from SGLang.
# Copyright 2025 Qwen Team
# Copyright 2025 SGLang Team
# Licensed under Apache-2.0; see LICENSES/Apache-2.0.txt.
# Modified for the VisionWeave routed serving integration.
"""Scaled M-RoPE for prefill, decode, retraction and speculative positions."""

import inspect
import logging

import torch

from . import POSITION_SCALE

logger = logging.getLogger(__name__)
_MARK = "_visionweave_position"
_installed = set()


def _require(source_owner, *markers) -> None:
    """Fail loudly if upstream no longer contains the lines a replacement was derived from."""
    src = inspect.getsource(source_owner)
    missing = [m for m in markers if m not in src]
    if missing:
        raise RuntimeError(
            f"visionweave expected these lines in {source_owner.__qualname__}, and this sglang does not have them: {missing}. The replacement in patches.py was derived from sglang main df2f34cca and must be re-derived before serving."
        )


def _first_time(name: str) -> bool:
    if name in _installed:
        return False
    _installed.add(name)
    return True


def install_position_scale_rope_patch() -> None:
    """Make integer position `PS*p` produce the angle fractional position `p` would have."""
    if not _first_time("position_scale_rope"):
        return
    if POSITION_SCALE == 1:
        return
    from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding

    if getattr(MRotaryEmbedding.__init__, _MARK, False):
        return
    _require(
        MRotaryEmbedding.__init__,
        "max_position_embeddings: int,",
        "mrope_section: Optional[List[int]] = None,",
        "            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype",
    )
    _require(MRotaryEmbedding._compute_inv_freq, "def _compute_inv_freq(self, base", "1.0 / (")
    original_init = MRotaryEmbedding.__init__
    original_inv_freq = MRotaryEmbedding._compute_inv_freq

    def visionweave_init(
        self,
        head_size,
        rotary_dim,
        max_position_embeddings,
        base,
        is_neox_style,
        dtype,
        *args,
        **kwargs,
    ):
        self._visionweave_position_scale = POSITION_SCALE
        original_init(
            self,
            head_size,
            rotary_dim,
            max_position_embeddings * POSITION_SCALE,
            base,
            is_neox_style,
            dtype,
            *args,
            **kwargs,
        )
        logger.info(
            "[visionweave] MRotaryEmbedding scaled: inv_freq/%d, max_position_embeddings %d -> %d, cache rows %d",
            POSITION_SCALE,
            max_position_embeddings,
            self.max_position_embeddings,
            int(self.cos_sin_cache.shape[0]),
        )

    def visionweave_compute_inv_freq(self, base):
        inv_freq = original_inv_freq(self, base)
        scale = getattr(self, "_visionweave_position_scale", 1)
        return inv_freq / scale if scale != 1 else inv_freq

    setattr(visionweave_init, _MARK, True)
    setattr(visionweave_compute_inv_freq, _MARK, True)
    MRotaryEmbedding.__init__ = visionweave_init
    MRotaryEmbedding._compute_inv_freq = visionweave_compute_inv_freq


def install_mrope_arithmetic_patches() -> None:
    """Replace the five position-continuation sites with PS-stepping versions."""
    if not _first_time("mrope_arithmetic"):
        return
    if POSITION_SCALE == 1:
        return
    from sglang.srt.managers import mm_utils
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.runtime_context import get_exec

    ps = POSITION_SCALE
    if not getattr(ForwardBatch._expand_mrope_from_input, _MARK, False):
        _require(
            ForwardBatch._expand_mrope_from_input,
            "mm_input.mrope_positions is not None",
            "pos = mm_input.mrope_positions[:, seq_len - 1 : seq_len]",
            "if mm_input.mrope_position_delta_repeated_cache is None:",
            "(mm_input.mrope_position_delta - 1).flatten().unsqueeze(0).repeat(3, 1)",
            "mrope_positions = mm_input.mrope_position_delta_repeated_cache + seq_len",
        )

        def _expand_mrope_from_input(self, mm_input, seq_len):
            if (
                mm_input.mrope_positions is not None
                and mm_input.mrope_positions.shape[1] >= seq_len
            ):
                return mm_input.mrope_positions[:, seq_len - 1 : seq_len]
            if mm_input.mrope_position_delta_repeated_cache is None:
                mm_input.mrope_position_delta_repeated_cache = (
                    (mm_input.mrope_position_delta - ps).flatten().unsqueeze(0).repeat(3, 1)
                )
            return mm_input.mrope_position_delta_repeated_cache + ps * seq_len

        setattr(_expand_mrope_from_input, _MARK, True)
        ForwardBatch._expand_mrope_from_input = _expand_mrope_from_input
    if not getattr(ForwardBatch.compute_spec_mrope_positions, _MARK, False):
        _require(
            ForwardBatch.compute_spec_mrope_positions,
            "def compute_spec_mrope_positions(",
            "if seq_positions is None:",
            "seq_positions = batch.spec_info.positions",
            "seq_positions = seq_positions.view(batch_size, -1)",
            "(seq_positions + mrope_delta_tensor).flatten().unsqueeze(0).repeat(3, 1)",
        )

        def compute_spec_mrope_positions(self, model_runner, batch, seq_positions=None):
            batch_size = self.seq_lens.shape[0]
            device = model_runner.device
            mm_inputs = batch.multimodal_inputs
            if seq_positions is None:
                seq_positions = batch.spec_info.positions
            seq_positions = seq_positions.view(batch_size, -1)
            if all((mm_input is None for mm_input in mm_inputs)):
                mrope_delta_tensor = torch.zeros((batch_size, 1), dtype=torch.int64, device=device)
            else:
                mrope_deltas = [
                    torch.zeros(1, dtype=torch.int64)
                    if mm_inputs[i] is None
                    else mm_inputs[i].mrope_position_delta.squeeze(0)
                    for i in range(batch_size)
                ]
                mrope_delta_tensor = torch.stack(mrope_deltas, dim=0).to(device=device)
            next_input_positions = (
                (seq_positions * ps + mrope_delta_tensor).flatten().unsqueeze(0).repeat(3, 1)
            )
            self.mrope_positions = next_input_positions

        setattr(compute_spec_mrope_positions, _MARK, True)
        ForwardBatch.compute_spec_mrope_positions = compute_spec_mrope_positions
    _require(
        ForwardBatch._compute_mrope_positions,
        "self._compute_mrope_positions_decode(model_runner, batch)",
        "self._compute_mrope_positions_extend(model_runner, batch)",
    )
    if not getattr(ForwardBatch._compute_mrope_positions_decode, _MARK, False):
        _require(
            ForwardBatch._compute_mrope_positions_decode,
            "rl_on_policy_target = get_exec().deterministic.rl_on_policy_target",
            "has_precomputed_mrope = any(",
            "positions_1d = seq_lens_int64 - 1",
            "positions_1d = (deltas - 1) + seq_lens_int64",
            "seq_lens_cpu[batch_idx] - 1,",
        )

        def _compute_mrope_positions_decode(self, model_runner, batch):
            seq_lens_cpu = self.seq_lens_cpu
            batch_size = seq_lens_cpu.shape[0]
            mm_inputs = batch.multimodal_inputs
            rl_on_policy_target = get_exec().deterministic.rl_on_policy_target
            seq_lens_int64 = self.seq_lens.to(torch.int64)
            has_precomputed_mrope = any(
                (
                    mm is not None
                    and mm.mrope_positions is not None
                    and (mm.mrope_positions.shape[1] >= seq_lens_cpu[i])
                    for i, mm in enumerate(mm_inputs)
                )
            )
            has_multimodal_input = any((mm is not None for mm in mm_inputs))
            if rl_on_policy_target is not None or not has_multimodal_input:
                positions_1d = (seq_lens_int64 - 1) * ps
                self.mrope_positions = positions_1d.unsqueeze(0).repeat(3, 1)
                return
            if not has_precomputed_mrope:
                deltas_list = [0] * batch_size
                for i, mm in enumerate(mm_inputs):
                    deltas_list[i] = mm.mrope_position_delta.item() if mm is not None else 0
                deltas = torch.tensor(deltas_list, dtype=torch.int64, device=model_runner.device)
                positions_1d = deltas - ps + ps * seq_lens_int64
                self.mrope_positions = positions_1d.unsqueeze(0).repeat(3, 1)
                return
            mrope_positions_list = [None] * batch_size
            for batch_idx in range(batch_size):
                mm_input = mm_inputs[batch_idx]
                if mm_input is None:
                    mrope_positions = torch.full(
                        (3, 1), ps * (int(seq_lens_cpu[batch_idx]) - 1), dtype=torch.int64
                    )
                else:
                    mrope_positions = self._expand_mrope_from_input(
                        mm_input, seq_lens_cpu[batch_idx]
                    )
                mrope_positions_list[batch_idx] = mrope_positions
            self.mrope_positions = torch.cat(mrope_positions_list, dim=1).to(
                dtype=torch.int64, device=model_runner.device, non_blocking=True
            )

        setattr(_compute_mrope_positions_decode, _MARK, True)
        ForwardBatch._compute_mrope_positions_decode = _compute_mrope_positions_decode
    if not getattr(ForwardBatch._compute_mrope_positions_extend, _MARK, False):
        _require(
            ForwardBatch._compute_mrope_positions_extend,
            "extend_lens = batch.extend_lens",
            "prefix_lens = batch.prefix_lens",
            "extend_prefix_len + extend_seq_len,",
            "mrope_positions = mm_input.mrope_positions[",
            "if mrope_positions.numel() == 0:",
        )

        def _compute_mrope_positions_extend(self, model_runner, batch):
            seq_lens_cpu = self.seq_lens_cpu
            batch_size = seq_lens_cpu.shape[0]
            mm_inputs = batch.multimodal_inputs
            rl_on_policy_target = get_exec().deterministic.rl_on_policy_target
            extend_lens = batch.extend_lens
            prefix_lens = batch.prefix_lens
            mrope_positions_list = [None] * batch_size
            for batch_idx in range(batch_size):
                mm_input = mm_inputs[batch_idx]
                extend_seq_len = extend_lens[batch_idx]
                extend_prefix_len = prefix_lens[batch_idx]
                if mm_input is None or rl_on_policy_target is not None:
                    mrope_positions = (
                        torch.arange(
                            extend_prefix_len, extend_prefix_len + extend_seq_len, dtype=torch.int64
                        )
                        .mul_(ps)
                        .unsqueeze(0)
                        .repeat(3, 1)
                    )
                else:
                    mrope_positions = mm_input.mrope_positions[
                        :, extend_prefix_len : extend_prefix_len + extend_seq_len
                    ]
                    if mrope_positions.numel() == 0:
                        mrope_positions = self._expand_mrope_from_input(
                            mm_input, seq_lens_cpu[batch_idx]
                        )
                mrope_positions_list[batch_idx] = mrope_positions
            self.mrope_positions = torch.cat(mrope_positions_list, dim=1).to(
                dtype=torch.int64, device=model_runner.device, non_blocking=True
            )

        setattr(_compute_mrope_positions_extend, _MARK, True)
        ForwardBatch._compute_mrope_positions_extend = _compute_mrope_positions_extend
    if not getattr(mm_utils.extend_mrope_positions_for_retracted_request, _MARK, False):
        _require(
            mm_utils.extend_mrope_positions_for_retracted_request,
            "if output_ids_len <= 0:",
            "last_position = mrope_positions[:, -1]",
            "start_pos = last_position[0] + 1",
            "start_pos + output_ids_len,",
        )

        def extend_mrope_positions_for_retracted_request(mrope_positions, output_ids_len):
            if output_ids_len <= 0:
                return mrope_positions
            last_position = mrope_positions[:, -1]
            start_pos = last_position[0] + ps
            output_positions = (
                torch.arange(
                    start_pos,
                    start_pos + ps * output_ids_len,
                    ps,
                    dtype=torch.int64,
                    device=mrope_positions.device,
                )
                .unsqueeze(0)
                .expand(3, -1)
            )
            return torch.cat([mrope_positions, output_positions], dim=1)

        extend_mrope_positions_for_retracted_request.__doc__ = (
            "visionweave: as upstream, but steps by POSITION_SCALE per generated token."
        )
        setattr(extend_mrope_positions_for_retracted_request, _MARK, True)
        mm_utils.extend_mrope_positions_for_retracted_request = (
            extend_mrope_positions_for_retracted_request
        )
    logger.info("[visionweave] mrope continuation arithmetic steps by %d", ps)


def install_mrope_fastpath_guard() -> None:
    """Disable native position shortcuts for routed multimodal requests."""
    if not _first_time("mrope_fastpath"):
        return
    from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

    shortcut = QwenVLImageProcessor._compute_image_only_mrope_positions_from_offsets
    if getattr(shortcut, _MARK, False):
        return
    _require(
        shortcut,
        '"qwen3_5",',
        "spatial_merge_size = self._spatial_merge_size",
        "llm_grid_h = h // spatial_merge_size",
        "torch.arange(text_len, dtype=dtype, device=device)",
        "if num_image_tokens != end - start + 1:",
    )
    _require(
        QwenVLImageProcessor._get_precomputed_mrope_from_output,
        'self._get_processor_output_value(ret, "mrope_positions")',
        "if mrope_positions is None or mrope_position_delta is None:",
    )
    _require(
        QwenVLImageProcessor.process_mm_data_async,
        "mrope_result = self._get_precomputed_mrope_from_output(ret)",
        "mrope_result = self._compute_image_only_mrope_positions_from_offsets(",
        "if mrope_result is None:",
        "mrope_result = MRotaryEmbedding.get_rope_index(",
    )

    def _decline_image_only_fastpath(self, *_args, **_kwargs):
        """visionweave: always decline -- see patches.install_mrope_fastpath_guard (A)."""
        return None

    def _decline_precomputed_mrope(self, *_args, **_kwargs):
        """visionweave: always decline -- see patches.install_mrope_fastpath_guard (B)."""
        return None

    setattr(_decline_image_only_fastpath, _MARK, True)
    setattr(_decline_precomputed_mrope, _MARK, True)
    QwenVLImageProcessor._compute_image_only_mrope_positions_from_offsets = (
        _decline_image_only_fastpath
    )
    QwenVLImageProcessor._get_precomputed_mrope_from_output = _decline_precomputed_mrope
    logger.info(
        "[visionweave] both mrope shortcuts disabled (image-only fast path, precomputed processor output); all multimodal positions come from route payloads"
    )
