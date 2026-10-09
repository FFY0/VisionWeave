# Portions derived from SGLang.
# Copyright 2025 Qwen Team
# Copyright 2025 SGLang Team
# Licensed under Apache-2.0; see LICENSES/Apache-2.0.txt.
# Modified for the VisionWeave routed serving integration.
"""Transport routed metadata between SGLang encoder and tokenizer processes."""

import ast
import inspect
import logging
import os
import textwrap
from contextvars import ContextVar
from typing import Optional

from . import ARCHITECTURE

logger = logging.getLogger(__name__)
__all__ = [
    "installed_targets",
    "install_arch_allowlist",
    "install_encoder_admission",
    "install_encoder_decode_offload",
    "install_encoder_embedding_dtype",
    "install_encoder_error_cleanup",
    "install_encoder_metadata",
    "install_encoder_video_factor",
    "install_encoder_video_timestamps",
    "install_receiver_fail_closed",
    "install_receiver_inflight_gate",
    "install_receiver_metadata",
    "publish_route",
    "ROUTE_META_KEY",
    "ROUTE_KWARG",
]
ROUTE_META_KEY = "visionweave_route"
ROUTE_KWARG = ROUTE_META_KEY
_MARK = "_visionweave"
_installed: set = set()
_ROUTE_SLOT: ContextVar[Optional[dict]] = ContextVar("visionweave_route_slot", default=None)


def publish_route(payload: dict) -> None:
    """Called by the vision tower once per encode, with the serialized `RoutePayload`."""
    _ROUTE_SLOT.set(payload)


def _take_route() -> Optional[dict]:
    payload = _ROUTE_SLOT.get()
    _ROUTE_SLOT.set(None)
    return payload


_ROUTE_BATCH: ContextVar[Optional[tuple]] = ContextVar("visionweave_route_batch", default=None)


def _take_route_batch() -> Optional[tuple]:
    batch = _ROUTE_BATCH.get()
    _ROUTE_BATCH.set(None)
    return batch


def _require(owner, *markers) -> None:
    src = inspect.getsource(owner)
    missing = [m for m in markers if m not in src]
    if missing:
        raise RuntimeError(
            f"visionweave.core_patches expected these lines in {owner.__qualname__}, and this sglang does not have them: {missing}. The patch was derived from df2f34cca and must be re-derived before serving."
        )


def _require_absent(owner, *markers) -> None:
    """The reverse of `_require`: refuse to start if a line we RETIRED a patch over came back."""
    src = inspect.getsource(owner)
    present = [m for m in markers if m in src]
    if present:
        raise RuntimeError(
            f"visionweave.core_patches retired a patch on {owner.__qualname__} because this sglang no longer contained {present}, and it does again. Re-read that function and either restore the patch or update this gate -- do not just delete it."
        )


def _require_trailing_raise(owner, needle: str) -> None:
    """Assert `owner`'s LAST statement is an `if ...: raise` whose message contains `needle`."""
    body = ast.parse(textwrap.dedent(inspect.getsource(owner))).body[0].body
    last = body[-1]
    ok = (
        isinstance(last, ast.If)
        and len(last.body) == 1
        and isinstance(last.body[0], ast.Raise)
        and (needle in ast.unparse(last.body[0]))
    )
    if not ok:
        raise RuntimeError(
            f"visionweave.core_patches needs {owner.__qualname__} to END with the `raise` carrying {needle!r}, because it widens that gate by catching the exception and everything before the raise must therefore already have run. On this sglang the last statement is {type(last).__name__} at line {last.lineno} of the function. Re-derive install_arch_allowlist before serving."
        )


def _first_time(name: str) -> bool:
    if name in _installed:
        return False
    _installed.add(name)
    return True


def install_arch_allowlist() -> None:
    """Let the VisionWeave architecture through the encoder-disaggregation model-type gate."""
    if not _first_time("arch_allowlist"):
        return
    from sglang.srt.arg_groups import pd_disaggregation_hook as hook
    from sglang.srt.arg_groups import pipeline

    if getattr(hook.handle_encoder_disaggregation, _MARK, False):
        return
    _require(
        hook.handle_encoder_disaggregation,
        "def handle_encoder_disaggregation(server_args: Any):",
        "model_arch = hf_config.architectures[0]",
        "if (cfg.encoder_only or cfg.language_only) and model_arch not in [",
        'f"Model type {model_arch} is not supported for encoder disaggregation. "',
    )
    _require_trailing_raise(
        hook.handle_encoder_disaggregation, "is not supported for encoder disaggregation"
    )
    _require(
        pipeline.run_resolution_pipeline,
        "    from sglang.srt.arg_groups.pd_disaggregation_hook import (\n        handle_encoder_disaggregation,\n",
        "    handle_encoder_disaggregation(server_args)\n",
    )
    original = hook.handle_encoder_disaggregation
    unsupported = f"Model type {ARCHITECTURE} is not supported for encoder disaggregation"

    def visionweave_handle_encoder_disaggregation(server_args):
        try:
            return original(server_args)
        except ValueError as exc:
            if not str(exc).startswith(unsupported):
                raise
            logger.info(
                "[visionweave] %s allowed for encoder disaggregation by visionweave.core_patches",
                ARCHITECTURE,
            )

    setattr(visionweave_handle_encoder_disaggregation, _MARK, True)
    hook.handle_encoder_disaggregation = visionweave_handle_encoder_disaggregation


def install_encoder_metadata() -> None:
    """Make the encoder tell the truth about a routed request's length, and ship the route with it."""
    if not _first_time("encoder_metadata"):
        return
    from sglang.srt.disaggregation.encoder import server as encoder_server

    MMEncoder = encoder_server.MMEncoder
    if getattr(MMEncoder._stage_embedding_batch, _MARK, False):
        return
    _require(
        MMEncoder._validate_embedding_token_count,
        "expected_tokens = sum(ctx.preprocess_result.token_counts)",
        "if mm_embedding.shape[0] != expected_tokens:",
    )
    _require(
        MMEncoder._stage_embeddings,
        "                    ctx.preprocess_result.token_counts[item_offset:item_end]\n",
        "embedding = mm_embedding[token_offset : token_offset + num_tokens]",
        "req_aux_data = dict(ctx.aux_data)",
        "self._stage_embedding_batch(staged_embeddings)",
    )
    _require(
        MMEncoder._stage_embedding_batch,
        "states = [self._embedding_state_for_stage(mm_data) for mm_data in embeddings]",
        "state.embedding_data = mm_data",
        "state.embedding_ready.set()",
    )
    _require(
        MMEncoder._publish_preprocess_metadata,
        "token_count = sum(ctx.preprocess_result.token_counts[item_offset:item_end])",
        "embedding_shape=[token_count, embedding_dim],",
        "await meta_registry.publish(",
    )
    _require(
        MMEncoder._embedding_state_for_stage,
        "metadata = state.embedding_data",
        "and (metadata.shape != mm_data.shape or metadata.dtype != mm_data.dtype)",
        'f"Embedding metadata mismatch for {mm_data.req_id}: "',
    )
    _require(
        encoder_server.EncoderMetaRegistry,
        "Mooncake decoder ranks consume it early to allocate landing buffers. ZMQ\n    publishes the same state for a uniform pipeline but does not consume it\n    before encode/send completes.",
    )
    from sglang.srt.disaggregation.encoder import runtime as encoder_runtime

    _require(
        encoder_runtime.execute_encode_pipeline,
        "# Publish the actual result for every backend. ZMQ does not consume this",
        "req_id, nbytes, embedding_len, embedding_dim",
    )
    original_validate = MMEncoder._validate_embedding_token_count
    original_stage_batch = MMEncoder._stage_embedding_batch

    def _routed_lengths(payload: dict) -> list:
        return [int(item["positions_x2"].shape[0]) for item in payload["items"]]

    def visionweave_validate_embedding_token_count(ctx, mm_embedding):
        payload = _take_route()
        if payload is None:
            raise RuntimeError(
                f"the VisionWeave vision tower did not publish a route payload for this encode (modality={ctx.modality}, {ctx.num_items} item(s)). Every visual encode must publish one: without it the routed length is unknown and the encoder would fall back to grid geometry, which is the native length. If this fired on a request that hit an embedding cache, that is the cache -- VisionWeave embeddings are routed products and both caches are refused at startup by launch.py::_check_caches."
            )
        lengths = _routed_lengths(payload)
        if len(lengths) != ctx.num_items:
            raise RuntimeError(
                f"route payload describes {len(lengths)} media item(s) but this encode covered {ctx.num_items} (items_per_req={list(ctx.items_per_req)}); the tower must publish one item per grid, in flatten order."
            )
        rows = int(mm_embedding.shape[0])
        if sum(lengths) != rows:
            raise RuntimeError(
                f"route payload describes {sum(lengths)} routed tokens but the encoder produced {rows} embedding rows (modality={ctx.modality}, per-item {lengths})."
            )
        ctx.preprocess_result.token_counts[:] = lengths
        original_validate(ctx, mm_embedding)
        _ROUTE_BATCH.set((payload, tuple((int(n) for n in ctx.items_per_req))))

    def visionweave_stage_embedding_batch(self, embeddings):
        batch = _take_route_batch()
        if batch is None:
            raise RuntimeError(
                "no VisionWeave route payload was stashed for this batch, so the per-request split cannot be done. The stash is written by the patched MMEncoder._validate_embedding_token_count, which every path that produces an embedding goes through; reaching staging without it means this package is only half installed in the encoder process."
            )
        payload, items_per_req = batch
        if len(embeddings) != len(items_per_req):
            raise RuntimeError(
                f"staging {len(embeddings)} request(s) but the encode context said {len(items_per_req)}; the route payload cannot be split safely."
            )
        items = payload["items"]
        item_offset = 0
        for mm_data, num_items in zip(embeddings, items_per_req):
            item_end = item_offset + num_items
            per_req = dict(payload)
            per_req["items"] = items[item_offset:item_end]
            rows = sum((int(i["positions_x2"].shape[0]) for i in per_req["items"]))
            staged_rows = int(mm_data.shape[0]) if mm_data.shape else 0
            if rows != staged_rows:
                raise RuntimeError(
                    f"route slice for {mm_data.req_id} describes {rows} routed tokens but its staged embedding has {staged_rows} rows; the batch split disagrees with the payload order."
                )
            setattr(mm_data, ROUTE_META_KEY, [per_req])
            _reconcile_preforward_shape(self, mm_data)
            item_offset = item_end
        return original_stage_batch(self, embeddings)

    def _reconcile_preforward_shape(encoder, mm_data) -> None:
        """Replace the pre-forward row count with the routed embedding row count."""
        state = encoder.req_states.get(mm_data.req_id)
        meta = getattr(state, "embedding_data", None) if state is not None else None
        if meta is None or meta.embedding is not None or mm_data.embedding is None:
            return
        want = list(mm_data.shape or [])
        have = list(meta.shape or [])
        if len(have) != 2 or len(want) != 2 or have[1] != want[1]:
            raise RuntimeError(
                f"pre-forward embedding metadata for {mm_data.req_id} is {have} but the staged embedding is {want}. VisionWeave expects these to differ in the ROW count only (grid geometry vs the routed length); a different embedding dimension, or a shape that is not 2-D, is a real mismatch and this patch will not paper over it."
            )
        meta.shape = want

    setattr(visionweave_validate_embedding_token_count, _MARK, True)
    setattr(visionweave_stage_embedding_batch, _MARK, True)
    MMEncoder._validate_embedding_token_count = staticmethod(
        visionweave_validate_embedding_token_count
    )
    MMEncoder._stage_embedding_batch = visionweave_stage_embedding_batch
    logger.info(
        "[visionweave] encoder token counts come from the route payload; %r rides per request on EmbeddingData",
        ROUTE_META_KEY,
    )


def install_encoder_video_timestamps() -> None:
    """Check and retain per-frame timestamps for Qwen3.5 videos."""
    if not _first_time("encoder_video_timestamps"):
        return
    from sglang.srt.disaggregation.encoder.preprocessor import EncoderPreprocessor

    if getattr(EncoderPreprocessor.__init__, _MARK, False):
        return
    _require(
        EncoderPreprocessor._process_video_items,
        'processor_input["video_timestamps"] = video_timestamps',
        "timestamps = self._calculate_timestamps(",
        "frames_indices, video_fps, merge_size",
    )
    _require(EncoderPreprocessor.__init__, "self.model_type = getattr(", '"model_type", "unknown"')
    covered = _timestamp_gate_model_types(EncoderPreprocessor)
    original_init = EncoderPreprocessor.__init__

    def visionweave_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.model_type not in covered:
            raise RuntimeError(
                f"this encoder is serving model_type {self.model_type!r}, and upstream's video timestamp gate in EncoderPreprocessor._process_video_items covers {covered}. Qwen3.5 puts a `<N.N seconds>` marker between per-frame vision groups and the VisionWeave processor always builds that layout, so without `video_timestamps` in aux_data every video request is refused rather than degraded. Either the checkpoint's model_type is unexpected, or upstream narrowed that list -- in which case restore the qwen3_vl remap this gate replaced."
            )
        logger.info(
            "[visionweave] video timestamps: upstream's gate covers model_type=%r (%s)",
            self.model_type,
            ", ".join(covered),
        )

    setattr(visionweave_init, _MARK, True)
    EncoderPreprocessor.__init__ = visionweave_init
    logger.info(
        "[visionweave] encoder video-timestamp patch retired to a gate; upstream covers %s",
        ", ".join(covered),
    )


def _timestamp_gate_model_types(preprocessor_cls) -> list:
    """Read the model_type allowlist that guards `processor_input["video_timestamps"]`."""
    src = textwrap.dedent(inspect.getsource(preprocessor_cls._process_video_items))
    func = ast.parse(src).body[0]
    for node in ast.walk(func):
        if not isinstance(node, ast.If):
            continue
        assigns = [
            st
            for st in ast.walk(node)
            if isinstance(st, ast.Assign)
            and len(st.targets) == 1
            and isinstance(st.targets[0], ast.Subscript)
            and isinstance(st.targets[0].slice, ast.Constant)
            and (st.targets[0].slice.value == "video_timestamps")
        ]
        if not assigns:
            continue
        for cmp in ast.walk(node.test):
            if (
                isinstance(cmp, ast.Compare)
                and len(cmp.ops) == 1
                and isinstance(cmp.ops[0], ast.In)
                and isinstance(cmp.left, ast.Attribute)
                and (cmp.left.attr == "model_type")
                and isinstance(cmp.comparators[0], ast.List)
                and all(
                    (
                        isinstance(e, ast.Constant) and isinstance(e.value, str)
                        for e in cmp.comparators[0].elts
                    )
                )
            ):
                return [e.value for e in cmp.comparators[0].elts]
    raise RuntimeError(
        'could not find the `self.model_type in [...]` test guarding `processor_input["video_timestamps"] = ...` in EncoderPreprocessor._process_video_items. sglang restructured the video timestamp path, so the VisionWeave assumption that qwen3_5 videos get per-frame timestamps is no longer checked by anything. Re-derive install_encoder_video_timestamps before serving.'
    )


def install_encoder_video_factor() -> None:
    """Make sure the ENCODER's `preprocess_video` is the factor-64 one."""
    if not _first_time("encoder_video_factor"):
        return
    from sglang.srt.disaggregation.encoder import preprocessor
    from sglang.srt.multimodal.processors import qwen_vl

    if not getattr(qwen_vl.preprocess_video, "_visionweave_video", False):
        raise RuntimeError(
            "qwen_vl.preprocess_video is not the visionweave.video factor-64 version. Video frames would be resized on factor 28, and half of those grids are not divisible by coarse=4, which the VisionWeave router refuses. Check that visionweave.video imported successfully in this process."
        )
    preprocessor.preprocess_video = qwen_vl.preprocess_video
    logger.info(
        "[visionweave] encoder/preprocessor.preprocess_video rebound to the factor-64 version"
    )


def install_encoder_decode_offload() -> None:
    """Run the video decode in a thread pool instead of on the encoder's event loop."""
    if not _first_time("encoder_decode_offload"):
        return
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from sglang.srt.disaggregation.encoder import preprocessor

    _require(
        preprocessor.EncoderPreprocessor._flatten_and_load_videos,
        'if "qwen" in self.model_type:',
        '            video_processed = [\n                await preprocess_video(\n                    video, video_config=self.vision_config.get("video", {})\n                )\n                for video in video_items\n            ]',
    )
    original = preprocessor.preprocess_video
    if getattr(original, _MARK, False):
        return
    target = getattr(original, "func", original)
    if not asyncio.iscoroutinefunction(target):
        raise RuntimeError(
            f"encoder/preprocessor.preprocess_video ({target!r}) is not a coroutine function, so it cannot be driven with asyncio.run in a worker thread. This patch was derived from df2f34cca, where it is `async def`; re-derive it before serving."
        )
    threads = _int_env("VISIONWEAVE_ENCODER_DECODE_THREADS", 8)
    pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="visionweave-decode")

    async def offloaded_preprocess_video(vr, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(pool, lambda: asyncio.run(original(vr, *args, **kwargs)))

    setattr(offloaded_preprocess_video, _MARK, True)
    setattr(offloaded_preprocess_video, "_visionweave_video", True)
    preprocessor.preprocess_video = offloaded_preprocess_video
    logger.info(
        "[visionweave] video decode offloaded to %d threads (the loop no longer blocks per video)",
        threads,
    )


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = int(raw)
    if value <= 0:
        raise RuntimeError(f"{name}={value} must be a positive integer")
    return value


def install_encoder_admission() -> None:
    """Check that SGLang bounds each encoder to one batch or one video at a time."""
    if not _first_time("encoder_admission"):
        return
    from sglang.srt.disaggregation.encoder import runtime
    from sglang.srt.disaggregation.encoder import server as encoder_server
    from sglang.srt.disaggregation.encoder.server import MMEncoder

    _require(
        runtime.EncoderScheduler.start,
        "if self._worker_task is None:",
        "self._worker_task = asyncio.create_task(self._batch_worker())",
    )
    _require(
        runtime.EncoderScheduler._batch_worker,
        "batch = await self._collect_batch()",
        "for modality, group in groups.items():",
        "await self._dispatch_group(group, modality)",
    )
    _require_absent(runtime.EncoderScheduler._batch_worker, "create_task", "gather")
    _require(runtime.EncoderScheduler._collect_batch, "while len(batch) < self.max_batch_size:")
    _require(
        runtime.EncoderScheduler._dispatch_group,
        "if modality not in _BATCHABLE_MODALITIES:",
        "async with self.encoder.encode_dispatch_lock:",
        "results = await self.encoder.batch_encode(requests, modality)",
    )
    _require(
        runtime.execute_encode_pipeline,
        "if sched is not None and modality in _BATCHABLE_MODALITIES:",
        "result = await sched.submit(request)",
        "async with enc.encode_dispatch_lock:",
        "result = await _run_dispatched_encode(enc, request, modality)",
    )
    if runtime.Modality.VIDEO in runtime._BATCHABLE_MODALITIES:
        raise RuntimeError(
            "VIDEO is now in _BATCHABLE_MODALITIES, so video reaches EncoderScheduler and fuses with other requests. install_encoder_metadata's per-request payload slicing was derived for fused IMAGE batches only and has to be re-checked for video before this gate can pass."
        )
    _require(
        runtime.EncoderScheduler.submit,
        "await self.pending_queue.put(pending)",
        "return await asyncio.wait_for(pending.future, timeout=self.request_timeout)",
        "except asyncio.TimeoutError:",
        "await self.encoder.release_request(req_id)",
    )
    _require(
        runtime._push_embedding_to_prefill,
        'if backend == "zmq_to_tokenizer":',
        "finally:\n            await enc.release_request(req_id)",
    )
    _require_absent(MMEncoder, "embedding_to_send")
    logger.info(
        "[visionweave] encoder admission: RETIRED, upstream bounds it harder -- one batch (<=%s) or one video per process under encode_dispatch_lock, %.0fs request timeout, no hold buffer",
        runtime.ENCODER_MAX_BATCH_SIZE,
        encoder_server.ENCODER_REQ_TIMEOUT,
    )


def install_encoder_error_cleanup() -> None:
    """Check that failed encoder requests are delivered and released on every exit."""
    if not _first_time("encoder_error_cleanup"):
        return
    from sglang.srt.disaggregation.encoder import runtime
    from sglang.srt.disaggregation.encoder.server import MMEncoder

    _require(
        runtime.execute_encode_pipeline,
        "nbytes, embedding_len, embedding_dim, error_msg, error_code = result",
        "if error_msg:",
        "error_published = await _publish_pipeline_error(req_id, error_msg)",
        'if backend == "mooncake":',
        "await _push_embedding_to_prefill(\n                    enc,\n                    request,\n                    background_url_send=True,\n                )",
        "await _release_failed_request(enc, req_id)",
    )
    _require(
        runtime.execute_encode_pipeline,
        "await asyncio.shield(enc.release_request(req_id))",
        'preserve_metadata=backend == "mooncake" and error_published,',
    )
    _require(
        runtime._release_failed_request,
        "await enc.release_request(req_id, preserve_metadata=preserve_metadata)",
    )
    _require(
        MMEncoder.release_request,
        "self.req_states.pop(req_id, None)",
        "await self.delivery.release(state)",
        "state.embedding_data = None",
        "await meta_registry.discard(req_id)",
    )
    _require_absent(MMEncoder, "embedding_to_send")
    logger.info(
        "[visionweave] encoder error cleanup: RETIRED, the error EmbeddingData is now DELIVERED to the language side and released in a finally, on all four exits of execute_encode_pipeline"
    )


_FP32_ROUTER_ISLAND = ("visual.mask_net.", "visual.last_layer_bias")
_DTYPE_ADOPTED = "_visionweave_embedding_dtype_adopted"


def _adopt_tower_embedding_dtype(encoder) -> None:
    """Re-derive `_embedding_dtype` / `_element_size` from the dtype the tower really outputs."""
    if getattr(encoder, _DTYPE_ADOPTED, False):
        return
    import torch

    want = encoder.model_config.dtype
    guessed = encoder._embedding_dtype
    visual = getattr(encoder.model, "visual", None)
    merger = getattr(visual, "merger", None)
    if visual is None or merger is None:
        raise RuntimeError(
            f"visionweave.core_patches cannot re-derive the encoder embedding dtype: {type(encoder.model).__name__} has no `visual` / `visual.merger`. That pair is where the embedding this encoder publishes comes from; without it there is nothing to cross-check `model_config.dtype` against, and publishing an unchecked dtype is how this patch came to exist."
        )
    tower_dtype = getattr(visual, "dtype", None)
    merger_param = next(merger.parameters(), None)
    merger_dtype = None if merger_param is None else merger_param.dtype
    disagree = {
        name: dt
        for name, dt in (("visual.dtype", tower_dtype), ("visual.merger", merger_dtype))
        if dt is not None and dt != want
    }
    if disagree:
        raise RuntimeError(
            f"visionweave.core_patches was about to declare {want} as the encoder embedding dtype (from model_config.dtype), but {disagree} disagree. The declared dtype has to be the dtype of what `visual.merger` returns; re-derive this patch instead of letting the declaration and the tensor drift apart."
        )
    strays, island = ([], {})
    for name, param in encoder.model.named_parameters():
        if name.startswith(_FP32_ROUTER_ISLAND):
            island[name] = param.dtype
        elif param.dtype != want:
            strays.append(f"{name}/{param.dtype}")
    if strays:
        raise RuntimeError(
            f"visionweave.core_patches expected every parameter outside {_FP32_ROUTER_ISLAND} to be {want}, and these are not: {sorted(strays)[:8]} ({len(strays)} total). The encoder publishes ONE dtype for the whole embedding, so a mixed-dtype tower means the declaration cannot be derived from parameters at all -- re-derive this patch against whatever produced those tensors."
        )
    bad_island = sorted((f"{n}/{d}" for n, d in island.items() if d is not torch.float32))
    if not island or bad_island:
        raise RuntimeError(
            f"visionweave.core_patches expected the router head {_FP32_ROUTER_ISLAND} to exist and to be fp32; found {island or 'nothing'}. vision.py keeps it in fp32 to match the checkpoint's `_keep_in_fp32_modules_strict` because at threshold 0.5 a bf16 head flips every block within ~4e-3 of the boundary, and each flip changes the served sequence length. If the head really did move to {want}, update _FP32_ROUTER_ISLAND and vision.py together, not just this list."
        )
    encoder._embedding_dtype = want
    encoder._element_size = torch.tensor([], dtype=want).element_size()
    setattr(encoder, _DTYPE_ADOPTED, True)
    logger.info(
        "[visionweave] encoder embedding dtype: %s (model_config.dtype, == visual.dtype == visual.merger); upstream's next(model.parameters()).dtype guessed %s from the %d-tensor fp32 router head, element_size now %d",
        want,
        guessed,
        len(island),
        encoder._element_size,
    )


def install_encoder_embedding_dtype() -> None:
    """Declare the embedding dtype from the TOWER, not from `next(model.parameters())`."""
    if not _first_time("encoder_embedding_dtype"):
        return
    from sglang.srt.disaggregation.encoder.server import MMEncoder

    if getattr(MMEncoder._publish_preprocess_metadata, _MARK, False):
        return
    _require(
        MMEncoder.__init__,
        "self._embedding_dtype = next(self.model.parameters()).dtype",
        "self._element_size = torch.tensor(",
        "self._embedding_dims = self._infer_embedding_dims()",
    )
    _require(
        MMEncoder._embedding_state_for_stage,
        "and (metadata.shape != mm_data.shape or metadata.dtype != mm_data.dtype)",
        "Embedding metadata mismatch for ",
    )
    _require(
        MMEncoder._publish_preprocess_metadata,
        "if self.rank != 0:",
        "dtype=self._embedding_dtype,",
        "token_count * embedding_dim * self._element_size,",
    )
    original_publish = MMEncoder._publish_preprocess_metadata

    async def visionweave_publish_preprocess_metadata(self, ctx, requests):
        _adopt_tower_embedding_dtype(self)
        return await original_publish(self, ctx, requests)

    setattr(visionweave_publish_preprocess_metadata, _MARK, True)
    MMEncoder._publish_preprocess_metadata = visionweave_publish_preprocess_metadata


def install_receiver_metadata() -> None:
    """Register the route payload with UPSTREAM's per-part metadata channel: two rebinds, no wrappers."""
    if not _first_time("receiver_metadata"):
        return
    from sglang.srt.disaggregation.encoder import receiver

    if ROUTE_META_KEY in receiver._GENERAL_IMAGE_META_ATTRS or getattr(
        receiver.video_meta_attrs_for, _MARK, False
    ):
        return
    _require(
        receiver.MultiModalEmbeddingData.__init__,
        "self.video_meta_attrs = video_meta_attrs_for(model_type)",
        "for attr in _GENERAL_IMAGE_META_ATTRS:\n            setattr(self, attr, [None] * num_parts)",
        "for attr in self.video_meta_attrs:\n            setattr(self, attr, [None] * num_parts)",
        "self._set_part_grid(part_idx, modality, self.get_grid())",
        "self._set_video_meta_for_part(part_idx, kwargs)",
        "self._set_image_meta_for_part(part_idx, kwargs)",
    )
    _require(
        receiver.MultiModalEmbeddingData._set_image_meta_for_part,
        "for attr_name in _GENERAL_IMAGE_META_ATTRS:",
        "getattr(self, attr_name)[part_idx] = val",
    )
    _require(
        receiver.MultiModalEmbeddingData._set_video_meta_for_part,
        "for attr_name in self.video_meta_attrs:",
        "getattr(self, attr_name)[part_idx] = val",
    )
    _require(
        receiver.MultiModalEmbeddingData.from_embedding_data,
        "for attr in video_meta_attrs_for(model_type):",
        "for attr in _GENERAL_IMAGE_META_ATTRS:",
        "mm_data.send_time = embedding_data.send_time",
    )
    _require(
        receiver.MultiModalEmbeddingData.add,
        "self._set_video_meta_for_part(pid, embedding_data)",
        "self._set_image_meta_for_part(pid, embedding_data)",
    )
    _require(
        receiver.MultiModalEmbeddingData.get_mm_extra_meta,
        "for attr in self.video_meta_attrs:",
        "if attr in _VIDEO_META_TENSOR_ATTRS:",
        "kwargs[attr] = list(itertools.chain(*valid))",
        "for attr in _GENERAL_IMAGE_META_ATTRS:",
    )
    if ROUTE_META_KEY in receiver._VIDEO_META_TENSOR_ATTRS:
        raise RuntimeError(
            f"{ROUTE_META_KEY!r} is in _VIDEO_META_TENSOR_ATTRS, so upstream would aggregate it with torch.cat instead of chaining it as a list."
        )
    _require(
        receiver.MultiModalEmbeddingData.get_embedding,
        "return {mod: torch.cat(tensors, dim=0) for mod, tensors in groups.items()}",
    )
    _require_absent(
        receiver.MultiModalEmbeddingData.get_embedding, ".cuda()", '.to("cpu"', "non_blocking=True"
    )
    original_video_attrs = receiver.video_meta_attrs_for

    def visionweave_video_meta_attrs_for(model_type=None):
        return original_video_attrs(model_type) + (ROUTE_META_KEY,)

    visionweave_video_meta_attrs_for.__doc__ = (
        f"visionweave: appends {ROUTE_META_KEY!r} to every model_type's video meta attrs. "
        + (original_video_attrs.__doc__ or "")
    )
    setattr(visionweave_video_meta_attrs_for, _MARK, True)
    receiver.video_meta_attrs_for = visionweave_video_meta_attrs_for
    receiver._GENERAL_IMAGE_META_ATTRS = receiver._GENERAL_IMAGE_META_ATTRS + (ROUTE_META_KEY,)
    logger.info(
        "[visionweave] %r registered on both receiver metadata channels (image=%r video=%r)",
        ROUTE_META_KEY,
        receiver._GENERAL_IMAGE_META_ATTRS,
        receiver.video_meta_attrs_for(None),
    )


def install_receiver_fail_closed() -> None:
    """Assert that a failed encode CANNOT become local vision work -- upstream now guarantees it."""
    if not _first_time("receiver_fail_closed"):
        return
    from sglang.srt.managers import tokenizer_manager as tm

    _require(
        tm._reject_missing_dispatched_encoder_embedding,
        "Do not silently turn a failed EPD request into local vision work.",
        "mm_inputs is None",
        "and disagg.language_only",
        'and disagg.encoder_transfer_backend == "zmq_to_tokenizer"',
        "and request_obj.need_wait_for_mm_inputs",
        "raise fastapi.HTTPException(",
        "status_code=HTTPStatus.SERVICE_UNAVAILABLE,",
    )
    _require(
        tm.TokenizerManager._tokenize_one_request,
        "                    _reject_missing_dispatched_encoder_embedding(obj, mm_inputs)\n                if mm_inputs is None:",
        "mm_inputs = await self.mm_processor.process_mm_data_async(",
        'in ["zmq_to_scheduler", "mooncake"]',
        "and not obj.need_wait_for_mm_inputs",
    )
    from sglang.srt.disaggregation.encoder.receiver import MMReceiverBase

    _require(MMReceiverBase.__init__, "self.recv_timeout = envs.SGLANG_ENCODER_RECV_TIMEOUT.get()")
    _require(
        MMReceiverBase.recv_mm_data,
        "return_when=asyncio.FIRST_COMPLETED,",
        "encode_task.exception() is not None",
        "or encode_task.result() is not None",
        "Encoder dispatch failed; skipping embedding wait",
        "timeout=self.recv_timeout - (time.monotonic() - send_time),",
    )
    _require_absent(MMReceiverBase.recv_mm_data, "timeout=20,")
    _require(
        tm.TokenizerManager._handle_epd_disaggregation_encode_request,
        "if get_disagg().enable_adaptive_dispatch_to_encoder:",
        "obj.need_wait_for_mm_inputs = True",
        "obj.need_wait_for_mm_inputs = False",
    )
    logger.info(
        "[visionweave] fail-closed verified upstream: _reject_missing_dispatched_encoder_embedding raises 503 before the local-fallback branch; adaptive dispatch refused at startup"
    )


def install_receiver_inflight_gate() -> None:
    """Bound in-flight encodes per tokenizer worker, so queue wait does not burn the deadline."""
    if not _first_time("receiver_inflight_gate"):
        return
    import asyncio
    import math

    from sglang.srt.disaggregation.encoder.receiver import MMReceiverBase

    mark = _MARK + "_inflight"
    if getattr(MMReceiverBase.recv_mm_data, mark, False):
        return
    _require(
        MMReceiverBase.recv_mm_data,
        "send_time = time.monotonic()",
        "encode_task = asyncio.create_task(",
        "timeout=self.recv_timeout,",
        "Embedding recv timeout after",
    )
    _require(MMReceiverBase.__init__, "self.encode_urls: List[str] = (")
    from sglang.srt.disaggregation.encoder import runtime as encoder_runtime
    from sglang.srt.disaggregation.encoder.server import ENCODER_MAX_BATCH_SIZE as encoder_max_batch

    if encoder_runtime.Modality.VIDEO in encoder_runtime._BATCHABLE_MODALITIES:
        raise RuntimeError(
            "VIDEO is now in runtime._BATCHABLE_MODALITIES, so video reaches EncoderScheduler and the per-encoder video capacity is no longer 1. Re-derive this gate's video limit (and see install_encoder_admission, which refuses on the same fact)."
        )
    _require(
        encoder_runtime.execute_encode_pipeline,
        "if sched is not None and modality in _BATCHABLE_MODALITIES:",
        "async with enc.encode_dispatch_lock:",
    )
    if not isinstance(encoder_max_batch, int) or encoder_max_batch < 1:
        raise RuntimeError(
            f"encoder/server.ENCODER_MAX_BATCH_SIZE is {encoder_max_batch!r}, not a positive int; the image/audio in-flight limit cannot be derived from it."
        )
    original = MMReceiverBase.recv_mm_data
    override = os.environ.get("VISIONWEAVE_MAX_INFLIGHT_ENCODES")

    def _limit(self, slots_per_encoder: int) -> int:
        if override is not None and override.strip():
            return _int_env("VISIONWEAVE_MAX_INFLIGHT_ENCODES", 1)
        from sglang.srt.server_args import get_global_server_args

        args = get_global_server_args()
        if not hasattr(args, "tokenizer_worker_num"):
            raise RuntimeError(
                "ServerArgs has no `tokenizer_worker_num`, so the per-worker encode limit cannot be derived. Set VISIONWEAVE_MAX_INFLIGHT_ENCODES explicitly, or re-derive this gate."
            )
        workers = max(1, args.tokenizer_worker_num or 1)
        return max(1, math.ceil(len(self.encode_urls) * slots_per_encoder / workers))

    def _gate_for(self, request_obj):
        """Which of the two semaphores this request queues on, building it on first use."""
        has_video = bool(getattr(request_obj, "video_data", None))
        attr = "_visionweave_inflight_video" if has_video else "_visionweave_inflight_batch"
        gate = getattr(self, attr, None)
        if gate is None:
            slots = 1 if has_video else encoder_max_batch
            limit = _limit(self, slots)
            gate = asyncio.Semaphore(limit)
            setattr(self, attr, gate)
            logger.info(
                "[visionweave] this tokenizer worker admits %d concurrent %s encodes (%d slot(s) per encoder x %d encoder(s)); the rest wait here, before the receive deadline starts",
                limit,
                "video" if has_video else "image/audio",
                slots,
                len(self.encode_urls),
            )
        return gate

    _require(MMReceiverBase._extract_url_data, "(request_obj.video_data, Modality.VIDEO),")

    async def gated_recv_mm_data(self, request_obj, *args, **kwargs):
        async with _gate_for(self, request_obj):
            return await original(self, request_obj, *args, **kwargs)

    setattr(gated_recv_mm_data, mark, True)
    setattr(gated_recv_mm_data, _MARK, True)
    MMReceiverBase.recv_mm_data = gated_recv_mm_data


def installed_targets() -> list:
    """`[(label, live object)]` for every object the eleven installers mark, resolved fresh per call."""
    from sglang.srt.arg_groups import pd_disaggregation_hook as hook
    from sglang.srt.disaggregation.encoder import receiver
    from sglang.srt.disaggregation.encoder.preprocessor import EncoderPreprocessor
    from sglang.srt.disaggregation.encoder.preprocessor import (
        preprocess_video as encoder_preprocess_video,
    )
    from sglang.srt.disaggregation.encoder.server import MMEncoder

    return [
        (
            "pd_disaggregation_hook.handle_encoder_disaggregation",
            hook.handle_encoder_disaggregation,
        ),
        ("MMEncoder._validate_embedding_token_count", MMEncoder._validate_embedding_token_count),
        ("MMEncoder._stage_embedding_batch", MMEncoder._stage_embedding_batch),
        ("MMEncoder._publish_preprocess_metadata", MMEncoder._publish_preprocess_metadata),
        ("EncoderPreprocessor.__init__", EncoderPreprocessor.__init__),
        ("encoder/preprocessor.preprocess_video", encoder_preprocess_video),
        ("receiver.video_meta_attrs_for", receiver.video_meta_attrs_for),
        ("MMReceiverBase.recv_mm_data", receiver.MMReceiverBase.recv_mm_data),
    ]
