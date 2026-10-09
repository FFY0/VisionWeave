"""Align video dimensions to the 64-pixel routing grid."""

import functools
import inspect
import logging

from sglang.srt.multimodal.processors import qwen_vl

logger = logging.getLogger(__name__)
VIDEO_FACTOR = 64
UPSTREAM_IMAGE_FACTOR = 28
_MARKER = "[visionweave] video resize factor"


def _apply() -> None:
    if getattr(qwen_vl.preprocess_video, "_visionweave_video", False):
        return
    assert qwen_vl.IMAGE_FACTOR == UPSTREAM_IMAGE_FACTOR, (
        f"expected upstream IMAGE_FACTOR {UPSTREAM_IMAGE_FACTOR}, found {qwen_vl.IMAGE_FACTOR}; sglang changed underneath this patch -- re-derive it before serving"
    )
    params = inspect.signature(qwen_vl.preprocess_video).parameters
    assert "image_factor" in params, (
        "preprocess_video no longer takes image_factor; this patch is stale"
    )
    assert params["image_factor"].default == UPSTREAM_IMAGE_FACTOR, (
        f"preprocess_video image_factor default is {params['image_factor'].default}, expected {UPSTREAM_IMAGE_FACTOR}"
    )
    patched = functools.partial(qwen_vl.preprocess_video, image_factor=VIDEO_FACTOR)
    patched._visionweave_video = True
    qwen_vl.preprocess_video = patched
    msg = f"{_MARKER} {UPSTREAM_IMAGE_FACTOR} -> {VIDEO_FACTOR} (matching qwen_vl_utils on the training side)"
    logger.info(msg)
    print(msg, flush=True)


_apply()
