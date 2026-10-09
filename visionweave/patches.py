"""Image grid alignment and routed position guards."""

import importlib
import inspect
import logging

from . import ARCHITECTURE, derive_geometry

logger = logging.getLogger(__name__)
__all__ = [
    "install_all",
    "install_resize_factor_patch",
    "install_rope_index_tripwire",
    "set_geometry",
]
_MARK = "_visionweave"
_installed: set = set()
_POOL = None


def set_geometry(vision_config) -> None:
    """Publish `pool` for the resize patch, which is installed before any config is parsed."""
    global _POOL
    _, pool, _ = derive_geometry(vision_config)
    if _POOL is not None and _POOL != pool:
        raise RuntimeError(
            f"visionweave already configured for pool={_POOL}, cannot reconfigure to {pool} in the same process."
        )
    _POOL = pool


def _pool() -> int:
    if _POOL is None:
        raise RuntimeError(
            "visionweave.patches.set_geometry() was never called in this process, so the resize factor is unknown. Rounding images on the native factor 32 would give grids the router refuses -- raising instead."
        )
    return _POOL


def _first_time(name: str) -> bool:
    if name in _installed:
        return False
    _installed.add(name)
    return True


def install_resize_factor_patch() -> None:
    """Multiply `smart_resize`'s `factor` by `pool` for every IMAGE binding."""
    if not _first_time("resize_factor"):
        return
    from transformers.models.qwen2_vl import image_processing_qwen2_vl as tensor_mod

    if getattr(tensor_mod.smart_resize, _MARK, False):
        return
    original = tensor_mod.smart_resize
    params = inspect.signature(original).parameters
    for name in ("height", "width", "factor"):
        if name not in params:
            raise RuntimeError(f"smart_resize has no {name!r} parameter any more: {list(params)}")
    upstream_default = params["factor"].default

    def make_visionweave(upstream, default):

        def visionweave_smart_resize(height, width, factor=default, *args, **kwargs):
            return upstream(height, width, factor * _pool(), *args, **kwargs)

        visionweave_smart_resize.__doc__ = (
            "visionweave: multiplies `factor` by the compressor's pool size so the grid is divisible by coarse. "
            + (upstream.__doc__ or "")
        )
        setattr(visionweave_smart_resize, _MARK, True)
        return visionweave_smart_resize

    tensor_mod.smart_resize = make_visionweave(original, upstream_default)
    patched = ["image_processing_qwen2_vl"]
    for modname in ("image_processing_pil_qwen2_vl", "image_processing_qwen2_vl_fast"):
        try:
            mod = importlib.import_module(f"transformers.models.qwen2_vl.{modname}")
        except ImportError:
            continue
        current = getattr(mod, "smart_resize", None)
        if current is None or getattr(current, _MARK, False):
            continue
        code = current.__code__
        names = code.co_varnames[: code.co_argcount]
        if names[:3] != ("height", "width", "factor"):
            raise RuntimeError(
                f"transformers.models.qwen2_vl.{modname}.smart_resize takes {names!r}, not (height, width, factor, ...); the `factor * pool` rewrite would land on the wrong argument."
            )
        defaults = current.__defaults__ or ()
        own_default = dict(zip(names[len(names) - len(defaults) :], defaults)).get(
            "factor", upstream_default
        )
        mod.smart_resize = make_visionweave(current, own_default)
        patched.append(modname)
    logger.info(
        "[visionweave] image smart_resize factor x pool in %s (upstream default %s); video left to visionweave.video",
        ", ".join(patched),
        upstream_default,
    )


def install_rope_index_tripwire() -> None:
    """Make `MRotaryEmbedding.get_rope_index` raise instead of returning native positions."""
    if not _first_time("rope_index_tripwire"):
        return
    from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding

    if getattr(MRotaryEmbedding.get_rope_index, _MARK, False):
        return

    def visionweave_get_rope_index(*args, **kwargs):
        raise RuntimeError(
            f"MRotaryEmbedding.get_rope_index was called while serving {ARCHITECTURE}. VisionWeave positions cannot be derived from a grid: kept and compressed tokens interleave per block, so they come from the route payload (processors/processor.py). This call site is unaudited -- if it is legitimate, it needs a routed implementation, not a native fallback."
        )

    setattr(visionweave_get_rope_index, _MARK, True)
    MRotaryEmbedding.get_rope_index = staticmethod(visionweave_get_rope_index)
    logger.info(
        "[visionweave] MRotaryEmbedding.get_rope_index -> tripwire (positions come from route)"
    )


def install_all() -> None:
    """Install everything, in every process, idempotently."""
    from . import (
        position_patches,
        video,  # noqa: F401 - install alignment in every worker.
    )

    install_resize_factor_patch()
    install_rope_index_tripwire()
    position_patches.install_position_scale_rope_patch()
    position_patches.install_mrope_arithmetic_patches()
    position_patches.install_mrope_fastpath_guard()
