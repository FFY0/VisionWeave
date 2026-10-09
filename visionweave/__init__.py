"""VisionWeave routed visual compression for SGLang."""

from typing import Tuple

POSITION_SCALE = 2
ARCHITECTURE = "Qwen3_5VisionWeaveForConditionalGeneration"
ROUTE_THRESHOLD_ENV = "VISIONWEAVE_ROUTE_THRESHOLD"
ROUTE_COMPARISON = ">="
__all__ = [
    "POSITION_SCALE",
    "ROUTE_COMPARISON",
    "ROUTE_THRESHOLD_ENV",
    "ARCHITECTURE",
    "derive_geometry",
    "validate_geometry",
    "route_threshold",
    "config_fingerprint",
]


def derive_geometry(vision_config) -> Tuple[int, int, int]:
    """`(fine, pool, coarse)` from the vision config, without judging them."""
    fine = int(getattr(vision_config, "spatial_merge_size", 2) or 2)
    pool = int(getattr(vision_config, "compression_spatial_merge_size", 1) or 1)
    return (fine, pool, fine * pool)


def validate_geometry(vision_config) -> Tuple[int, int, int]:
    """`derive_geometry`, rejecting anything this package cannot serve correctly."""
    fine, pool, coarse = derive_geometry(vision_config)
    if fine != 2 or pool != 2:
        raise ValueError(
            f"{ARCHITECTURE} requires vision_config.spatial_merge_size == 2 and compression_spatial_merge_size == 2, got fine={fine} pool={pool}."
        )
    eff = int(getattr(vision_config, "effective_compression_spatial_merge_size", 1) or 1)
    if eff != 1:
        raise ValueError(
            f"{ARCHITECTURE} requires effective_compression_spatial_merge_size == 1 (native LLM geometry), got {eff}. Only routed checkpoints with native LLM geometry are supported."
        )
    depth = int(getattr(vision_config, "router_cross_attn_depth", 0) or 0)
    sample_depths = list(getattr(vision_config, "router_sample_depths", []) or [])
    if depth != 6 or len(sample_depths) != 3:
        raise ValueError(
            f"{ARCHITECTURE} requires router_cross_attn_depth == 6 over three sampled depths, got depth={depth} sample_depths={sample_depths}."
        )
    num_blocks = int(getattr(vision_config, "depth", 0) or 0)
    if sample_depths != sorted(sample_depths) or any(
        (d < 0 or d > num_blocks for d in sample_depths)
    ):
        raise ValueError(
            f"router_sample_depths must be nondecreasing within [0, {num_blocks}], got {sample_depths}."
        )
    expanded = [d for sample in sample_depths for d in (sample, sample)]
    configured = list(getattr(vision_config, "router_cross_attn_layer_indices", expanded))
    if configured != expanded:
        raise ValueError(
            f"router_cross_attn_layer_indices must expand each sampled depth as (Local, Global), expected {expanded}, got {configured}."
        )
    if not bool(getattr(vision_config, "router_cross_attn_use_kv_norm", True)):
        raise ValueError("VisionWeave router requires router_cross_attn_use_kv_norm=true.")
    deepstack = list(getattr(vision_config, "deepstack_visual_indexes", []) or [])
    if deepstack:
        raise ValueError(
            f"{ARCHITECTURE} does not support deepstack; vision_config.deepstack_visual_indexes is {deepstack!r}, expected []."
        )
    return (fine, pool, coarse)


def route_threshold() -> float:
    """The effective hard-routing threshold, from the environment."""
    import math
    import os

    raw = os.environ.get(ROUTE_THRESHOLD_ENV)
    if raw is None or not raw.strip():
        raise RuntimeError(
            f"{ROUTE_THRESHOLD_ENV} is not set. The routed threshold is a property of the trained checkpoint (args.json `router_hard_threshold`) and must be exported by the launcher for both the encoder and the language server."
        )
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{ROUTE_THRESHOLD_ENV}={raw!r} is not a float.") from exc
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError(f"{ROUTE_THRESHOLD_ENV}={value} must be finite and non-negative.")
    return value


def config_fingerprint(hf_config) -> str:
    """Identity the encoder and the language server must agree on, as a short hex digest."""
    import hashlib
    import json

    vision = hf_config.vision_config
    fine, pool, coarse = derive_geometry(vision)
    payload = {
        "arch": list(getattr(hf_config, "architectures", []) or []),
        "position_scale": POSITION_SCALE,
        "fine": fine,
        "pool": pool,
        "coarse": coarse,
        "eff": int(getattr(vision, "effective_compression_spatial_merge_size", 1) or 1),
        "patch_size": int(getattr(vision, "patch_size", 0) or 0),
        "temporal_patch_size": int(getattr(vision, "temporal_patch_size", 0) or 0),
        "vision_hidden": int(getattr(vision, "hidden_size", 0) or 0),
        "out_hidden": int(getattr(vision, "out_hidden_size", 0) or 0),
        "vision_depth": int(getattr(vision, "depth", 0) or 0),
        "router_depth": int(getattr(vision, "router_cross_attn_depth", 0) or 0),
        "router_sample_depths": list(getattr(vision, "router_sample_depths", []) or []),
        "router_heads": int(getattr(vision, "router_cross_attn_num_heads", 0) or 0),
        "router_mlp_ratio": int(getattr(vision, "router_cross_attn_mlp_ratio", 0) or 0),
        "image_token_id": int(getattr(hf_config, "image_token_id", -1)),
        "video_token_id": int(getattr(hf_config, "video_token_id", -1)),
        "vision_start_token_id": int(getattr(hf_config, "vision_start_token_id", -1)),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
