"""Algorithm settings shared by the processor and model."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CompressionConfig:
    method: str
    keep_ratio: float = 0.5
    contextual_frac: float = 0.16
    query_chunk_size: int = 128

    def __post_init__(self):
        if self.method not in ("fastv", "visionzip"):
            raise ValueError(f"unsupported post-ViT method: {self.method}")
        if not math.isfinite(self.keep_ratio) or not 0 < self.keep_ratio <= 1:
            raise ValueError("keep_ratio must be finite and in (0, 1]")
        if not math.isfinite(self.contextual_frac) or not 0 <= self.contextual_frac <= 1:
            raise ValueError("contextual_frac must be finite and in [0, 1]")
        if type(self.query_chunk_size) is not int or self.query_chunk_size < 1:
            raise ValueError("query_chunk_size must be a positive integer")

    @classmethod
    def from_hf(cls, config):
        value = getattr(config, "post_vit_compression", None)
        if not isinstance(value, dict):
            raise ValueError("external post-ViT model requires post_vit_compression")
        return cls(**value)

    def count(self, n):
        if n < 1:
            raise ValueError("empty visual frame")
        return max(1, round(n * self.keep_ratio))

    def as_dict(self):
        return asdict(self)


def validate_runtime(args):
    """Check resolved arguments, including model-specific SGLang adjustments."""
    from sglang.srt.runtime_context import get_context

    resolved = get_context().resolved_server_args_dict()
    required = {
        "max_running_requests": 1,
        "chunked_prefill_size": -1,
        "disable_radix_cache": True,
        "disable_overlap_schedule": True,
        "enable_mixed_chunk": False,
        "pp_size": 1,
        "attn_cp_size": 1,
        "dcp_size": 1,
        "dp_size": 1,
        "enable_dp_attention": False,
        "mm_enable_dp_encoder": False,
        "speculative_algorithm": None,
    }
    for key, expected in required.items():
        actual = resolved.get(key, "<missing>")
        if actual != expected:
            raise ValueError(f"post-ViT requires {key}={expected!r}, got {actual!r}")
    if resolved["tp_size"] not in (1, 2):
        raise ValueError("post-ViT supports TP1/TP2 only")
    from sglang.srt.environ import envs

    graph = resolved["cuda_graph_config"]
    if graph is None or graph["prefill"]["backend"] != "disabled":
        raise ValueError("post-ViT requires prefill CUDA graph disabled")
    if graph["decode"]["backend"] not in ("disabled", "full"):
        raise ValueError("post-ViT supports only disabled/full decode CUDA graph")
    if envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.get():
        raise ValueError("post-ViT requires ViT CUDA graph disabled")
