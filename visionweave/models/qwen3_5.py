# Portions derived from SGLang.
# Copyright 2025 Qwen Team
# Copyright 2025 SGLang Team
# Licensed under Apache-2.0; see LICENSES/Apache-2.0.txt.
# Modified for the VisionWeave routed serving integration.
"""SGLang model registration and checkpoint loading for VisionWeave."""

import logging
from contextlib import contextmanager
from typing import Iterable, Optional, Set, Tuple

import torch
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix

from .. import ARCHITECTURE, POSITION_SCALE, config_fingerprint, route_threshold, validate_geometry
from ..patches import install_all, set_geometry
from ..vision import Qwen3_5VisionWeaveVisionModel

logger = logging.getLogger(__name__)


@contextmanager
def _visual_names_exempt_from_the_layer_range():
    """Stop the language-layer range filter from eating the router's weights."""
    from sglang.srt.models import qwen3_5 as qwen3_5_module

    original = qwen3_5_module.get_layer_id

    def visual_aware_get_layer_id(name):
        return None if "visual" in name else original(name)

    qwen3_5_module.get_layer_id = visual_aware_get_layer_id
    try:
        yield
    finally:
        qwen3_5_module.get_layer_id = original


VISIONWEAVE_PARAM_PREFIXES = (
    "visual.compression_projector.",
    "visual.router_cross_attn.",
    "visual.mask_net.",
)
VISIONWEAVE_PARAM_NAMES = ("visual.last_layer_bias",)
EXPECTED_VISIONWEAVE_PARAMS = 15 + 109 + 3 + 1
ROUTER_BIAS_CHECKPOINT_NAME = "model.visual.router_bias"


class Qwen3_5VisionWeaveForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Native Qwen3.5 with the VisionWeave routed vision tower, in one of two roles."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        language_model_cls=Qwen3_5ForCausalLM,
    ) -> None:
        self.fine, self.pool, self.coarse = validate_geometry(config.vision_config)
        set_geometry(config.vision_config)
        install_all()
        server_args = get_global_server_args()
        if server_args.encoder_only and server_args.language_only:
            raise ValueError("--encoder-only and --language-only are mutually exclusive.")
        self.encoder_role = bool(getattr(config, "encoder_only", False))
        self.language_role = bool(getattr(config, "language_only", False))
        if self.encoder_role == self.language_role:
            raise ValueError(
                f"{ARCHITECTURE} must be served in exactly one role: --encoder-only for the vision/router process or --language-only for the LLM process (got encoder_only={self.encoder_role} language_only={self.language_role}). A single-process deployment cannot work: the routed length is only known after the ViT has run, and sglang fixes it before that."
            )
        super().__init__(config, quant_config, prefix, language_model_cls)
        if self.use_data_parallel:
            raise ValueError(
                "--mm-enable-dp-encoder shards one request's grids across ranks, which would split a route payload across processes. Serve the VisionWeave encoder at tp 1."
            )
        del self.visual
        if self.language_role:
            self.visual = None
            logger.info(
                "[visionweave] language-only role: visual tower dropped; embeddings arrive precomputed"
            )
        else:
            self.visual = Qwen3_5VisionWeaveVisionModel(
                config.vision_config,
                quant_config=None,
                norm_eps=getattr(config, "rms_norm_eps", 1e-06),
                prefix=add_prefix("model.visual", prefix),
                use_data_parallel=False,
            )
            self.visual.fingerprint = config_fingerprint(config)
            self.deepstack_visual_indexes = self.visual.deepstack_visual_indexes
            self.num_deepstack_embeddings = len(self.deepstack_visual_indexes)
        if self.language_role:
            self._assert_rope_is_scaled()
        logger.info(
            "[visionweave] %s ready: role=%s fine=%d pool=%d coarse=%d threshold=%.6g position_scale=%d fingerprint=%s",
            type(self).__name__,
            "encoder" if self.encoder_role else "language",
            self.fine,
            self.pool,
            self.coarse,
            route_threshold(),
            POSITION_SCALE,
            config_fingerprint(config),
        )

    def get_image_feature(self, items):
        if self.language_role:
            raise RuntimeError(
                "the VisionWeave language server was asked to run the vision tower. It has none: embeddings must arrive as precomputed_embeddings from the encoder. A raw pixel item here means the encoder path was bypassed."
            )
        with self.visual.modality_ctx("image"):
            return super().get_image_feature(items)

    def get_video_feature(self, items):
        if self.language_role:
            raise RuntimeError(
                "the VisionWeave language server was asked to run the vision tower. It has none: embeddings must arrive as precomputed_embeddings from the encoder."
            )
        with self.visual.modality_ctx("video"):
            return super().get_video_feature(items)

    def _assert_rope_is_scaled(self) -> None:
        """Confirm the language model's rope really is the scaled `MRotaryEmbedding`."""
        ropes = {id(m): m for m in self.modules() if isinstance(m, MRotaryEmbedding)}
        if not ropes:
            raise RuntimeError(
                "no MRotaryEmbedding found in the language model; the position scaling has nothing to attach to and the served positions would be native."
            )
        for rope in ropes.values():
            if type(rope) is not MRotaryEmbedding:
                raise RuntimeError(
                    f"rope is {type(rope).__name__}, not MRotaryEmbedding; it overrides the frequency computation that patches.py scales."
                )
            if getattr(rope, "_visionweave_position_scale", 1) != POSITION_SCALE:
                raise RuntimeError(
                    "MRotaryEmbedding was constructed without the position scaling (most likely built before patches.install_all(), then reused from get_rope's _ROPE_DICT cache). Every rope angle would be halved."
                )
        logger.info(
            "[visionweave] %d shared MRotaryEmbedding instance(s) carry position_scale=%d, cos_sin_cache rows=%s",
            len(ropes),
            POSITION_SCALE,
            [int(r.cos_sin_cache.shape[0]) for r in ropes.values()],
        )

    def _checked_weights(self, weights):
        """Pass weights through, inspecting `router_bias` and dropping visual ones for the LLM."""
        for name, tensor in weights:
            if name == ROUTER_BIAS_CHECKPOINT_NAME:
                if torch.count_nonzero(tensor).item():
                    raise RuntimeError(
                        f"{ROUTER_BIAS_CHECKPOINT_NAME} is {tensor.tolist()}, expected exactly zero. It is a buffer, so the parent loader would drop it without a word and the served compression rate would silently differ from the trained one."
                    )
                continue
            if self.language_role and ".visual." in f".{name}":
                continue
            yield (name, tensor)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        with _visual_names_exempt_from_the_layer_range():
            loaded = super().load_weights(self._checked_weights(weights))
        expected = {
            name
            for name, _ in self.named_parameters(remove_duplicate=False)
            if name.startswith(VISIONWEAVE_PARAM_PREFIXES) or name in VISIONWEAVE_PARAM_NAMES
        }
        if self.language_role:
            if expected:
                raise RuntimeError(
                    f"the language-only role still exposes {len(expected)} visual parameters; the tower was supposed to be dropped."
                )
            logger.info("[visionweave] language-only role loaded, no visual tensors expected")
            return loaded
        if len(expected) != EXPECTED_VISIONWEAVE_PARAMS:
            raise RuntimeError(
                f"the VisionWeave tower exposes {len(expected)} routed parameters, expected {EXPECTED_VISIONWEAVE_PARAMS}: {sorted(expected)}"
            )
        missing = sorted(expected - set(loaded or ()))
        if missing:
            raise RuntimeError(
                f"the checkpoint did not supply {len(missing)}/{len(expected)} VisionWeave tensors: {missing}. In encoder/language role the parent loader SILENTLY skips names it cannot place (qwen3_vl.py:1315-1319), so without this check the encoder would route with a randomly-initialised router. Check that --model-path points at the VisionWeave checkpoint (its tensors are named `model.visual.router_cross_attn.*`, `model.visual.mask_net.*`) and that the serve dir was derived with python -m visionweave.prepare."
            )
        logger.info(
            "[visionweave] all %d routed visual tensors loaded from the checkpoint", len(expected)
        )
        return loaded


assert Qwen3_5VisionWeaveForConditionalGeneration.__name__ == ARCHITECTURE, (
    f"the entry class is named {Qwen3_5VisionWeaveForConditionalGeneration.__name__!r} but the package's canonical arch name is {ARCHITECTURE!r}; the processor package registers that name independently (it must not import this module), and a mismatch means the model and its processor would be selected by different keys."
)
EntryClass = [Qwen3_5VisionWeaveForConditionalGeneration]
