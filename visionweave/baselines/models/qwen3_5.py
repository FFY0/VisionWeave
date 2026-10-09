import logging

import torch
from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration as NativeQwen
from sglang.srt.runtime_context import get_server_args

from ..config import CompressionConfig, validate_runtime
from ..layout import FRAMES, POSITIONS, fill_positions
from ..reduction import reduce_frame
from ..scoring import observe_last_attention

logger = logging.getLogger(__name__)


class Qwen3_5ForConditionalGeneration(NativeQwen):
    def __init__(self, config, quant_config=None, prefix=""):
        compression = CompressionConfig.from_hf(config)
        validate_runtime(get_server_args())
        if config.vision_config.deepstack_visual_indexes or quant_config is not None:
            raise ValueError("post-ViT requires native unquantized ViT without deepstack")
        super().__init__(config, quant_config, prefix)
        self.post_vit_config = compression
        if self.visual is None or self.use_data_parallel or (not self.is_mrope_enabled):
            raise ValueError("post-ViT requires local native ViT and M-RoPE")
        attn = self.visual.blocks[-1].attn
        if attn.num_attention_heads_per_partition * attn.tp_size != self.visual.num_heads:
            raise ValueError("dummy ViT heads are unsupported")
        logger.info("[post-vit] model registered: %s", compression.as_dict())

    @torch.no_grad()
    def forward(
        self, input_ids, positions, forward_batch, get_embedding=False, pp_proxy_tensors=None
    ):
        cfg = self.post_vit_config
        mm_list = forward_batch.mm_inputs
        if (
            cfg.keep_ratio < 1
            and (not forward_batch.forward_mode.is_decode())
            and mm_list
            and any((mm and mm.mm_items for mm in mm_list))
        ):
            if (
                forward_batch.batch_size != 1
                or len(mm_list) != 1
                or forward_batch.extend_prefix_lens_cpu != [0]
                or (forward_batch.extend_seq_lens_cpu != [input_ids.numel()])
            ):
                raise ValueError("post-ViT requires one complete prefill with prefix_length=0")
            mm = mm_list[0]
            if mm.mrope_positions.shape != (3, input_ids.numel()):
                raise ValueError("post-ViT prefill length differs from processor layout")
            native_count = kept_count = 0
            for item in mm.mm_items:
                if (
                    FRAMES not in item.model_specific_data
                    or item.precomputed_embeddings is not None
                ):
                    raise ValueError(
                        "post-ViT processor/model contract missing or already consumed"
                    )
                with observe_last_attention(self.visual, cfg) as captured:
                    getter = self.get_image_feature if item.is_image() else self.get_video_feature
                    features = getter([item])
                scores, keys = captured.pop()
                parts, selections = ([], [])
                frames = item.model_specific_data[FRAMES]
                if sum((row[1] for row in frames)) != features.shape[0]:
                    raise ValueError("native merger rows differ from processor grid")
                for offset, n, k, start in frames:
                    reduced, selected = reduce_frame(
                        features[offset : offset + n],
                        scores[offset : offset + n],
                        keys[offset : offset + n],
                        cfg,
                    )
                    if reduced.shape[0] != k:
                        raise ValueError("reducer output differs from reserved placeholders")
                    parts.append(reduced)
                    selections.append(selected)
                    native_count += n
                    kept_count += k
                fill_positions(mm, item, selections)
                item.precomputed_embeddings = torch.cat(parts)
                item.feature = None
                del item.model_specific_data[FRAMES]
                del item.model_specific_data[POSITIONS]
            forward_batch.mrope_positions.copy_(mm.mrope_positions)
            logger.info(
                "[post-vit-route] method=%s ratio=%s native=%d kept=%d frames=%d",
                cfg.method,
                cfg.keep_ratio,
                native_count,
                kept_count,
                sum((len(i.offsets) for i in mm.mm_items)),
            )
        return super().forward(
            input_ids,
            positions,
            forward_batch,
            get_embedding=get_embedding,
            pp_proxy_tensors=pp_proxy_tensors,
        )


EntryClass = Qwen3_5ForConditionalGeneration
