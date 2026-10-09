from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

from visionweave import video  # noqa: F401 - install video alignment.

from ..config import CompressionConfig, validate_runtime
from ..layout import compress_layout


class Qwen3_5ForConditionalGeneration:
    pass


class PostVitProcessor(QwenVLImageProcessor):
    models = [Qwen3_5ForConditionalGeneration]

    def __init__(self, hf_config, server_args, processor, *args, **kwargs):
        self.post_vit_config = CompressionConfig.from_hf(hf_config)
        validate_runtime(server_args)
        super().__init__(hf_config, server_args, processor, *args, **kwargs)

    async def process_mm_data_async(self, *args, **kwargs):
        output = await super().process_mm_data_async(*args, **kwargs)
        return compress_layout(output, self.post_vit_config, self._spatial_merge_size)
