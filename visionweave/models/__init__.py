"""Register the VisionWeave model and install encoder patches."""

from ..core_patches import (
    install_encoder_admission,
    install_encoder_decode_offload,
    install_encoder_embedding_dtype,
    install_encoder_error_cleanup,
    install_encoder_metadata,
    install_encoder_video_factor,
    install_encoder_video_timestamps,
)
from ..patches import install_all
from .qwen3_5 import EntryClass, Qwen3_5VisionWeaveForConditionalGeneration

install_all()
install_encoder_metadata()
install_encoder_embedding_dtype()
install_encoder_video_timestamps()
install_encoder_video_factor()
install_encoder_decode_offload()
install_encoder_admission()
install_encoder_error_cleanup()

__all__ = ["EntryClass", "Qwen3_5VisionWeaveForConditionalGeneration"]
