"""Install VisionWeave position and receiver patches."""

from ..core_patches import (
    install_receiver_fail_closed,
    install_receiver_inflight_gate,
    install_receiver_metadata,
)
from ..patches import install_all

install_all()
install_receiver_metadata()
install_receiver_fail_closed()
install_receiver_inflight_gate()

# SGLang downgrades errors during submodule discovery to warnings. Import here
# so a broken routed processor fails startup instead of selecting a native one.
from .processor import VisionWeaveImageProcessor  # noqa: E402

__all__ = ["VisionWeaveImageProcessor"]
