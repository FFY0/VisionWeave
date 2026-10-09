"""Check the CUDA serving stack and SGLang patch interfaces in an isolated process."""

import os
from importlib import metadata


def main():
    required = {
        "torch": "2.13.0+cu129",
        "transformers": "5.12.1",
        "sglang-kernel": "0.4.6.post1+cu129",
        "torchcodec": "0.15.0+cu129",
    }
    for name, version in required.items():
        actual = metadata.version(name)
        if actual != version:
            raise RuntimeError(f"{name}: expected {version}, found {actual}")
        print(f"{name} {actual}", flush=True)
    version = metadata.version("sglang")
    if "gdf2f34cca" not in version:
        raise RuntimeError(f"SGLang must be built from df2f34cca, found {version}")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the NVIDIA driver and container GPU access")
    print(f"GPU: {torch.cuda.get_device_name(0)}; CUDA {torch.version.cuda}", flush=True)
    os.environ.setdefault("VISIONWEAVE_ROUTE_THRESHOLD", "0.5")
    from . import models, processors  # noqa: F401
    from .core_patches import install_arch_allowlist

    install_arch_allowlist()
    from sglang.srt.utils.video_decoder import _BACKEND

    if _BACKEND == "decord":
        import decord  # noqa: F401

    print(f"Video decoder: {_BACKEND}")

    from .processors.processor import VisionWeaveImageProcessor  # noqa: F401

    print("VisionWeave model, processor, video decoder and SGLang patch interfaces: OK")


if __name__ == "__main__":
    main()
