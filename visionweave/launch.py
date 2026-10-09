"""Launch one VisionWeave SGLang encoder or language role."""

import os
import sys
import warnings

from . import ARCHITECTURE, ROUTE_THRESHOLD_ENV, route_threshold
from .core_patches import (
    install_arch_allowlist,
    install_encoder_admission,
    install_encoder_decode_offload,
    install_encoder_embedding_dtype,
    install_encoder_error_cleanup,
    install_encoder_metadata,
    install_encoder_video_factor,
    install_encoder_video_timestamps,
    install_receiver_fail_closed,
    install_receiver_inflight_gate,
    install_receiver_metadata,
)

_REQUIRED_TRANSFER_BACKEND = "zmq_to_tokenizer"
_FORBIDDEN_CACHE_FLAGS = ("enable_prefix_mm_cache", "enable_mm_global_cache")
_FORBIDDEN_DISPATCH_FLAG = "enable_adaptive_dispatch_to_encoder"


def _resolved(server_args, field: str):
    """What resolution DECIDED for `field` -- which is not what the field holds."""
    from sglang.srt.arg_groups.overrides import resolution_result

    return resolution_result(server_args, field)


def _check_transfer_backend(server_args, when: str) -> None:
    """Refuse any encoder transfer backend but `zmq_to_tokenizer`."""
    backend = _resolved(server_args, "encoder_transfer_backend")
    if backend != _REQUIRED_TRANSFER_BACKEND:
        raise SystemExit(
            f"visionweave.launch: encoder transfer backend is {backend!r} ({when}), but VisionWeave requires {_REQUIRED_TRANSFER_BACKEND!r}. VisionWeave sequence length is only known after the ViT and router have run, so the embeddings must arrive at the tokenizer process, where input_ids are built from them; zmq_to_scheduler and mooncake deliver to the scheduler ranks, after input_ids already exist. Note the default (--encoder-transfer-backend auto) resolves to zmq_to_scheduler for this architecture, and argparse honours the LAST occurrence of the flag."
        )


def _check_caches(server_args, when: str) -> None:
    """Refuse the two embedding caches until their keys are known to cover the route."""
    on = [f for f in _FORBIDDEN_CACHE_FLAGS if _resolved(server_args, f)]
    if on:
        raise SystemExit(
            "visionweave.launch: "
            + ", ".join(("--" + f.replace("_", "-") for f in on))
            + f" is set ({when}). VisionWeave embeddings are routed products: their cache key must cover the route fingerprint and threshold, and today it covers content only, so a hit can return an embedding of a different length together with a payload that agrees with it. Serve without the caches."
        )


def _check_dispatch(server_args, when: str) -> None:
    """Refuse `--enable-adaptive-dispatch-to-encoder`: it routes small requests AROUND the encoder."""
    if _resolved(server_args, _FORBIDDEN_DISPATCH_FLAG):
        raise SystemExit(
            f"visionweave.launch: --enable-adaptive-dispatch-to-encoder is set ({when}). It deliberately processes small multimodal requests locally, and VisionWeave cannot: the routed token count only exists after the encoder's ViT and router have run, and this process has neither. Every request must go to the encoder."
        )


def _check_env() -> None:
    """Refuse to start if the registration variables do not name THIS package."""
    expected = {
        "SGLANG_EXTERNAL_MODEL_PACKAGE": "visionweave.models",
        "SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE": "visionweave.processors",
        "SGLANG_EXTERNAL_MM_MODEL_ARCH": ARCHITECTURE,
    }
    wrong = {k: os.environ.get(k) for k, v in expected.items() if os.environ.get(k) != v}
    if wrong:
        raise SystemExit(
            f"visionweave.launch: these registration variables are not set to this package's values: {wrong}. Expected {expected}. Start the server through python -m visionweave.serve, which sets all three."
        )


def main(argv=None) -> None:
    from . import video  # noqa: F401 - install alignment before checking encoder bindings.

    threshold = route_threshold()
    _check_env()
    install_arch_allowlist()
    install_encoder_metadata()
    install_encoder_embedding_dtype()
    install_encoder_video_timestamps()
    install_encoder_video_factor()
    install_encoder_decode_offload()
    install_encoder_admission()
    install_encoder_error_cleanup()
    install_receiver_metadata()
    install_receiver_fail_closed()
    install_receiver_inflight_gate()
    from sglang.launch_server import run_server
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    load_plugins()
    server_args = prepare_server_args(list(sys.argv[1:] if argv is None else argv))
    if server_args.encoder_only == server_args.language_only:
        raise SystemExit(
            f"visionweave.launch requires exactly one of --encoder-only / --language-only (got encoder_only={server_args.encoder_only} language_only={server_args.language_only}). A single-process VisionWeave deployment cannot work: the routed token count is only known after the ViT has run, and sglang fixes placeholders, mrope, radix keys, the prefill budget and the KV allocation before that."
        )
    _check_transfer_backend(server_args, "as given on the command line")
    _check_caches(server_args, "as given on the command line")
    _check_dispatch(server_args, "as given on the command line")
    server_args.resolve_once()
    _check_transfer_backend(server_args, "after argument resolution")
    _check_caches(server_args, "after argument resolution")
    _check_dispatch(server_args, "after argument resolution")
    role = "encoder" if server_args.encoder_only else "language"
    print(
        f"[visionweave] launching {role} role for {ARCHITECTURE} ({ROUTE_THRESHOLD_ENV}={threshold:.6g}, transfer={_resolved(server_args, 'encoder_transfer_backend')}, caches off, every request dispatched to the encoder)",
        flush=True,
    )
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="sglang.launch_server")
    main()
