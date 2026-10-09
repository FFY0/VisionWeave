"""Launch VisionWeave, FastV or VisionZip through SGLang."""

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError
from urllib.request import urlopen

from . import ARCHITECTURE, ROUTE_THRESHOLD_ENV, validate_geometry
from .baselines.config import CompressionConfig
from .prepare import finite_threshold, read_json


def gpu_list(value):
    devices = value.split(",")
    if (
        not devices
        or any(not item.isdecimal() for item in devices)
        or len(set(devices)) != len(devices)
    ):
        raise argparse.ArgumentTypeError("use distinct numeric GPU IDs, for example 0,1")
    return devices


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def memory_fraction(value):
    value = float(value)
    if not 0 < value < 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, type=Path)
    p.add_argument("--method", choices=("visionweave", "fastv", "visionzip"), default="visionweave")
    p.add_argument(
        "--gpus",
        type=gpu_list,
        default=["0"],
        help="language GPUs; baseline uses these for the whole model",
    )
    p.add_argument(
        "--encoder-gpus",
        type=gpu_list,
        help="one VisionWeave encoder per GPU; may share language GPUs",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=positive_int, default=30000)
    p.add_argument("--encoder-port", type=positive_int, default=31000)
    p.add_argument("--served-model-name", default=None)
    p.add_argument("--route-threshold", type=finite_threshold)
    p.add_argument("--keep-ratio", type=float, default=0.5)
    p.add_argument("--contextual-fraction", type=float, default=0.16)
    p.add_argument("--context-length", type=positive_int, default=32768)
    p.add_argument(
        "--image-tokens",
        type=positive_int,
        default=512,
        help="input native visual token budget per image",
    )
    p.add_argument(
        "--video-tokens",
        type=positive_int,
        default=512,
        help="input native visual token budget per frame",
    )
    p.add_argument("--max-frames", type=positive_int, default=64)
    p.add_argument("--mem-fraction", type=memory_fraction, default=0.65)
    p.add_argument("--encoder-mem-fraction", type=memory_fraction, default=0.10)
    p.add_argument("--max-running-requests", type=positive_int, default=8)
    p.add_argument("--attention-backend", default="fa3")
    p.add_argument("--startup-timeout", type=positive_int, default=900)
    p.add_argument("--log-dir", type=Path, default=Path("logs"))
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="validate config and print commands without starting services",
    )
    return p


@dataclass
class Role:
    name: str
    command: list[str]
    environment: dict[str, str]
    host: str
    port: int


def build_roles(args):
    model = args.model_path.resolve(strict=True)
    config = read_json(model / "config.json")
    if len(args.gpus) not in (1, 2):
        raise ValueError("this integration supports language TP1 or TP2")
    if config.get("model_type") != "qwen3_5":
        raise ValueError("only dense Qwen3.5 is supported")
    if (
        args.image_tokens < 4
        or args.video_tokens < 16
        or args.max_frames < 4
        or args.max_frames % 2
    ):
        raise ValueError(
            "image-tokens >= 4, video-tokens >= 16 and even max-frames >= 4 are required"
        )
    for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
        processor = read_json(model / name)
        if processor.get("merge_size") != 2 or processor.get("patch_size") != 16:
            raise ValueError(f"{name} requires merge_size=2 and patch_size=16")
    env = dict(os.environ)
    env.update(
        SGLANG_NUMA_BIND_V2="0",
        TORCH_NCCL_USE_COMM_NONBLOCKING="0",
        SGLANG_VIT_ENABLE_CUDA_GRAPH="0",
        SGLANG_RUST_SERVER="0",
        SGLANG_MM_SKIP_COMPUTE_HASH="0",
    )
    env.setdefault("SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT", "60")
    env.setdefault("SGLANG_ENCODER_RECV_TIMEOUT", "600")
    env.setdefault("SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_INTERVAL", "0")
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MM_PER_REQUEST_TIMEOUT", "600")
    env.pop("SGLANG_EXTERNAL_MM_MODEL_ARCH", None)
    budget = {
        "image": {"size": {"shortest_edge": 4096, "longest_edge": args.image_tokens * 1024}},
        "video": {
            "min_pixels": args.video_tokens * 256,
            "max_pixels": args.video_tokens * 1024,
            "total_pixels": args.video_tokens * 1024 * args.max_frames,
            "min_frames": 4,
            "max_frames": args.max_frames,
            "fps": 2.0,
        },
    }
    common = [
        "--model-path",
        str(model),
        "--attention-backend",
        args.attention_backend,
        "--cuda-graph-backend-prefill",
        "disabled",
        "--enforce-disable-flashinfer-allreduce-fusion",
    ]
    language = [
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tp-size",
        str(len(args.gpus)),
        "--served-model-name",
        args.served_model_name or args.method,
        "--context-length",
        str(args.context_length),
        "--mem-fraction-static",
        str(args.mem_fraction),
        "--tokenizer-worker-num",
        "1",
        "--mm-process-config",
        json.dumps(budget),
    ]
    if args.method != "visionweave":
        if args.encoder_gpus is not None or args.route_threshold is not None:
            raise ValueError("encoder-gpus and route-threshold apply only to VisionWeave")
        if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
            raise ValueError("FastV/VisionZip require a native Qwen3.5 checkpoint")
        cfg = CompressionConfig(args.method, args.keep_ratio, args.contextual_fraction)
        env.update(
            SGLANG_EXTERNAL_MODEL_PACKAGE="visionweave.baselines.models",
            SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE="visionweave.baselines.processors",
            CUDA_VISIBLE_DEVICES=",".join(args.gpus),
        )
        command = (
            [sys.executable, "-m", "sglang.launch_server"]
            + common
            + language
            + [
                "--max-running-requests",
                "1",
                "--chunked-prefill-size",
                "-1",
                "--disable-radix-cache",
                "--disable-overlap-schedule",
                "--cuda-graph-max-bs",
                "1",
                "--cuda-graph-backend-decode",
                "full",
                "--json-model-override-args",
                json.dumps({"post_vit_compression": cfg.as_dict()}),
            ]
        )
        roles = [Role(args.method, command, env, args.host, args.port)]
    else:
        if config.get("architectures") != [ARCHITECTURE]:
            raise ValueError("prepare this checkpoint first: python -m visionweave.prepare --help")
        validate_geometry(SimpleNamespace(**config["vision_config"]))
        threshold = args.route_threshold
        if threshold is None:
            threshold = config.get("visionweave", {}).get("route_threshold")
        if threshold is None:
            raise ValueError("missing route threshold; pass --route-threshold")
        encoders = args.encoder_gpus or args.gpus
        env.update(
            SGLANG_EXTERNAL_MODEL_PACKAGE="visionweave.models",
            SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE="visionweave.processors",
            SGLANG_EXTERNAL_MM_MODEL_ARCH=ARCHITECTURE,
        )
        env[ROUTE_THRESHOLD_ENV] = str(finite_threshold(threshold))
        base = (
            [sys.executable, "-m", "visionweave.launch"]
            + common
            + ["--encoder-transfer-backend", "zmq_to_tokenizer"]
        )
        roles = []
        for i, gpu in enumerate(encoders):
            port = args.encoder_port + i
            command = base + [
                "--encoder-only",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--tp-size",
                "1",
                "--mem-fraction-static",
                str(args.encoder_mem_fraction),
                "--mm-process-config",
                json.dumps({"vision_config": budget}),
            ]
            roles.append(
                Role(
                    f"encoder-{i}", command, env | {"CUDA_VISIBLE_DEVICES": gpu}, "127.0.0.1", port
                )
            )
        command = (
            base
            + language
            + ["--language-only", "--encoder-urls"]
            + [f"http://127.0.0.1:{role.port}" for role in roles]
            + [
                "--encoder-bootstrap-port",
                "0",
                "--max-running-requests",
                str(args.max_running_requests),
                "--cuda-graph-max-bs",
                str(args.max_running_requests),
            ]
        )
        roles.append(
            Role(
                "language",
                command,
                env | {"CUDA_VISIBLE_DEVICES": ",".join(args.gpus)},
                args.host,
                args.port,
            )
        )
    if any(not 1 <= r.port <= 65535 for r in roles) or len({r.port for r in roles}) != len(roles):
        raise ValueError("server ports must be distinct and in 1..65535")
    return roles


def stop_children(children):
    # Popen(start_new_session=True) guarantees each child owns this process group.
    for sig, timeout in ((signal.SIGTERM, 15), (signal.SIGKILL, 3)):
        for child in children:
            try:
                os.killpg(child.pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + timeout
        for child in children:
            try:
                child.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass


def run(roles, log_dir, timeout):
    for role in roles:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((role.host, role.port))
    log_dir.mkdir(parents=True, exist_ok=True)
    children, logs = [], []
    previous = {}

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, interrupted)
        for role in roles:
            log = (log_dir / f"{role.name}.log").open("a")
            logs.append(log)
            child = subprocess.Popen(
                role.command,
                env=role.environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            children.append(child)
            deadline = time.monotonic() + timeout
            host = "127.0.0.1" if role.host == "0.0.0.0" else role.host
            while True:
                for process in children:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"server exited with code {process.returncode}; see {log_dir}"
                        )
                try:
                    with urlopen(f"http://{host}:{role.port}/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError, ConnectionError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{role.name} did not become healthy; see {log_dir}")
                time.sleep(1)
            print(f"{role.name} ready on port {role.port}", flush=True)
        print(f"{roles[-1].name.upper()} SERVICE READY", flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(1)
        raise RuntimeError(f"a server exited; see {log_dir}")
    finally:
        stop_children(children)
        for log in logs:
            log.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        roles = build_roles(args)
        if args.dry_run:
            for role in roles:
                exports = {
                    k: v
                    for k, v in role.environment.items()
                    if k.startswith("SGLANG_EXTERNAL_")
                    or k in ("CUDA_VISIBLE_DEVICES", ROUTE_THRESHOLD_ENV)
                }
                print(
                    role.name
                    + ": "
                    + " ".join(f"{k}={shlex.quote(v)}" for k, v in exports.items())
                    + " "
                    + shlex.join(role.command)
                )
            return
        run(roles, args.log_dir.resolve(), args.startup_timeout)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        p.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
