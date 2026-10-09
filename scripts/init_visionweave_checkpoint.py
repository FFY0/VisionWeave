"""Initialize a VisionWeave checkpoint from local, native Qwen3.5 weights."""

import argparse
import copy
import json
import logging
import math
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from visionweave import ARCHITECTURE, validate_geometry
from visionweave.compressor import VisionWeaveCompressionProjector
from visionweave.prepare import ASSETS, finite_threshold, read_json
from visionweave.router import VisionWeaveHierCrossAttnRouter, build_mask_net

LOGGER = logging.getLogger(__name__)
VISUAL_PREFIX = "model.visual."
EXTRA_PREFIXES = tuple(
    VISUAL_PREFIX + name
    for name in (
        "compression_projector.",
        "router_cross_attn.",
        "mask_net.",
        "last_layer_bias",
        "router_bias",
    )
)
DTYPES = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}


def _visionweave_config(native, seed, probability, threshold):
    config = copy.deepcopy(native)
    if config.get("model_type") != "qwen3_5" or config.get("architectures") != [
        "Qwen3_5ForConditionalGeneration"
    ]:
        raise ValueError("source must be a native dense Qwen3.5 checkpoint")
    vision, text = config["vision_config"], config["text_config"]
    if any(block.get("quantization_config") for block in (config, vision, text)):
        raise ValueError("quantized checkpoints are not supported")
    if vision.get("compression_spatial_merge_size", 1) != 1 or any(
        key.startswith("router_") for key in vision
    ):
        raise ValueError("source already contains compression or router configuration")
    if vision.get("patch_size") != 16 or vision.get("spatial_merge_size") != 2:
        raise ValueError("source must use patch_size=16 and spatial_merge_size=2")
    hidden, heads, depth = (vision[key] for key in ("hidden_size", "num_heads", "depth"))
    if (
        not all(isinstance(value, int) and value > 0 for value in (hidden, heads, depth))
        or depth < 2
        or hidden % heads
        or (hidden // heads) % 4
    ):
        raise ValueError(
            "invalid vision dimensions: depth >= 2 and head dimension divisible by 4 required"
        )
    if vision["out_hidden_size"] != text["hidden_size"]:
        raise ValueError("vision output size must match text hidden size")
    std = float(vision.get("initializer_range", 0.02))
    if not math.isfinite(std) or std <= 0:
        raise ValueError("vision initializer_range must be finite and positive")
    samples = [0, depth // 2, depth]
    vision.update(
        compression_spatial_merge_size=2,
        effective_compression_spatial_merge_size=1,
        router_cross_attn_depth=6,
        router_sample_depths=samples,
        router_cross_attn_layer_indices=[sample for sample in samples for _ in range(2)],
        router_cross_attn_num_heads=heads,
        router_cross_attn_mlp_ratio=2,
        router_cross_attn_use_kv_norm=True,
    )
    validate_geometry(SimpleNamespace(**vision))
    config["architectures"] = [ARCHITECTURE]
    config["visionweave"] = {
        "route_threshold": threshold,
        "initialization": {
            "type": "visionweave_init",
            "seed": seed,
            "compression_probability": probability,
            "compressor": "layernorm_mean_pool",
            "router_linear_std": std,
        },
    }
    for block in (config, vision, text):
        for key in ("auto_map", "_name_or_path"):
            block.pop(key, None)
    return config


def _inspect_weights(source, config):
    """Read shard headers, validating the index without loading the base model."""
    index_path = source / "model.safetensors.index.json"
    declared = read_json(index_path)["weight_map"] if index_path.is_file() else None
    if declared is not None and (not isinstance(declared, dict) or not declared):
        raise ValueError("weight_map must be a nonempty object")
    shards = sorted(set(declared.values())) if declared is not None else ["model.safetensors"]
    if any(
        not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors")
        for name in shards
    ):
        raise ValueError("weight shards must be safetensors files in the source directory")
    vision, text = config["vision_config"], config["text_config"]
    merged = vision["hidden_size"] * vision["spatial_merge_size"] ** 2
    expected_shapes = {
        "model.language_model.embed_tokens.weight": [text["vocab_size"], text["hidden_size"]],
        VISUAL_PREFIX + "patch_embed.proj.weight": [
            vision["hidden_size"],
            vision.get("in_channels", 3),
            vision.get("temporal_patch_size", 2),
            16,
            16,
        ],
        VISUAL_PREFIX + "merger.linear_fc1.weight": [merged, merged],
        VISUAL_PREFIX + "merger.linear_fc2.weight": [vision["out_hidden_size"], merged],
    }
    actual, found_shapes, dtypes = {}, {}, {}
    total_size = 0
    for shard in shards:
        path = source / shard
        if not path.is_file():
            raise ValueError(f"missing weight shard: {shard}")
        with safe_open(path, framework="pt", device="cpu") as weights:
            for name in weights.keys():
                if name in actual:
                    raise ValueError(f"duplicate tensor across shards: {name}")
                if name.startswith(EXTRA_PREFIXES):
                    raise ValueError(f"source already contains VisionWeave weights: {name}")
                actual[name] = shard
                if name in expected_shapes:
                    tensor = weights.get_slice(name)
                    found_shapes[name] = tensor.get_shape()
                    dtypes[name] = tensor.get_dtype()
        # Safetensors stores tensor data immediately after its length-prefixed JSON header.
        with path.open("rb") as stream:
            header_size = int.from_bytes(stream.read(8), "little")
        total_size += path.stat().st_size - 8 - header_size
    if declared is not None and actual != declared:
        raise ValueError("weight_map does not match the tensors in the source shards")
    for name, shape in expected_shapes.items():
        if found_shapes.get(name) != shape:
            raise ValueError(f"missing or incompatible {name}: expected shape {shape}")
    dtype_name = dtypes[VISUAL_PREFIX + "merger.linear_fc2.weight"]
    if dtype_name not in DTYPES:
        raise ValueError("native vision weights must be float32, float16 or bfloat16")
    return shards, actual, total_size, DTYPES[dtype_name]


def _initial_weights(config, dtype, seed, probability):
    vision = config["vision_config"]
    hidden = vision["hidden_size"]
    # Use only the CPU RNG and restore its state for callers importing this script.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(seed)
        compressor = VisionWeaveCompressionProjector(hidden, fine=2, pool=2)
        compressor.apply_transparent_init()
        router = VisionWeaveHierCrossAttnRouter(
            hidden, depth=6, num_heads=vision["router_cross_attn_num_heads"], mlp_ratio=2
        )
        for module in router.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=vision.get("initializer_range", 0.02))
                nn.init.zeros_(module.bias)
        mask = build_mask_net(hidden).float()
        nn.init.zeros_(mask[1].weight)
        weights = {}
        for prefix, module in (
            ("compression_projector", compressor.to(dtype=dtype)),
            ("router_cross_attn", router.to(dtype=dtype)),
            ("mask_net", mask),
        ):
            weights.update(
                (f"{VISUAL_PREFIX}{prefix}.{name}", tensor.contiguous())
                for name, tensor in module.state_dict().items()
            )
        weights[VISUAL_PREFIX + "last_layer_bias"] = torch.tensor(
            [0.0, math.log(probability) - math.log1p(-probability)], dtype=torch.float32
        )
        weights[VISUAL_PREFIX + "router_bias"] = torch.zeros(1, dtype=torch.float32)
    return weights


def initialize_checkpoint(
    source,
    destination,
    *,
    seed=0,
    initial_compression_prob=0.1,
    route_threshold=0.5,
    link_base_weights=False,
):
    """Write native weights plus initialized VisionWeave tensors into a new checkpoint."""
    source = Path(source).expanduser().resolve(strict=True)
    destination = Path(destination).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("destination must not already exist")
    destination = destination.resolve()
    if destination.is_relative_to(source):
        raise ValueError("destination must be outside the source directory")
    if not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    probability = float(initial_compression_prob)
    if not math.isfinite(probability) or not 0 < probability < 1:
        raise ValueError("initial compression probability must be strictly between 0 and 1")
    threshold = finite_threshold(route_threshold)
    config = _visionweave_config(read_json(source / "config.json"), seed, probability, threshold)
    shards, native_map, total_size, dtype = _inspect_weights(source, config)
    processor_names = ("preprocessor_config.json", "video_preprocessor_config.json")
    for name in ("tokenizer.json", "tokenizer_config.json", *processor_names):
        if not (source / name).is_file():
            raise ValueError(f"source is missing {name}")
    for name in processor_names:
        processor = read_json(source / name)
        if processor.get("patch_size") != 16 or processor.get("merge_size") != 2:
            raise ValueError(f"{name} must use patch_size=16 and merge_size=2")
    assets = [
        source / name
        for name in (
            *ASSETS,
            *processor_names,
            "processor_config.json",
            "LICENSE",
            "LICENSE.txt",
            "NOTICE",
        )
        if (source / name).is_file()
    ]
    LOGGER.info("Initializing compression and router tensors on CPU (seed=%d)", seed)
    weights = _initial_weights(config, dtype, seed, probability)
    config["dtype"] = str(dtype).removeprefix("torch.")
    config["vision_config"]["dtype"] = config["dtype"]
    total_size += sum(t.numel() * t.element_size() for t in weights.values())
    # Rename even a single native shard so loaders cannot prefer model.safetensors over the index.
    count = len(shards) + 1
    renamed = {
        name: f"model-{i:05d}-of-{count:05d}.safetensors" for i, name in enumerate(shards, 1)
    }
    extra_name = f"model-{count:05d}-of-{count:05d}.safetensors"
    weight_map = {name: renamed[shard] for name, shard in native_map.items()}
    weight_map.update((name, extra_name) for name in weights)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".visionweave-init-", dir=destination.parent))
    try:
        for name, target in renamed.items():
            LOGGER.info("%s %s", "Linking" if link_base_weights else "Copying", name)
            if link_base_weights:
                (temporary / target).symlink_to((source / name).resolve())
            else:
                shutil.copyfile(source / name, temporary / target)
        save_file(weights, temporary / extra_name, metadata={"format": "pt"})
        for path in assets:
            shutil.copyfile(path, temporary / path.name)
        for name, data in (
            ("config.json", config),
            (
                "model.safetensors.index.json",
                {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            ),
        ):
            (temporary / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        if destination.exists() or destination.is_symlink():
            raise ValueError("destination appeared during conversion; refusing to overwrite it")
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    LOGGER.info(
        "Added %d tensors; initial P(compress)=%.6g, threshold=%.6g",
        len(weights),
        probability,
        threshold,
    )
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="local native Qwen3.5 checkpoint directory")
    parser.add_argument(
        "--destination", required=True, help="new VisionWeave initialization checkpoint directory"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--initial-compression-prob",
        type=float,
        default=0.1,
        help="initial P(compress), default: 0.1",
    )
    parser.add_argument("--route-threshold", type=finite_threshold, default=0.5)
    parser.add_argument(
        "--link-base-weights",
        action="store_true",
        help="symlink native shards to save disk; the source must remain available",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        destination = initialize_checkpoint(
            args.source,
            args.destination,
            seed=args.seed,
            initial_compression_prob=args.initial_compression_prob,
            route_threshold=args.route_threshold,
            link_base_weights=args.link_base_weights,
        )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(f"Created VisionWeave initialization checkpoint: {destination}")
    print("The compression and routing weights are initialized, not trained.")


if __name__ == "__main__":
    main()
