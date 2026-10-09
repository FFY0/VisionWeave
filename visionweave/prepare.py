"""Create a serving directory without modifying the source checkpoint."""

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from . import ARCHITECTURE, validate_geometry

ASSETS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "generation_config.json",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def finite_threshold(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("route threshold must be finite and non-negative")
    return value


def prepare(source, destination, tokenizer_source=None, threshold=None):
    source = Path(source).resolve(strict=True)
    destination = Path(destination).resolve()
    auxiliary = Path(tokenizer_source).resolve(strict=True) if tokenizer_source else source
    if (
        destination.exists()
        or destination.is_relative_to(source)
        or destination.is_relative_to(auxiliary)
    ):
        raise ValueError("destination must be new and outside the source directories")
    config = read_json(source / "config.json")
    if config.get("model_type") != "qwen3_5":
        raise ValueError("only dense Qwen3.5 checkpoints are supported")
    validate_geometry(SimpleNamespace(**config["vision_config"]))
    if config["vision_config"].get("patch_size") != 16:
        raise ValueError("VisionWeave requires patch_size=16")
    if threshold is None:
        threshold = config.get("visionweave", {}).get("route_threshold")
    if threshold is None and (source / "args.json").is_file():
        threshold = read_json(source / "args.json").get("router_hard_threshold")
    if threshold is None:
        raise ValueError("supply --route-threshold; no checkpoint threshold was found")
    config["visionweave"] = {
        **config.get("visionweave", {}),
        "route_threshold": finite_threshold(threshold),
    }
    config["architectures"] = [ARCHITECTURE]
    # Built-in Transformers configs load these weights; private remote code is unnecessary.
    for block in (config, config["vision_config"], config.get("text_config", {})):
        for key in ("auto_map", "_name_or_path"):
            block.pop(key, None)

    index = source / "model.safetensors.index.json"
    if index.is_file():
        weight_map = read_json(index)["weight_map"]
        for prefix in (
            "model.visual.compression_projector.",
            "model.visual.router_cross_attn.",
            "model.visual.mask_net.",
        ):
            if not any(key.startswith(prefix) for key in weight_map):
                raise ValueError(f"checkpoint is missing {prefix} weights")
        shards = sorted(set(weight_map.values()))
        if any(Path(name).name != name for name in shards):
            raise ValueError("weight shards must be files in the source directory")
        weights = [index] + [source / name for name in shards]
    else:
        weights = [source / "model.safetensors"]
    for path in weights:
        if not path.is_file():
            raise ValueError(f"missing weight file: {path.name}")

    materialized = {"config.json": config}
    for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
        path = source / name
        if not path.is_file():
            path = auxiliary / name
        processor = read_json(path)
        if processor.get("patch_size") != 16:
            raise ValueError(f"{name} must use patch_size=16")
        processor["merge_size"] = 2  # This determines patch order, not routed output count.
        # The launcher applies per-image/per-frame pixel limits before the HF video pass.
        processor["size"] = {"shortest_edge": 4096, "longest_edge": 2**31}
        for key in ("min_pixels", "max_pixels", "auto_map", "_name_or_path"):
            processor.pop(key, None)
        materialized[name] = processor
    # Nested processor objects can override the separate merge_size settings in Transformers.
    materialized["processor_config.json"] = {"processor_class": "Qwen3VLProcessor"}
    assets = []
    for name in ASSETS:
        path = auxiliary / name
        if not path.is_file():
            path = source / name
        if path.is_file():
            assets.append(path)
    if not {"tokenizer.json", "tokenizer_config.json"}.issubset({p.name for p in assets}):
        raise ValueError(
            "tokenizer.json and tokenizer_config.json are required; use --tokenizer-source"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".visionweave-", dir=destination.parent))
    try:
        for path in weights + assets:
            (temporary / path.name).symlink_to(path.resolve())
        for name, value in materialized.items():
            (temporary / name).write_text(json.dumps(value, indent=2) + "\n")
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="local routed checkpoint")
    parser.add_argument("--destination", required=True, help="new serving directory")
    parser.add_argument(
        "--tokenizer-source", help="original tokenizer/processor directory, if needed"
    )
    parser.add_argument("--route-threshold", type=finite_threshold)
    args = parser.parse_args(argv)
    try:
        print(prepare(args.source, args.destination, args.tokenizer_source, args.route_threshold))
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
