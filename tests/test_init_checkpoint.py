import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from scripts.init_visionweave_checkpoint import initialize_checkpoint
from visionweave import ARCHITECTURE, validate_geometry
from visionweave.compressor import VisionWeaveCompressionProjector
from visionweave.prepare import prepare
from visionweave.route import hard_route_layout
from visionweave.router import VisionWeaveHierCrossAttnRouter, build_mask_net, router_probabilities


def make_native(root, *, sharded=True, dtype=torch.float32):
    root.mkdir()
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "tie_word_embeddings": True,
        "text_config": {"hidden_size": 12, "vocab_size": 32, "num_hidden_layers": 1},
        "vision_config": {
            "hidden_size": 8,
            "num_heads": 2,
            "depth": 4,
            "out_hidden_size": 12,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "deepstack_visual_indexes": [],
            "initializer_range": 0.02,
        },
    }
    (root / "config.json").write_text(json.dumps(config))
    generator = torch.Generator().manual_seed(123)
    shapes = {
        "model.language_model.embed_tokens.weight": (32, 12),
        "model.language_model.layers.0.mlp.down_proj.weight": (12, 8),
        "model.visual.patch_embed.proj.weight": (8, 3, 2, 16, 16),
        "model.visual.merger.linear_fc1.weight": (32, 32),
        "model.visual.merger.linear_fc2.weight": (12, 32),
    }
    tensors = {
        name: torch.randn(shape, generator=generator).to(dtype) for name, shape in shapes.items()
    }
    if sharded:
        items = list(tensors.items())
        first, second = dict(items[:2]), dict(items[2:])
        save_file(first, root / "part1.safetensors")
        save_file(second, root / "part2.safetensors")
        weight_map = {name: "part1.safetensors" for name in first}
        weight_map.update({name: "part2.safetensors" for name in second})
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    else:
        save_file(tensors, root / "model.safetensors")
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        (root / name).write_text("{}")
    for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
        (root / name).write_text(json.dumps({"patch_size": 16, "merge_size": 2, "size": {}}))
    (root / "LICENSE").write_text("synthetic source license\n")
    (root / "args.json").write_text('{"unrelated_training_setting":true}')
    return config, tensors


def read_checkpoint(root):
    index = json.loads((root / "model.safetensors.index.json").read_text())
    tensors = {}
    for shard in set(index["weight_map"].values()):
        tensors.update(load_file(root / shard))
    return index, tensors


def hashes(root):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}


class InitializationTests(unittest.TestCase):
    def test_preserves_native_tensors_and_emits_complete_portable_checkpoint(self):
        for sharded, dtype in ((False, torch.float32), (True, torch.bfloat16)):
            with self.subTest(sharded=sharded), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "native"
                native_config, native = make_native(source, sharded=sharded, dtype=dtype)
                before = hashes(source)
                out = initialize_checkpoint(source, root / "init")
                self.assertEqual(before, hashes(source))
                self.assertFalse(any(p.is_symlink() for p in out.iterdir()))
                self.assertFalse((out / "model.safetensors").exists())
                self.assertFalse((out / "args.json").exists())
                self.assertEqual((out / "LICENSE").read_bytes(), (source / "LICENSE").read_bytes())
                for name in (
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "preprocessor_config.json",
                    "video_preprocessor_config.json",
                ):
                    self.assertEqual((out / name).read_bytes(), (source / name).read_bytes())
                for shard in source.glob("*.safetensors"):
                    self.assertIn(
                        hashlib.sha256(shard.read_bytes()).hexdigest(), hashes(out).values()
                    )
                config = json.loads((out / "config.json").read_text())
                self.assertEqual(config["architectures"], [ARCHITECTURE])
                self.assertEqual(
                    config["visionweave"]["initialization"]["type"], "visionweave_init"
                )
                self.assertEqual(config["text_config"], native_config["text_config"])
                self.assertEqual(
                    validate_geometry(SimpleNamespace(**config["vision_config"])), (2, 2, 4)
                )
                self.assertEqual(
                    config["vision_config"]["router_cross_attn_layer_indices"], [0, 0, 2, 2, 4, 4]
                )
                index, tensors = read_checkpoint(out)
                self.assertEqual(set(index["weight_map"]), set(tensors))
                for name, tensor in native.items():
                    self.assertTrue(torch.equal(tensors[name], tensor), name)
                    self.assertEqual(tensors[name].dtype, tensor.dtype)
                self.assertEqual(len(set(tensors) - set(native)), 129)
                self.assertEqual(
                    index["metadata"]["total_size"],
                    sum(t.numel() * t.element_size() for t in tensors.values()),
                )
                self.assertEqual(tensors["model.visual.router_cross_attn.query"].dtype, dtype)
                self.assertEqual(tensors["model.visual.mask_net.1.weight"].dtype, torch.float32)
                served = prepare(out, root / "serve")
                prepared_config = json.loads((served / "config.json").read_text())
                self.assertEqual(prepared_config["visionweave"], config["visionweave"])
                self.assertEqual(read_checkpoint(served)[0], index)

    def test_loaded_modules_start_with_normalized_pooling_and_explicit_routing_probability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "native"
            make_native(source)
            for probability in (0.1, 0.8):
                out = initialize_checkpoint(
                    source, root / str(probability), initial_compression_prob=probability
                )
                _, tensors = read_checkpoint(out)

                def load(module, prefix):
                    prefix = "model.visual." + prefix + "."
                    module.load_state_dict(
                        {
                            name[len(prefix) :]: value
                            for name, value in tensors.items()
                            if name.startswith(prefix)
                        },
                        strict=True,
                    )
                    return module

                compressor = load(VisionWeaveCompressionProjector(8, 2, 2), "compression_projector")
                blocks = torch.randn(3, 4, 8)
                torch.testing.assert_close(
                    compressor(blocks),
                    torch.nn.functional.layer_norm(blocks, (8,), eps=1e-6).mean(1),
                )
                router = load(VisionWeaveHierCrossAttnRouter(8, num_heads=2), "router_cross_attn")
                mask = load(build_mask_net(8), "mask_net")
                grids = [(1, 4, 8)]
                query, context = router.init_state(grids, "cpu", torch.float32)
                for layer in range(6):
                    query = router.run_layer(query, layer, torch.randn(32, 8), grids, context)
                probabilities = router_probabilities(
                    mask,
                    tensors["model.visual.last_layer_bias"],
                    tensors["model.visual.router_bias"],
                    query,
                )
                torch.testing.assert_close(probabilities, torch.full((2,), probability))
                layout = hard_route_layout(grids[0], (probabilities >= 0.5).detach().numpy())
                self.assertEqual(len(layout.gather_index), 8 if probability < 0.5 else 2)

    def test_seed_is_reproducible_and_does_not_change_callers_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "native"
            make_native(source)
            rng_state = torch.get_rng_state().clone()
            outputs = [
                initialize_checkpoint(source, root / f"init-{i}", seed=seed)
                for i, seed in enumerate((7, 7, 8))
            ]
            self.assertTrue(torch.equal(rng_state, torch.get_rng_state()))
            first, second, third = [read_checkpoint(out)[1] for out in outputs]
            self.assertTrue(all(torch.equal(value, second[name]) for name, value in first.items()))
            key = "model.visual.router_cross_attn.layers.0.k_proj.weight"
            self.assertFalse(torch.equal(first[key], third[key]))

    def test_link_mode_links_only_unchanged_native_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "native"
            _, native = make_native(source)
            out = initialize_checkpoint(source, root / "init", link_base_weights=True)
            index, _ = read_checkpoint(out)
            for name, shard in index["weight_map"].items():
                self.assertEqual((out / shard).is_symlink(), name in native)
            self.assertFalse((out / "tokenizer.json").is_symlink())

    def test_invalid_options_and_destinations_leave_source_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "native"
            make_native(source)
            before = hashes(source)
            for kwargs in (
                {"seed": -1},
                {"initial_compression_prob": 0},
                {"initial_compression_prob": 1},
                {"initial_compression_prob": float("nan")},
                {"route_threshold": -0.1},
                {"route_threshold": float("inf")},
            ):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    initialize_checkpoint(source, root / "invalid", **kwargs)
                self.assertFalse((root / "invalid").exists())
            for target in (source, source / "nested"):
                with self.assertRaises(ValueError):
                    initialize_checkpoint(source, target)
            dangling = root / "dangling"
            dangling.symlink_to(root / "absent")
            with self.assertRaises(ValueError):
                initialize_checkpoint(source, dangling)
            self.assertTrue(dangling.is_symlink())
            self.assertEqual(before, hashes(source))

    def test_incompatible_sources_are_rejected_before_writing(self):
        for variant in (
            "moe",
            "quantized",
            "deepstack",
            "missing_shard",
            "bad_index",
            "bad_shape",
            "missing_processor",
            "already_initialized",
        ):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "native"
                config, _ = make_native(source)
                if variant == "moe":
                    config["model_type"] = "qwen3_5_moe"
                elif variant == "quantized":
                    config["quantization_config"] = {"quant_method": "fp8"}
                elif variant == "deepstack":
                    config["vision_config"]["deepstack_visual_indexes"] = [1]
                elif variant == "missing_shard":
                    (source / "part1.safetensors").unlink()
                elif variant == "bad_index":
                    path = source / "model.safetensors.index.json"
                    index = json.loads(path.read_text())
                    index["weight_map"]["nonexistent.weight"] = "part1.safetensors"
                    path.write_text(json.dumps(index))
                elif variant == "bad_shape":
                    config["text_config"]["vocab_size"] = 33
                elif variant == "missing_processor":
                    (source / "video_preprocessor_config.json").unlink()
                elif variant == "already_initialized":
                    config["vision_config"]["router_cross_attn_depth"] = 6
                (source / "config.json").write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    initialize_checkpoint(source, root / "invalid")
                self.assertFalse((root / "invalid").exists())

    def test_failed_write_removes_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "native"
            make_native(source)
            before = hashes(source)
            with patch(
                "scripts.init_visionweave_checkpoint.save_file", side_effect=OSError("disk failure")
            ):
                with self.assertRaisesRegex(OSError, "disk failure"):
                    initialize_checkpoint(source, root / "init")
            self.assertEqual(before, hashes(source))
            self.assertEqual(list(root.iterdir()), [source])


if __name__ == "__main__":
    unittest.main()
