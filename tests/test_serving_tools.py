import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from visionweave import ARCHITECTURE, ROUTE_THRESHOLD_ENV
from visionweave.prepare import prepare
from visionweave.serve import Role, build_roles, parser, run, stop_children


def checkpoint(root):
    root.mkdir()
    vision = {
        "spatial_merge_size": 2,
        "compression_spatial_merge_size": 2,
        "effective_compression_spatial_merge_size": 1,
        "patch_size": 16,
        "router_cross_attn_depth": 6,
        "router_sample_depths": [0, 12, 24],
        "router_cross_attn_use_kv_norm": True,
        "depth": 24,
    }
    config = {
        "architectures": [ARCHITECTURE],
        "model_type": "qwen3_5",
        "vision_config": vision,
        "auto_map": {"AutoModel": "private.Model"},
    }
    (root / "config.json").write_text(json.dumps(config))
    (root / "args.json").write_text('{"router_hard_threshold":0.5,"private_setting":true}')
    (root / "trainer_state.json").write_text('{"step":100}')
    (root / "private_script.py").write_text("raise RuntimeError\n")
    for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
        (root / name).write_text('{"patch_size":16,"merge_size":4,"size":{}}')
    (root / "processor_config.json").write_text('{"image_processor":{"merge_size":4}}')
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (root / name).write_text("{}")
    (root / "model.safetensors").write_bytes(b"synthetic weight fixture")
    return root


class PrepareTests(unittest.TestCase):
    def test_preparation_preserves_source_and_exports_only_serving_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = checkpoint(root / "checkpoint")
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            out = prepare(source, root / "serve")
            self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})
            self.assertEqual(
                set(p.name for p in out.iterdir()),
                {
                    "config.json",
                    "processor_config.json",
                    "preprocessor_config.json",
                    "video_preprocessor_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "model.safetensors",
                },
            )
            self.assertTrue((out / "model.safetensors").is_symlink())
            config = json.loads((out / "config.json").read_text())
            self.assertEqual(config["architectures"], [ARCHITECTURE])
            self.assertEqual(config["visionweave"]["route_threshold"], 0.5)
            self.assertNotIn("auto_map", config)
            self.assertFalse((out / "config.json").is_symlink())
            for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
                self.assertEqual(json.loads((out / name).read_text())["merge_size"], 2)
            self.assertNotIn(
                "image_processor", json.loads((out / "processor_config.json").read_text())
            )

    def test_existing_or_nested_destination_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = checkpoint(Path(tmp) / "checkpoint")
            for destination in (source, source / "serve"):
                with self.assertRaises(ValueError):
                    prepare(source, destination)

    def test_missing_threshold_requires_explicit_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = checkpoint(root / "checkpoint")
            (source / "args.json").unlink()
            with self.assertRaisesRegex(ValueError, "threshold"):
                prepare(source, root / "serve")
            self.assertFalse((root / "serve").exists())
            prepare(source, root / "serve", threshold=0)

    def test_incompatible_geometry_and_missing_shards_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = checkpoint(root / "checkpoint")
            config = json.loads((source / "config.json").read_text())
            config["vision_config"]["effective_compression_spatial_merge_size"] = 2
            (source / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "effective_compression"):
                prepare(source, root / "serve")
            config["vision_config"]["effective_compression_spatial_merge_size"] = 1
            (source / "config.json").write_text(json.dumps(config))
            (source / "model.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "missing weight"):
                prepare(source, root / "serve")


class LauncherTests(unittest.TestCase):
    def test_roles_share_threshold_and_route_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = prepare(checkpoint(root / "checkpoint"), root / "serve")
            args = parser().parse_args(
                [
                    "--model-path",
                    str(model),
                    "--gpus",
                    "1",
                    "--encoder-gpus",
                    "0",
                    "--route-threshold",
                    "1.1",
                ]
            )
            with patch.dict(os.environ, {"SGLANG_EXTERNAL_MODEL_PACKAGE": "stale.models"}):
                encoder, language = build_roles(args)
            for role in (encoder, language):
                self.assertEqual(role.environment[ROUTE_THRESHOLD_ENV], "1.1")
                self.assertEqual(
                    role.environment["SGLANG_EXTERNAL_MODEL_PACKAGE"], "visionweave.models"
                )
                self.assertEqual(
                    role.command[role.command.index("--encoder-transfer-backend") + 1],
                    "zmq_to_tokenizer",
                )
            self.assertEqual(encoder.environment["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(language.environment["CUDA_VISIBLE_DEVICES"], "1")
            enc_budget = json.loads(
                encoder.command[encoder.command.index("--mm-process-config") + 1]
            )
            lang_budget = json.loads(
                language.command[language.command.index("--mm-process-config") + 1]
            )
            self.assertEqual(enc_budget["vision_config"], lang_budget)
            args.encoder_port = args.port
            with self.assertRaisesRegex(ValueError, "ports"):
                build_roles(args)

    def test_baselines_use_native_registration_and_single_prefill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = prepare(checkpoint(root / "checkpoint"), root / "serve")
            config = json.loads((model / "config.json").read_text())
            config["architectures"] = ["Qwen3_5ForConditionalGeneration"]
            (model / "config.json").write_text(json.dumps(config))
            for method in ("fastv", "visionzip"):
                args = parser().parse_args(["--model-path", str(model), "--method", method])
                (role,) = build_roles(args)
                self.assertNotIn("SGLANG_EXTERNAL_MM_MODEL_ARCH", role.environment)
                for flag in ("--disable-radix-cache", "--disable-overlap-schedule"):
                    self.assertIn(flag, role.command)
                self.assertEqual(
                    role.command[role.command.index("--max-running-requests") + 1], "1"
                )
                override = json.loads(
                    role.command[role.command.index("--json-model-override-args") + 1]
                )
                self.assertEqual(override["post_vit_compression"]["method"], method)

    def test_occupied_port_fails_before_spawning(self):
        with socket.socket() as listener, tempfile.TemporaryDirectory() as tmp:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            role = Role("test", [], {}, "127.0.0.1", listener.getsockname()[1])
            with patch("subprocess.Popen") as spawn:
                with self.assertRaises(OSError):
                    run([role], Path(tmp), 1)
                spawn.assert_not_called()

    def test_cleanup_terminates_owned_process_group(self):
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
        )
        try:
            stop_children([child])
            self.assertIsNotNone(child.poll())
            with self.assertRaises(ProcessLookupError):
                os.killpg(child.pid, 0)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()


if __name__ == "__main__":
    unittest.main()
