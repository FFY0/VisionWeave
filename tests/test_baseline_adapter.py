import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch
from sglang.srt.managers.schedule_batch import MultimodalInputs
from test_baseline_layout import make_output

from visionweave.baselines.config import CompressionConfig
from visionweave.baselines.layout import compress_layout
from visionweave.baselines.models.qwen3_5 import NativeQwen, Qwen3_5ForConditionalGeneration


class AdapterTest(unittest.TestCase):
    def model(self, cfg):
        model = Qwen3_5ForConditionalGeneration.__new__(Qwen3_5ForConditionalGeneration)
        torch.nn.Module.__init__(model)
        model.post_vit_config = cfg
        model.visual = torch.nn.Identity()
        return model

    def batch(self, output):
        mm = MultimodalInputs(
            mm_items=output.mm_items,
            mrope_positions=output.mrope_positions,
            mrope_position_delta=output.mrope_position_delta,
        )
        return SimpleNamespace(
            mm_inputs=[mm],
            forward_mode=SimpleNamespace(is_decode=lambda: False),
            batch_size=1,
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[len(output.input_ids)],
            mrope_positions=output.mrope_positions.clone(),
        )

    def test_r1_calls_native_without_observation(self):
        model = self.model(CompressionConfig("visionzip", 1.0))
        out = make_output([("image", 1, 6, 6)])
        batch = self.batch(out)
        with (
            patch.object(NativeQwen, "forward", return_value="native") as parent,
            patch("visionweave.baselines.models.qwen3_5.observe_last_attention") as observe,
        ):
            self.assertEqual(model.forward(torch.tensor(out.input_ids), None, batch), "native")
            parent.assert_called_once()
            observe.assert_not_called()
        self.assertIsNone(out.mm_items[0].precomputed_embeddings)

    def test_complete_prefill_prepares_every_item_and_both_position_copies(self):
        cfg = CompressionConfig("fastv", 0.5)
        out = make_output([("image", 1, 6, 6), ("video", 2, 6, 6)])
        compress_layout(out, cfg)
        batch = self.batch(out)
        placeholder_positions = batch.mrope_positions.clone()
        model = self.model(cfg)
        calls = []
        captured = []

        @contextmanager
        def observation(*args):
            yield captured

        def encode(items):
            calls.append(items[0].modality)
            grid = items[0].image_grid_thw if items[0].is_image() else items[0].video_grid_thw
            n = int(grid.prod()) // 4
            captured[:] = [(torch.arange(n).float(), torch.ones(n, 4))]
            return torch.arange(n).float()[:, None].expand(n, 8)

        model.get_image_feature = model.get_video_feature = encode

        def parent(*args, **kwargs):
            for item in batch.mm_inputs[0].mm_items:
                self.assertIsNone(item.feature)
                self.assertEqual(
                    item.precomputed_embeddings.shape[0], sum((b - a + 1 for a, b in item.offsets))
                )
            torch.testing.assert_close(batch.mrope_positions, batch.mm_inputs[0].mrope_positions)
            self.assertFalse(torch.equal(batch.mrope_positions, placeholder_positions))
            return "prepared"

        with (
            patch("visionweave.baselines.models.qwen3_5.observe_last_attention", observation),
            patch.object(NativeQwen, "forward", side_effect=parent),
        ):
            self.assertEqual(model.forward(torch.tensor(out.input_ids), None, batch), "prepared")
        self.assertEqual(len(calls), 2)

    def test_prefix_reuse_is_rejected(self):
        cfg = CompressionConfig("fastv", 0.5)
        out = compress_layout(make_output([("image", 1, 6, 6)]), cfg)
        batch = self.batch(out)
        batch.extend_prefix_lens_cpu = [1]
        with self.assertRaisesRegex(ValueError, "prefix_length=0"):
            self.model(cfg).forward(torch.tensor(out.input_ids), None, batch)


if __name__ == "__main__":
    unittest.main()
