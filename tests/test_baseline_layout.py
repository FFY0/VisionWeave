import unittest
from types import SimpleNamespace

import torch
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.managers.mm_utils import embed_mm_inputs
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from visionweave.baselines.config import CompressionConfig
from visionweave.baselines.layout import FRAMES, compress_layout, fill_positions


def make_output(media):
    ids, items, images, videos = ([1, 2], [], [], [])
    for group, (kind, t, h, w) in enumerate(media):
        n = h * w // 4
        offsets = []
        for frame in range(t if kind == "video" else 1):
            ids.extend([20 + frame, 102])
            a = len(ids)
            ids.extend([100 if kind == "image" else 101] * (n * t if kind == "image" else n))
            offsets.append((a, len(ids) - 1))
            ids.extend([103, 7])
        grid = torch.tensor([[t, h, w]])
        item = MultimodalDataItem(
            modality=Modality.IMAGE if kind == "image" else Modality.VIDEO,
            offsets=offsets,
            hash=1000 + group,
            pad_value=1000 + group,
            feature=torch.zeros(t * h * w, 4),
        )
        item.model_specific_data[f"{kind}_grid_thw"] = grid
        items.append(item)
        (images if kind == "image" else videos).append([t, h, w])
    ids.extend([8, 9])
    positions, delta = MRotaryEmbedding.get_rope_index(
        spatial_merge_size=2,
        image_token_id=100,
        video_token_id=101,
        vision_start_token_id=102,
        model_type="qwen3_5",
        input_ids=torch.tensor([ids]),
        image_grid_thw=torch.tensor(images) if images else None,
        video_grid_thw=torch.tensor(videos) if videos else None,
    )
    return MultimodalProcessorOutput(
        mm_items=items,
        input_ids=ids,
        padded_input_ids=MultimodalProcessorOutput.build_padded_input_ids(ids, items),
        mrope_positions=positions.squeeze(1),
        mrope_position_delta=delta,
        token_type_ids=torch.tensor([int(i in (100, 101)) for i in ids]),
    )


class LayoutTest(unittest.TestCase):
    def test_native_injection_and_decode_positions(self):
        cases = [
            [],
            [("image", 1, 6, 6)],
            [("image", 2, 4, 6)],
            [("image", 1, 4, 6), ("image", 1, 6, 6)],
            [("video", 3, 6, 6)],
            [("video", 2, 4, 6), ("video", 3, 6, 6)],
            [("image", 1, 4, 6), ("video", 2, 6, 6)],
        ]
        for media in cases:
            for ratio in (1.0, 0.5, 0.25):
                with self.subTest(media=media, ratio=ratio):
                    out = make_output(media)
                    native_ids = list(out.input_ids)
                    native_positions = out.mrope_positions.clone()
                    native_delta = out.mrope_position_delta.clone()
                    original_items = len(out.mm_items)
                    cfg = CompressionConfig("fastv", ratio)
                    compress_layout(out, cfg)
                    self.assertEqual(len(out.mm_items), original_items)
                    if ratio == 1:
                        self.assertEqual(out.input_ids, native_ids)
                        torch.testing.assert_close(
                            out.mrope_positions, native_positions, rtol=0, atol=0
                        )
                        self.assertTrue(
                            all((FRAMES not in i.model_specific_data for i in out.mm_items))
                        )
                        continue
                    mm = MultimodalInputs(
                        mm_items=out.mm_items,
                        mrope_positions=out.mrope_positions,
                        mrope_position_delta=out.mrope_position_delta,
                    )
                    for item in mm.mm_items:
                        selected = [
                            torch.arange(n - k, n)
                            for _, n, k, _ in item.model_specific_data[FRAMES]
                        ]
                        fill_positions(mm, item, selected)
                        parts = [
                            torch.arange(k).float()[:, None].expand(k, 8)
                            for _, _, k, _ in item.model_specific_data[FRAMES]
                        ]
                        item.precomputed_embeddings = torch.cat(parts)
                        item.feature = None
                    self.assertEqual(
                        [i for i in out.input_ids if i not in (100, 101)],
                        [i for i in native_ids if i not in (100, 101)],
                    )
                    new_text = [i for i, v in enumerate(out.input_ids) if v not in (100, 101)]
                    old_text = [i for i, v in enumerate(native_ids) if v not in (100, 101)]
                    torch.testing.assert_close(
                        out.mrope_positions[:, new_text],
                        native_positions[:, old_text],
                        rtol=0,
                        atol=0,
                    )
                    if out.mm_items:
                        self.assertEqual(len(out.padded_input_ids), len(out.input_ids))
                    self.assertEqual(out.token_type_ids.numel(), len(out.input_ids))
                    original = MultimodalInputs(
                        mm_items=[],
                        mrope_positions=native_positions,
                        mrope_position_delta=native_delta,
                    )
                    for step in range(1, 6):
                        a = ForwardBatch._expand_mrope_from_input(
                            None, mm, len(out.input_ids) + step
                        )
                        b = ForwardBatch._expand_mrope_from_input(
                            None, original, len(native_ids) + step
                        )
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    if mm.mm_items:

                        def unexpected(items):
                            self.fail("precomputed features must bypass native ViT")

                        embedded, _ = embed_mm_inputs(
                            mm_inputs_list=[mm],
                            extend_prefix_lens=[0],
                            extend_seq_lens=[len(out.input_ids)],
                            input_ids=torch.tensor(out.padded_input_ids),
                            input_embedding=torch.nn.Embedding(512, 8),
                            multimodal_model=SimpleNamespace(
                                get_image_feature=unexpected, get_video_feature=unexpected
                            ),
                        )
                        for item in mm.mm_items:
                            actual = torch.cat([embedded[a : b + 1] for a, b in item.offsets])
                            torch.testing.assert_close(
                                actual, item.precomputed_embeddings, rtol=0, atol=0
                            )

    def test_invalid_native_layout_fails(self):
        out = make_output([("video", 2, 4, 6)])
        out.mm_items[0].offsets[0] = (4, 4)
        with self.assertRaises(ValueError):
            compress_layout(out, CompressionConfig("visionzip"))


if __name__ == "__main__":
    unittest.main()
