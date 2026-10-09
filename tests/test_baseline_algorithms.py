import math
import unittest

import torch

from visionweave.baselines.config import CompressionConfig
from visionweave.baselines.reduction import reduce_frame
from visionweave.baselines.scoring import head_sums, merged_statistics


def reference(features, scores, keys, cfg):
    """Deliberately scalar/list-based mathematical reference, independent selection."""
    n, k = (len(features), cfg.count(len(features)))
    order = sorted(range(n), key=lambda i: (-float(scores[i]), i))
    if cfg.method == "fastv" or k == n:
        idx = sorted(order[:k]) if k != n else list(range(n))
        return (features[idx], idx)
    c = min(k - 1, max(1, round(k * cfg.contextual_frac))) if k >= 2 else 0
    if c == 0:
        idx = sorted(order[:k])
        return (features[idx], idx)
    dominant = order[: k - c]
    rest = [i for i in range(n) if i not in dominant]
    targets = rest[:: max(1, len(rest) // c)][:c] if c else []
    sources = [i for i in rest if i not in targets]
    normal = [row.double() / max(float(row.double().norm()), 1e-12) for row in keys]
    groups = {t: [] for t in targets}
    for s in sources:
        target = max(targets, key=lambda t: float(normal[s] @ normal[t]))
        groups[target].append(s)
    idx = sorted(dominant + targets)
    out = []
    for i in idx:
        value = features[i].clone()
        if i in groups and groups[i]:
            value += sum((features[j].float() for j in groups[i])) / len(groups[i])
        out.append(value)
    return (torch.stack(out), idx)


class AlgorithmsTest(unittest.TestCase):
    def test_reducers_against_reference(self):
        gen = torch.Generator().manual_seed(2718)
        for n in (1, 2, 3, 5, 9, 32):
            x = torch.randn(n, 7, generator=gen)
            keys = torch.randn(n, 6, generator=gen)
            for tied in (False, True):
                scores = torch.zeros(n) if tied else torch.randn(n, generator=gen)
                for ratio in (1.0, 0.75, 0.5, 0.1):
                    for method in ("visionzip", "fastv"):
                        for fraction in (0, 0.16, 1):
                            cfg = CompressionConfig(method, ratio, fraction)
                            with self.subTest(
                                n=n, ratio=ratio, method=method, frac=fraction, tied=tied
                            ):
                                actual, indices = reduce_frame(x, scores, keys, cfg)
                                expected, selection = reference(x, scores, keys, cfg)
                                self.assertEqual(indices.tolist(), selection)
                                torch.testing.assert_close(actual, expected)

    def test_contextual_is_target_plus_source_mean(self):
        x = torch.arange(1, 6, dtype=torch.float32)[:, None]
        y, idx = reduce_frame(
            x,
            torch.tensor([9.0, 1, 1, 1, 1]),
            torch.ones(5, 2),
            CompressionConfig("visionzip", 0.4),
        )
        self.assertEqual(idx.tolist(), [0, 1])
        torch.testing.assert_close(y[:, 0], torch.tensor([1.0, 6.0]))

    def test_segmented_attention_and_head_partition(self):
        gen = torch.Generator().manual_seed(17)
        q, k = [torch.randn(28, 4, 8, dtype=torch.float64, generator=gen) for _ in range(2)]
        expected = []
        for a, b in ((0, 12), (12, 28)):
            attn = torch.softmax(torch.einsum("qhd,khd->hqk", q[a:b], k[a:b]) / math.sqrt(8), -1)
            expected.append(attn.sum(1).mean(0))
        expected = torch.cat(expected).reshape(-1, 4).mean(1)
        for chunk in (1, 3, 128):
            scores, keys = merged_statistics(q, k, [0, 12, 28], 8 ** (-0.5), chunk, 2)
            torch.testing.assert_close(scores, expected, atol=1e-12, rtol=1e-12)
        full = head_sums(q, k, [0, 12, 28], 8 ** (-0.5), 3)
        parts = [
            head_sums(q[:, i : i + 2], k[:, i : i + 2], [0, 12, 28], 8 ** (-0.5), 3) for i in (0, 2)
        ]
        for i in (0, 1):
            torch.testing.assert_close(full[i], parts[0][i] + parts[1][i], atol=1e-12, rtol=1e-12)

    def test_invalid_and_rounding(self):
        for r in (0, -1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                CompressionConfig("visionzip", r)
        cfg = CompressionConfig("fastv", 0.5)
        self.assertEqual([cfg.count(n) for n in (1, 3, 5, 7, 9)], [1, 2, 2, 4, 4])


if __name__ == "__main__":
    unittest.main()
