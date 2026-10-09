"""Frame-local selection; features are native spatial-merger outputs."""

import torch
import torch.nn.functional as F


def reduce_frame(features, scores, keys, config):
    n = features.shape[0]
    k = config.count(n)
    if scores.shape != (n,) or keys.shape[0] != n:
        raise ValueError("feature/statistics length mismatch")
    if not torch.isfinite(scores).all() or not torch.isfinite(keys).all():
        raise ValueError("nonfinite visual statistics")
    if k == n:
        return (features, torch.arange(n, device=features.device))
    ranked = torch.argsort(scores, descending=True, stable=True)
    if config.method == "fastv":
        selected = ranked[:k].sort().values
        return (features[selected], selected)
    contextual = min(k - 1, max(1, round(k * config.contextual_frac))) if k >= 2 else 0
    dominant = ranked[: k - contextual]
    if contextual == 0:
        selected = dominant.sort().values
        return (features[selected], selected)
    mask = torch.ones(n, dtype=torch.bool, device=features.device)
    mask[dominant] = False
    remaining = mask.nonzero(as_tuple=True)[0]
    targets = remaining[:: max(1, len(remaining) // contextual)][:contextual]
    mask[targets] = False
    sources = mask.nonzero(as_tuple=True)[0]
    contextual_features = features[targets].clone()
    if len(sources):
        normalized = F.normalize(keys.float(), dim=-1)
        assignment = (normalized[sources] @ normalized[targets].T).argmax(dim=-1)
        for j in range(contextual):
            members = sources[assignment == j]
            if len(members):
                contextual_features[j] = (
                    features[targets[j]].float() + features[members].float().mean(0)
                ).to(features.dtype)
    selected = torch.cat((dominant, targets))
    order = selected.argsort()
    return (torch.cat((features[dominant], contextual_features))[order], selected[order])
