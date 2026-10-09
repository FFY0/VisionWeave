"""Observe the last native ViT attention without replacing its forward."""

from contextlib import contextmanager

import torch
import torch.nn.functional as F


def head_sums(q, k, cu_seqlens, scale, chunk_size):
    """Return received-attention and RoPE-key sums over LOCAL heads (FP32)."""
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    scores = torch.zeros(q.shape[0], device=q.device, dtype=dtype)
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        qs = q[start:end].transpose(0, 1).to(dtype)
        ks = k[start:end].transpose(0, 1).to(dtype)
        for query in qs.split(chunk_size, dim=1):
            attention = torch.softmax(query @ ks.transpose(-1, -2) * scale, dim=-1)
            scores[start:end] += attention.sum(dim=1).sum(dim=0)
    return (scores, k.to(dtype).sum(dim=1))


def merged_statistics(q, k, cu_seqlens, scale, chunk_size, merge_size, group=None):
    cuts = [int(x) for x in cu_seqlens]
    if cuts[0] != 0 or cuts[-1] != q.shape[0] or q.shape != k.shape:
        raise ValueError("invalid Q/K segments")
    unit = merge_size**2
    if any((b <= a or (b - a) % unit for a, b in zip(cuts[:-1], cuts[1:]))):
        raise ValueError("frame is not divisible into merger groups")
    scores, keys = head_sums(q, k, cuts, scale, chunk_size)
    world = 1 if group is None else group.world_size
    if world > 1:
        torch.distributed.all_reduce(scores, group=group.device_group)
        torch.distributed.all_reduce(keys, group=group.device_group)
    heads = q.shape[1] * world
    scores = (scores / heads).reshape(-1, unit).mean(dim=1)
    keys = (keys / heads).reshape(-1, unit, keys.shape[-1]).mean(dim=1)
    return (scores, F.normalize(keys, dim=-1))


@contextmanager
def observe_last_attention(visual, config):
    from sglang.srt.distributed.parallel_state import get_attn_tp_group
    from sglang.srt.layers.rotary_embedding.utils import apply_rotary_pos_emb_native_eager

    captured = []
    attention = visual.blocks[-1].attn
    if (
        not attention.use_qkv_parallel
        or attention.qk_normalization
        or attention.qk_normalization_by_head_size
        or (attention.customized_position_embedding_applier is not None)
        or (attention.q_size != attention.kv_size)
    ):
        raise ValueError("unsupported ViT attention configuration")

    def hook(module, args, kwargs):
        if captured:
            raise RuntimeError("last ViT attention ran more than once")
        x = args[0]
        qkv, _ = module.qkv_proj(x)
        q, k, _ = qkv.split([module.q_size, module.kv_size, module.kv_size], dim=-1)
        heads = module.num_attention_heads_per_partition
        q = q.reshape(-1, heads, module.head_size)
        k = k.reshape(-1, heads, module.head_size)
        cos, sin = (kwargs["rotary_pos_emb_cos"], kwargs["rotary_pos_emb_sin"])
        if cos.shape[-1] * 2 == module.head_size:
            cos, sin = (torch.cat((cos, cos), -1), torch.cat((sin, sin), -1))
        q, k = apply_rotary_pos_emb_native_eager(q, k, cos, sin)
        captured.append(
            merged_statistics(
                q,
                k,
                kwargs["cu_seqlens"].tolist(),
                module.softmax_scale or module.head_size ** (-0.5),
                config.query_chunk_size,
                visual.spatial_merge_size,
                get_attn_tp_group() if module.tp_size > 1 else None,
            )
        )

    handle = attention.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield captured
        if len(captured) != 1:
            raise RuntimeError("post-ViT statistics hook did not execute")
    finally:
        handle.remove()
