"""Reserve final placeholder lengths while retaining native sparse M-RoPE."""

import torch

FRAMES = "post_vit_frames"
POSITIONS = "post_vit_native_positions"


def compress_layout(output, config, merge_size=2):
    if output is None or config.keep_ratio == 1 or (not output.mm_items):
        return output
    ids = output.input_ids
    positions = output.mrope_positions
    if positions is None or positions.shape != (3, len(ids)):
        raise ValueError("post-ViT requires native (3, L) M-RoPE positions")
    segments = []
    for index, item in enumerate(output.mm_items):
        if not (item.is_image() or item.is_video()) or item.precomputed_embeddings is not None:
            raise ValueError("post-ViT accepts native raw image/video items only")
        grid = item.image_grid_thw if item.is_image() else item.video_grid_thw
        rows = torch.as_tensor(grid).reshape(-1, 3).tolist()
        if len(rows) != 1:
            raise ValueError("expected one native grid per multimodal item")
        t, h, w = map(int, rows[0])
        if min(t, h, w) <= 0 or h % merge_size or w % merge_size:
            raise ValueError("invalid native visual grid")
        n = h * w // merge_size**2
        if sum((b - a + 1 for a, b in item.offsets)) != t * n:
            raise ValueError("native offsets/grid mismatch")
        item.model_specific_data[FRAMES] = []
        item.model_specific_data[POSITIONS] = torch.cat(
            [positions[:, a : b + 1] for a, b in item.offsets], dim=1
        ).clone()
        feature_start = 0
        for run_index, (a, b) in enumerate(item.offsets):
            if (b - a + 1) % n:
                raise ValueError("visual run cuts across a temporal frame")
            for start in range(a, b + 1, n):
                segments.append((start, n, index, run_index, feature_start))
                feature_start += n
    selected, cursor = ([], 0)
    offsets = [{} for _ in output.mm_items]
    for start, n, index, run_index, feature_start in sorted(segments):
        if start < cursor or start + n > len(ids):
            raise ValueError("overlapping or out-of-range multimodal offsets")
        selected.extend(range(cursor, start))
        new_start, k = (len(selected), config.count(n))
        selected.extend(range(start, start + k))
        cursor = start + n
        item = output.mm_items[index]
        item.model_specific_data[FRAMES].append([feature_start, n, k, new_start])
        old = offsets[index].get(run_index, (new_start, new_start))
        offsets[index][run_index] = (old[0], new_start + k - 1)
    selected.extend(range(cursor, len(ids)))
    for index, item in enumerate(output.mm_items):
        item.offsets = [offsets[index][j] for j in sorted(offsets[index])]
    output.input_ids = [ids[i] for i in selected]
    if output.padded_input_ids is not None:
        output.padded_input_ids = [output.padded_input_ids[i] for i in selected]
    if output.token_type_ids is not None:
        value = output.token_type_ids
        output.token_type_ids = (
            value[..., selected] if torch.is_tensor(value) else [value[i] for i in selected]
        )
    output.mrope_positions = positions[:, selected].clone()
    output.mrope_position_delta = output.mrope_position_delta + len(ids) - len(selected)
    return output


def fill_positions(mm_input, item, selections):
    frames = item.model_specific_data[FRAMES]
    native = item.model_specific_data[POSITIONS]
    if len(selections) != len(frames):
        raise ValueError("frame selection count mismatch")
    for (offset, n, k, start), chosen in zip(frames, selections):
        chosen = chosen.to(native.device)
        if chosen.numel() != k or chosen.min() < 0 or chosen.max() >= n:
            raise ValueError("invalid retained indices")
        mm_input.mrope_positions[:, start : start + k] = native[:, offset + chosen]
