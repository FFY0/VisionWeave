"""Local/global cross-attention router with checkpoint-compatible parameter names."""

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "ROUTER_DEPTH",
    "VisionWeaveGlobalCrossAttnRouterLayer",
    "VisionWeaveHierCrossAttnRouter",
    "VisionWeaveLocalCrossAttnRouterLayer",
    "apply_cross_rope",
    "build_mask_net",
    "rotate_half",
    "router_probabilities",
]
ROUTER_DEPTH = 6


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    half = hidden_states.shape[-1] // 2
    return torch.cat((-hidden_states[..., half:], hidden_states[..., :half]), dim=-1)


def apply_cross_rope(query, key, query_cos, query_sin, key_cos, key_sin):
    """Rope with SEPARATE tables for q and k -- they live on different grids."""
    query_dtype, key_dtype = (query.dtype, key.dtype)
    query = query.float()
    key = key.float()
    query_cos = query_cos.unsqueeze(-2).float()
    query_sin = query_sin.unsqueeze(-2).float()
    key_cos = key_cos.unsqueeze(-2).float()
    key_sin = key_sin.unsqueeze(-2).float()
    query = query * query_cos + rotate_half(query) * query_sin
    key = key * key_cos + rotate_half(key) * key_sin
    return (query.to(query_dtype), key.to(key_dtype))


class VisionWeaveLocalCrossAttnRouterLayer(nn.Module):
    """One query attends to the 16 patches of its own 4x4 block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 2, use_kv_norm: bool = True):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-06) if use_kv_norm else nn.Identity()
        self.norm_attn = nn.LayerNorm(dim, eps=1e-06)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )
        self.norm_mlp = nn.LayerNorm(dim, eps=1e-06)

    def forward(self, query, key_value, query_rope, key_rope):
        num_blocks = query.shape[0]
        key_value = self.norm_kv(key_value)
        q = self.q_proj(query).reshape(num_blocks, self.num_heads, self.head_dim)
        k = self.k_proj(key_value).reshape(num_blocks, 16, self.num_heads, self.head_dim)
        v = self.v_proj(key_value).reshape(num_blocks, 16, self.num_heads, self.head_dim)
        q, k = apply_cross_rope(q, k, *query_rope, *key_rope)
        attention = F.scaled_dot_product_attention(
            q.unsqueeze(2), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), is_causal=False
        )
        attention = self.out_proj(attention.squeeze(2).reshape(num_blocks, -1))
        query = self.norm_attn(query + attention)
        return self.norm_mlp(query + self.mlp(query))


class VisionWeaveGlobalCrossAttnRouterLayer(nn.Module):
    """Every query in a frame attends to every patch in that frame."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 2, use_kv_norm: bool = True):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-06) if use_kv_norm else nn.Identity()
        self.norm_attn = nn.LayerNorm(dim, eps=1e-06)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )
        self.norm_mlp = nn.LayerNorm(dim, eps=1e-06)

    def forward(self, query, key_value, query_rope, key_rope, cu_seqlens_query, cu_seqlens_key):
        key_value = self.norm_kv(key_value)
        q = self.q_proj(query).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(key_value).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(key_value).reshape(-1, self.num_heads, self.head_dim)
        q, k = apply_cross_rope(q, k, *query_rope, *key_rope)
        outputs = []
        for query_start, query_end, key_start, key_end in zip(
            cu_seqlens_query[:-1].tolist(),
            cu_seqlens_query[1:].tolist(),
            cu_seqlens_key[:-1].tolist(),
            cu_seqlens_key[1:].tolist(),
        ):
            frame_q = q[query_start:query_end].transpose(0, 1).unsqueeze(0)
            frame_k = k[key_start:key_end].transpose(0, 1).unsqueeze(0)
            frame_v = v[key_start:key_end].transpose(0, 1).unsqueeze(0)
            frame_output = F.scaled_dot_product_attention(
                frame_q, frame_k, frame_v, is_causal=False
            )
            outputs.append(frame_output.squeeze(0).transpose(0, 1))
        attention = self.out_proj(torch.cat(outputs).reshape(query.shape[0], -1))
        query = self.norm_attn(query + attention)
        return self.norm_mlp(query + self.mlp(query))


class VisionWeaveHierCrossAttnRouter(nn.Module):
    """Six alternating Local/Global cross-attention layers over three ViT depths."""

    def __init__(
        self,
        hidden_size: int,
        depth: int = ROUTER_DEPTH,
        num_heads: int = 16,
        mlp_ratio: int = 2,
        use_kv_norm: bool = True,
    ):
        super().__init__()
        if depth != ROUTER_DEPTH:
            raise ValueError(f"VisionWeave router depth must be {ROUTER_DEPTH}, got {depth}.")
        if hidden_size % num_heads:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})."
            )
        head_dim = hidden_size // num_heads
        if head_dim % 4:
            raise ValueError(
                f"router head dimension ({head_dim}) must be divisible by 4 for 2D rope."
            )
        if not use_kv_norm:
            raise ValueError("VisionWeave router requires KV LayerNorm.")
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.query = nn.Parameter(torch.zeros(1, hidden_size))
        self.layers = nn.ModuleList(
            (
                (
                    VisionWeaveLocalCrossAttnRouterLayer
                    if layer_index % 2 == 0
                    else VisionWeaveGlobalCrossAttnRouterLayer
                )(hidden_size, num_heads, mlp_ratio, use_kv_norm=True)
                for layer_index in range(depth)
            )
        )
        inv_freq = 1.0 / 10000.0 ** (
            torch.arange(0, head_dim // 2, 2, dtype=torch.float32) / (head_dim // 2)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _validate_grid(grid_thw_list) -> None:
        for temporal, height, width in grid_thw_list:
            temporal, height, width = (int(temporal), int(height), int(width))
            if temporal <= 0 or height <= 0 or width <= 0:
                raise ValueError(
                    f"grid dimensions must be positive, got {(temporal, height, width)}."
                )
            if height % 4 or width % 4:
                raise ValueError(
                    f"VisionWeave requires grid h and w divisible by 4, got h={height}, w={width}. The image processor should have resized on factor patch_size*4=64."
                )

    def _positions(self, grid_thw_list, device):
        """Absolute 2-D positions for the global layers, plus the per-FRAME cu_seqlens."""
        query_positions = []
        key_positions = []
        query_lengths = []
        key_lengths = []
        for temporal, height, width in grid_thw_list:
            temporal, height, width = (int(temporal), int(height), int(width))
            block_rows = torch.arange(height // 4, dtype=torch.float32, device=device) * 4 + 1.5
            block_cols = torch.arange(width // 4, dtype=torch.float32, device=device) * 4 + 1.5
            query_frame = torch.stack(
                torch.meshgrid(block_rows, block_cols, indexing="ij"), dim=-1
            ).reshape(-1, 2)
            patch_rows = torch.arange(height, dtype=torch.float32, device=device)
            patch_cols = torch.arange(width, dtype=torch.float32, device=device)
            key_frame = torch.stack(
                torch.meshgrid(patch_rows, patch_cols, indexing="ij"), dim=-1
            ).reshape(-1, 2)
            for _ in range(temporal):
                query_positions.append(query_frame)
                key_positions.append(key_frame)
                query_lengths.append(query_frame.shape[0])
                key_lengths.append(key_frame.shape[0])
        cu_query = F.pad(
            torch.tensor(query_lengths, dtype=torch.int32, device=device).cumsum(
                0, dtype=torch.int32
            ),
            (1, 0),
        )
        cu_key = F.pad(
            torch.tensor(key_lengths, dtype=torch.int32, device=device).cumsum(
                0, dtype=torch.int32
            ),
            (1, 0),
        )
        return (torch.cat(query_positions), torch.cat(key_positions), cu_query, cu_key)

    @staticmethod
    def _local_positions(device):
        """The 4x4 grid every Local layer sees, and the block centre in those coordinates."""
        query = torch.tensor([[1.5, 1.5]], dtype=torch.float32, device=device)
        coords = torch.arange(4, dtype=torch.float32, device=device)
        key = torch.stack(torch.meshgrid(coords, coords, indexing="ij"), dim=-1).reshape(16, 2)
        return (query, key)

    @staticmethod
    def _local_key_value(raster_states: torch.Tensor, grid_thw_list) -> torch.Tensor:
        """`(tokens, hidden)` in RASTER order -> `(blocks, 16, hidden)` in block-raster order."""
        chunks = []
        offset = 0
        hidden_size = raster_states.shape[-1]
        for temporal, height, width in grid_thw_list:
            temporal, height, width = (int(temporal), int(height), int(width))
            token_count = temporal * height * width
            chunk = raster_states[offset : offset + token_count].view(
                temporal, height, width, hidden_size
            )
            chunk = (
                chunk.view(temporal, height // 4, 4, width // 4, 4, hidden_size)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(-1, 16, hidden_size)
            )
            chunks.append(chunk)
            offset += token_count
        return torch.cat(chunks) if len(chunks) > 1 else chunks[0]

    def _rope(self, positions: torch.Tensor, dtype: torch.dtype):
        embedding = (positions[..., None] * self.inv_freq.to(positions.device)).flatten(1)
        embedding = torch.cat((embedding, embedding), dim=-1)
        return (embedding.cos().to(dtype), embedding.sin().to(dtype))

    def init_state(self, grid_thw_list, device, dtype):
        """`(query, context)` -- one query row per block, plus the four rope tables."""
        self._validate_grid(grid_thw_list)
        query_positions, key_positions, cu_query, cu_key = self._positions(grid_thw_list, device)
        local_query_positions, local_key_positions = self._local_positions(device)
        context = {
            "global_query_rope": self._rope(query_positions, dtype),
            "global_key_rope": self._rope(key_positions, dtype),
            "local_query_rope": self._rope(local_query_positions, dtype),
            "local_key_rope": self._rope(local_key_positions, dtype),
            "cu_query": cu_query,
            "cu_key": cu_key,
        }
        return (self.query.expand(query_positions.shape[0], -1).to(dtype), context)

    def run_layer(self, query, layer_index: int, raster_states, grid_thw_list, context):
        layer = self.layers[layer_index]
        if isinstance(layer, VisionWeaveLocalCrossAttnRouterLayer):
            key_value = self._local_key_value(raster_states, grid_thw_list)
            return layer(query, key_value, context["local_query_rope"], context["local_key_rope"])
        return layer(
            query,
            raster_states,
            context["global_query_rope"],
            context["global_key_rope"],
            context["cu_query"],
            context["cu_key"],
        )


def router_probabilities(
    mask_net: nn.Module,
    last_layer_bias: torch.Tensor,
    router_bias: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    """Router query -> P(compress) per block, fp32."""
    if torch.count_nonzero(router_bias).item():
        raise RuntimeError(
            f"VisionWeave router_bias must be exactly zero, got {router_bias.tolist()}."
        )
    logits = mask_net(query.float()) + last_layer_bias
    logits = logits + torch.cat((torch.zeros_like(router_bias), router_bias)).unsqueeze(0)
    return torch.softmax(logits.float(), dim=-1)[:, 1]


def build_mask_net(hidden_size: int) -> nn.Sequential:
    """`LayerNorm(h)` then `Linear(h -> 2, bias=False)`, matching the three checkpoint tensors."""
    return nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 2, bias=False))
