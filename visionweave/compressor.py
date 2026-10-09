"""Visual compression projector and patch ordering operations."""

import torch
from torch import nn

__all__ = ["VisionWeaveCompressionProjector", "compression_block_index"]


def compression_block_index(t: int, h: int, w: int, fine: int, pool: int, device) -> torch.Tensor:
    """Flat gather index turning native ViT tokens into merger-ordered `pool x pool` blocks."""
    m, p = (fine, pool)
    row = torch.arange(h, device=device)
    col = torch.arange(w, device=device)
    frame = (row // m * (w // m * m * m) + row % m * m).view(h, 1) + (
        col // m * (m * m) + col % m
    ).view(1, w)
    gh, gw = (h // p, w // p)
    index = (
        frame.view(gh, p, gw, p)
        .permute(0, 2, 1, 3)
        .reshape(gh // m, m, gw // m, m, p * p)
        .permute(0, 2, 1, 3, 4)
        .reshape(1, gh * gw, p * p)
    )
    frames = (torch.arange(t, device=device) * (h * w)).view(t, 1, 1)
    return (index + frames).reshape(-1)


class VisionWeaveCompressionProjector(nn.Module):
    """Softmax-gated pooling of `pool x pool` native tokens into one."""

    def __init__(self, hidden_size: int, fine: int, pool: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.fine = fine
        self.pool = pool
        self.spatial = pool * pool
        h, spatial = (hidden_size, self.spatial)
        self.norm_resid = nn.LayerNorm(h, eps=1e-06)
        self.gate_mlp = nn.Sequential(nn.Linear(h + h, 4 * h), nn.GELU(), nn.Linear(4 * h, h))
        self.global_map = nn.Sequential(nn.Linear(spatial * h, h), nn.GELU(), nn.Linear(h, h))
        self.position_bias = nn.Parameter(torch.zeros(1, 1, 1, spatial, h))
        self.trans_mlp = nn.Sequential(nn.Linear(h + h, 4 * h), nn.GELU(), nn.Linear(4 * h, h))

    @torch.no_grad()
    def apply_transparent_init(self) -> None:
        """Zero the output projections so the module starts as plain mean pooling."""
        for mlp in (self.gate_mlp, self.trans_mlp):
            nn.init.xavier_uniform_(mlp[0].weight)
            nn.init.zeros_(mlp[0].bias)
            nn.init.zeros_(mlp[2].weight)
            nn.init.zeros_(mlp[2].bias)
        for layer in (self.global_map[0], self.global_map[2]):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.position_bias)

    def forward(self, blocks: torch.Tensor) -> torch.Tensor:
        """`(N, spatial, hidden) -> (N, hidden)`."""
        resid = self.norm_resid(blocks)
        global_feat = self.global_map(resid.flatten(1)).unsqueeze(1) + self.position_bias.view(
            1, self.spatial, self.hidden_size
        )
        paired = torch.cat((resid, global_feat), dim=-1)
        gate = torch.softmax(self.gate_mlp(paired), dim=1)
        trans = resid + self.trans_mlp(paired)
        return (gate * trans).sum(dim=1)
