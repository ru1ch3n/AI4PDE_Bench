from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Sequence[int] = (256, 256), act: str = "gelu"):
        super().__init__()
        acts = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}
        if act not in acts:
            raise ValueError(f"Unknown act={act}")
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(acts[act]())
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DeepONetConfig:
    latent_dim: int = 256
    branch_hidden: Tuple[int, ...] = (512, 512)
    trunk_hidden: Tuple[int, ...] = (256, 256)
    act: str = "gelu"


class DeepONetGrid(nn.Module):
    """Grid-based DeepONet.

    - Branch input: flattened (in_channels * grid_points)
    - Trunk input: coordinates (d)
    - Output: function values on the full grid

    Notes:
      * For fixed grids, trunk features can be precomputed.
      * This implementation supports multi-channel outputs via per-channel basis coefficients.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_shape: Sequence[int],
        cfg: DeepONetConfig,
        coord_range: Optional[Tuple[Tuple[float, float], ...]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_shape = tuple(int(s) for s in grid_shape)
        self.dim = len(self.grid_shape)
        self.cfg = cfg

        n_points = int(torch.tensor(self.grid_shape).prod().item())
        branch_in = in_channels * n_points
        branch_out = out_channels * cfg.latent_dim

        self.branch = MLP(branch_in, branch_out, hidden=cfg.branch_hidden, act=cfg.act)
        self.trunk = MLP(self.dim, cfg.latent_dim, hidden=cfg.trunk_hidden, act=cfg.act)

        # Build coordinate grid in [0,1]^d by default
        if coord_range is None:
            coord_range = tuple((0.0, 1.0) for _ in range(self.dim))
        coords = self._make_coords(self.grid_shape, coord_range)  # (n_points, dim)
        self.register_buffer("coords", coords, persistent=False)

        # Precompute trunk features for fixed coords
        with torch.no_grad():
            trunk_feat = self.trunk(self.coords)  # (n_points, latent_dim)
        self.register_buffer("trunk_feat", trunk_feat, persistent=False)

    @staticmethod
    def _make_coords(shape: Sequence[int], coord_range: Tuple[Tuple[float, float], ...]) -> torch.Tensor:
        grids = []
        for n, (lo, hi) in zip(shape, coord_range):
            g = torch.linspace(lo, hi, steps=n)
            grids.append(g)
        mesh = torch.meshgrid(*grids, indexing="ij")
        stacked = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
        return stacked.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, *grid_shape)
        b = x.shape[0]
        x_flat = x.reshape(b, -1)  # (B, C_in * n_points)
        coeff = self.branch(x_flat)  # (B, out_channels * latent)
        coeff = coeff.view(b, self.out_channels, self.cfg.latent_dim)  # (B, C_out, latent)

        # trunk_feat: (n_points, latent)
        # output: (B, C_out, n_points)
        out = torch.einsum("bcl,pl->bcp", coeff, self.trunk_feat)
        out = out.view(b, self.out_channels, *self.grid_shape)
        return out
