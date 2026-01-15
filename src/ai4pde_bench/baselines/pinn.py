from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ai4pde_bench.utils.metrics import relative_l2_error


class CoordMLP(nn.Module):
    """Simple coordinate MLP used by the PINN baseline."""
    def __init__(self, in_dim: int, hidden: Tuple[int, ...] = (256, 256, 256), act: str = "tanh"):
        super().__init__()
        acts = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
        if act not in acts:
            raise ValueError(f"Unknown act={act}")
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(acts[act]())
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)  # (N,1)


@dataclass
class PINNConfig:
    device: str = "cpu"
    lr: float = 1e-3
    steps: int = 2000
    obs_weight: float = 1.0
    pde_weight: float = 1.0
    boundary_weight: float = 1.0
    # Burgers / NS viscosity
    nu: float = 1e-3
    # Network
    hidden: Tuple[int, ...] = (256, 256, 256)
    act: str = "tanh"
    # Use only a subset of test instances for speed
    max_test_instances: int = 10


def _coords_grid(shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Return coordinate grid in [0,1]^d flattened to (N, d)."""
    grids = [torch.linspace(0.0, 1.0, steps=s, device=device) for s in shape]
    mesh = torch.meshgrid(*grids, indexing="ij")
    coords = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
    return coords


def _laplacian_2d(u: torch.Tensor, h: float) -> torch.Tensor:
    # u: (B,1,H,W)
    u_pad = F.pad(u, (1, 1, 1, 1), mode="constant", value=0.0)
    lap = (
        u_pad[:, :, 2:, 1:-1]
        + u_pad[:, :, :-2, 1:-1]
        + u_pad[:, :, 1:-1, 2:]
        + u_pad[:, :, 1:-1, :-2]
        - 4.0 * u_pad[:, :, 1:-1, 1:-1]
    ) / (h * h)
    # pad back to full shape with zeros at boundary
    out = torch.zeros_like(u)
    out[:, :, 1:-1, 1:-1] = lap[:, :, 1:-1, 1:-1]
    return out


def _poisson_residual(u: torch.Tensor, a: Optional[torch.Tensor]) -> torch.Tensor:
    # Assume -Δu = a (a is source term)
    b, c, h, w = u.shape
    dx = 1.0 / (h - 1)
    lap = _laplacian_2d(u, dx)
    rhs = a if a is not None else torch.zeros_like(u)
    return -lap - rhs


def _darcy_residual(u: torch.Tensor, a: Optional[torch.Tensor]) -> torch.Tensor:
    # Darcy: -div(a * grad u) = 1, with Dirichlet u=0 on boundary.
    # a: permeability field (B,1,H,W)
    if a is None:
        raise ValueError("Darcy residual requires coefficient field a (permeability).")
    b, c, h, w = u.shape
    dx = 1.0 / (h - 1)

    # Variable-coefficient 5-point stencil
    u_pad = F.pad(u, (1, 1, 1, 1), mode="constant", value=0.0)
    a_pad = F.pad(a, (1, 1, 1, 1), mode="constant", value=0.0)

    u_c = u_pad[:, :, 1:-1, 1:-1]
    u_r = u_pad[:, :, 1:-1, 2:]
    u_l = u_pad[:, :, 1:-1, :-2]
    u_u = u_pad[:, :, 2:, 1:-1]
    u_d = u_pad[:, :, :-2, 1:-1]

    a_c = a_pad[:, :, 1:-1, 1:-1]
    a_r = a_pad[:, :, 1:-1, 2:]
    a_l = a_pad[:, :, 1:-1, :-2]
    a_u = a_pad[:, :, 2:, 1:-1]
    a_d = a_pad[:, :, :-2, 1:-1]

    a_xp = 0.5 * (a_c + a_r)
    a_xm = 0.5 * (a_c + a_l)
    a_yp = 0.5 * (a_c + a_u)
    a_ym = 0.5 * (a_c + a_d)

    div = (a_xp * (u_r - u_c) - a_xm * (u_c - u_l) + a_yp * (u_u - u_c) - a_ym * (u_c - u_d)) / (dx * dx)

    f = torch.ones_like(u_c)
    res = -div - f  # interior
    out = torch.zeros_like(u)
    out[:, :, 1:-1, 1:-1] = res[:, :, 1:-1, 1:-1]
    return out


def _burgers_residual(u: torch.Tensor, nu: float) -> torch.Tensor:
    # u: (B,1,X,T)
    b, c, x, t = u.shape
    dx = 1.0 / (x - 1)
    dt = 1.0 / (t - 1)

    # pad with periodic in x, and constant in t
    u_xp = torch.roll(u, shifts=-1, dims=2)
    u_xm = torch.roll(u, shifts=1, dims=2)
    u_t_p = F.pad(u, (0, 1, 0, 0), mode="replicate")[:, :, :, 1:]
    u_t_m = F.pad(u, (1, 0, 0, 0), mode="replicate")[:, :, :, :-1]

    u_x = (u_xp - u_xm) / (2 * dx)
    u_xx = (u_xp - 2 * u + u_xm) / (dx * dx)
    u_t = (u_t_p - u_t_m) / (2 * dt)

    res = u_t + u * u_x - nu * u_xx
    # zero boundary in time edges (optional)
    res[:, :, :, 0] = 0
    res[:, :, :, -1] = 0
    return res


def _ns_velocity_from_vorticity(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute (u,v) from vorticity w using stream function on a periodic domain.

    w: (B, H, W) real
    returns u,v: (B, H, W) real
    """
    b, h, w_ = w.shape
    w_hat = torch.fft.fft2(w)  # complex

    kx = torch.fft.fftfreq(h, d=1.0 / h).to(w.device) * 2 * torch.pi
    ky = torch.fft.fftfreq(w_, d=1.0 / w_).to(w.device) * 2 * torch.pi
    kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing="ij")
    k2 = kx_grid ** 2 + ky_grid ** 2
    k2[0, 0] = 1.0  # avoid div0

    psi_hat = -w_hat / k2  # Δpsi = w -> psi_hat = -w_hat/k^2
    psi_hat[..., 0, 0] = 0.0

    u_hat = 1j * ky_grid * psi_hat
    v_hat = -1j * kx_grid * psi_hat

    u = torch.fft.ifft2(u_hat).real
    v = torch.fft.ifft2(v_hat).real
    return u, v


def _ns_residual(w: torch.Tensor, nu: float) -> torch.Tensor:
    # w: (B,1,H,W,T)
    b, c, h, w_, t = w.shape
    dx = 1.0 / h
    dt = 1.0 / (t - 1)

    # spatial derivatives (periodic)
    w_xp = torch.roll(w, shifts=-1, dims=3)
    w_xm = torch.roll(w, shifts=1, dims=3)
    w_yp = torch.roll(w, shifts=-1, dims=2)
    w_ym = torch.roll(w, shifts=1, dims=2)

    w_x = (w_xp - w_xm) / (2 * dx)
    w_y = (w_yp - w_ym) / (2 * dx)
    lap = (w_xp + w_xm + w_yp + w_ym - 4.0 * w) / (dx * dx)

    # time derivative (central)
    w_tp = F.pad(w, (0, 1, 0, 0, 0, 0), mode="replicate")[:, :, :, :, 1:]
    w_tm = F.pad(w, (1, 0, 0, 0, 0, 0), mode="replicate")[:, :, :, :, :-1]
    w_t = (w_tp - w_tm) / (2 * dt)

    # velocity for each time slice
    adv = torch.zeros_like(w)
    for ti in range(t):
        w_slice = w[:, 0, :, :, ti]  # (B,H,W)
        u, v = _ns_velocity_from_vorticity(w_slice)
        adv[:, 0, :, :, ti] = u * w_x[:, 0, :, :, ti] + v * w_y[:, 0, :, :, ti]

    res = w_t + adv - nu * lap
    res[:, :, :, :, 0] = 0
    res[:, :, :, :, -1] = 0
    return res


def _boundary_mask_2d(h: int, w: int, device: torch.device) -> torch.Tensor:
    m = torch.zeros((1, 1, h, w), device=device)
    m[:, :, 0, :] = 1
    m[:, :, -1, :] = 1
    m[:, :, :, 0] = 1
    m[:, :, :, -1] = 1
    return m


def solve_pinn_instance(pde: str, batch: Dict[str, torch.Tensor], cfg: PINNConfig) -> Dict[str, torch.Tensor]:
    """Solve one instance with a PINN-style optimization."""
    device = torch.device(cfg.device)
    pde = pde.lower()

    u_true = batch["u"].to(device)  # (1,1,...)
    u_obs = batch["u_obs"].to(device)
    mask = batch["mask"].to(device)
    a = batch.get("a", None)
    if a is not None:
        a = a.to(device)

    # Determine coordinate dimension from u shape (excluding batch/channel)
    spatial_shape = tuple(int(s) for s in u_true.shape[2:]) if u_true.dim() > 2 else tuple(int(s) for s in u_true.shape[1:])
    # We always store u as (B=1, C=1, *shape)
    shape = tuple(int(s) for s in u_true.shape[2:])
    coord_dim = len(shape)

    coords = _coords_grid(shape, device=device)  # (N, coord_dim)

    net = CoordMLP(in_dim=coord_dim, hidden=cfg.hidden, act=cfg.act).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    # Precompute observed indices for faster loss
    obs_idx = torch.nonzero(mask.reshape(-1) > 0.5, as_tuple=False).squeeze(1)
    y_obs_flat = u_obs.reshape(-1)[obs_idx]

    # boundary mask for Poisson/Darcy
    boundary_mask = None
    if pde in {"poisson", "darcy"} and coord_dim == 2:
        boundary_mask = _boundary_mask_2d(shape[0], shape[1], device=device).reshape(-1) > 0.5

    for _ in tqdm(range(cfg.steps), desc=f"PINN[{pde}]", leave=False):
        opt.zero_grad(set_to_none=True)
        u_pred_flat = net(coords).reshape(1, 1, *shape)

        # observation loss
        pred_obs = u_pred_flat.reshape(-1)[obs_idx]
        loss_obs = torch.mean((pred_obs - y_obs_flat) ** 2)

        # PDE residual loss
        if pde == "poisson":
            res = _poisson_residual(u_pred_flat, a)
        elif pde == "darcy":
            res = _darcy_residual(u_pred_flat, a)
        elif pde == "burgers":
            res = _burgers_residual(u_pred_flat, nu=cfg.nu)
        elif pde in {"ns", "navier_stokes", "navier-stokes"}:
            res = _ns_residual(u_pred_flat, nu=cfg.nu)
        else:
            raise ValueError(f"Unknown PDE: {pde}")

        loss_pde = torch.mean(res ** 2)

        # boundary loss (Dirichlet u=0)
        loss_b = torch.tensor(0.0, device=device)
        if boundary_mask is not None:
            u_flat = u_pred_flat.reshape(-1)
            loss_b = torch.mean((u_flat[boundary_mask]) ** 2)

        loss = cfg.obs_weight * loss_obs + cfg.pde_weight * loss_pde + cfg.boundary_weight * loss_b
        loss.backward()
        opt.step()

    with torch.no_grad():
        u_pred = net(coords).reshape(1, 1, *shape)
        err = relative_l2_error(u_pred, u_true).item()

    return {"u_pred": u_pred.detach().cpu(), "rel_l2": torch.tensor(err)}
