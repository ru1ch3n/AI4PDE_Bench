from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ai4pde_bench.masks import MaskConfig, apply_mask, make_mask


_COEFF_KEYS = ["coeff", "coef", "perme", "perm", "k", "a", "input"]
_SOL_KEYS = ["sol", "solution", "field", "u", "output", "vorticity", "omega"]


def _score_name(name: str, keys: list[str]) -> int:
    n = name.lower()
    return sum(1 for k in keys if k in n)


def _find_split_dir(data_dir: Path, split: str) -> Path:
    """Try a few common DiffusionPDE layouts."""
    split = split.lower()
    candidates = [
        data_dir / split,
        data_dir / {"train": "training", "test": "testing"}.get(split, split),
        data_dir / "data" / split,
        data_dir / "data" / {"train": "training", "test": "testing"}.get(split, split),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not find split directory for split='{}'. Tried: {}\nHint: set cfg.data.data_dir correctly.".format(
            split, [str(c) for c in candidates]
        )
    )


def _find_pde_dir(split_dir: Path, pde: str) -> Path:
    pde = pde.lower()
    candidates = [
        split_dir / pde,
        split_dir / pde.upper(),
        split_dir / pde.capitalize(),
        split_dir / pde.replace("_", "-"),
        split_dir / pde.replace("-", "_"),
    ]
    for c in candidates:
        if c.exists():
            return c
    # fallback: search one level deep
    for child in split_dir.iterdir():
        if child.is_dir() and child.name.lower() == pde:
            return child
    raise FileNotFoundError(
        f"Could not find PDE folder '{pde}' under '{split_dir}'.\n"

        f"Available subfolders: {[p.name for p in split_dir.iterdir() if p.is_dir()]}"

    )


def discover_arrays(data_dir: Path, split: str, pde: str) -> Tuple[Optional[Path], Path]:
    """Heuristically discover (coeff_path, sol_path)."""
    split_dir = _find_split_dir(data_dir, split)
    pde_dir = _find_pde_dir(split_dir, pde)
    npys = sorted(pde_dir.glob("*.npy"))
    if not npys:
        npys = sorted(pde_dir.rglob("*.npy"))
    if not npys:
        raise FileNotFoundError(f"No .npy files found under {pde_dir}")

    if len(npys) == 1:
        return None, npys[0]

    coeff_scores = [(p, _score_name(p.name, _COEFF_KEYS)) for p in npys]
    sol_scores = [(p, _score_name(p.name, _SOL_KEYS)) for p in npys]

    coeff_path = max(coeff_scores, key=lambda t: t[1])[0]
    sol_path = max(sol_scores, key=lambda t: t[1])[0]

    if coeff_path == sol_path:
        coeff_path, sol_path = npys[0], npys[1]

    return coeff_path, sol_path


@dataclass
class DiffusionPDEDataConfig:
    data_dir: str = "data/DiffusionPDE"
    pde: str = "darcy"
    split: str = "train"
    coeff_path: Optional[str] = None
    sol_path: Optional[str] = None
    max_instances: Optional[int] = None


class DiffusionPDEDataset(Dataset):
    """Loads DiffusionPDE-style datasets from `.npy` files.

    Returns a dict with:
      - a: (1, ...) or omitted if not available
      - u: (1, ...) full solution
      - mask: (1, ...) observation mask (1=observed)
      - u_obs: (1, ...) masked noisy observation
    """

    def __init__(self, data_cfg: DiffusionPDEDataConfig, mask_cfg: MaskConfig):
        self.data_cfg = data_cfg
        self.mask_cfg = mask_cfg

        data_dir = Path(data_cfg.data_dir)

        # Resolve paths
        if data_cfg.sol_path is not None:
            sol_path = Path(data_cfg.sol_path)
            if not sol_path.is_absolute():
                sol_path = data_dir / sol_path
            coeff_path = None
            if data_cfg.coeff_path is not None:
                cp = Path(data_cfg.coeff_path)
                coeff_path = cp if cp.is_absolute() else (data_dir / cp)
        else:
            coeff_path, sol_path = discover_arrays(data_dir, data_cfg.split, data_cfg.pde)

        self.sol_path = sol_path
        self.coeff_path = coeff_path

        self._u = np.load(self.sol_path, mmap_mode="r")
        self._a = np.load(self.coeff_path, mmap_mode="r") if self.coeff_path is not None else None

        if data_cfg.max_instances is not None:
            self._u = self._u[: data_cfg.max_instances]
            if self._a is not None:
                self._a = self._a[: data_cfg.max_instances]

        if self._a is not None and len(self._a) != len(self._u):
            raise ValueError(
                f"Coeff and solution have different N: {len(self._a)} vs {len(self._u)}\n"

                f"coeff_path={self.coeff_path}, sol_path={self.sol_path}"

            )

    def __len__(self) -> int:
        return int(self._u.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        u = np.array(self._u[idx], dtype=np.float32)  # copy slice
        if u.ndim == 1:
            u = u[None, :]
        else:
            u = u[None, ...]  # (1, ...)

        a_t: Optional[torch.Tensor] = None
        if self._a is not None:
            a = np.array(self._a[idx], dtype=np.float32)
            if a.ndim == 1:
                a = a[None, :]
            else:
                a = a[None, ...]
            a_t = torch.from_numpy(a)

        spatial_shape = u.shape[1:]
        rng = np.random.default_rng(self.mask_cfg.seed + int(idx))
        mask = make_mask(spatial_shape, self.mask_cfg, rng=rng)
        u_obs = apply_mask(u[0], mask, noise_std=self.mask_cfg.noise_std, rng=rng)

        out: Dict[str, torch.Tensor] = {
            "u": torch.from_numpy(u),
            "mask": torch.from_numpy(mask[None, ...]),
            "u_obs": torch.from_numpy(u_obs[None, ...]),
        }
        if a_t is not None:
            out["a"] = a_t
        return out
