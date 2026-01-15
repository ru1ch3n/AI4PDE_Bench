from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class MaskConfig:
    """Mask settings.

    Attributes:
        name: One of {m1_random, m2_regular, m3_block}.
        obs_ratio: Fraction of entries observed (mask==1).
        noise_std: Optional Gaussian noise std added on observed entries.
        seed: Base seed (deterministic per-index masks are created with seed+idx).
    """

    name: str
    obs_ratio: float
    noise_std: float = 0.0
    seed: int = 0


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def make_mask(shape: Sequence[int], cfg: MaskConfig, *, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Create a {0,1} mask with the given spatial shape.

    Mask semantics:
      - mask == 1 means observed
      - mask == 0 means missing
    """
    if rng is None:
        rng = _rng(cfg.seed)

    shape = tuple(int(s) for s in shape)
    if any(s <= 0 for s in shape):
        raise ValueError(f"Invalid shape={shape}")

    total = int(np.prod(shape))
    num_obs = int(round(cfg.obs_ratio * total))
    num_obs = max(1, min(total, num_obs))

    name = cfg.name.lower()

    if name in {"m1_random", "random", "m1"}:
        mask = np.zeros(total, dtype=np.float32)
        idx = rng.choice(total, size=num_obs, replace=False)
        mask[idx] = 1.0
        return mask.reshape(shape)

    if name in {"m2_regular", "regular", "grid", "m2"}:
        d = len(shape)
        # Rough stride so that 1/stride^d ≈ obs_ratio.
        stride = int(round(cfg.obs_ratio ** (-1.0 / d))) if cfg.obs_ratio > 0 else max(shape)
        stride = max(1, stride)

        mask = np.zeros(shape, dtype=np.float32)
        slicer = tuple(slice(0, s, stride) for s in shape)
        mask[slicer] = 1.0

        # Adjust to match desired count more closely.
        cur = int(mask.sum())
        if cur > num_obs:
            ones = np.flatnonzero(mask.reshape(-1) > 0)
            drop = rng.choice(ones, size=(cur - num_obs), replace=False)
            flat = mask.reshape(-1)
            flat[drop] = 0.0
            mask = flat.reshape(shape)
        elif cur < num_obs:
            zeros = np.flatnonzero(mask.reshape(-1) == 0)
            add = rng.choice(zeros, size=(num_obs - cur), replace=False)
            flat = mask.reshape(-1)
            flat[add] = 1.0
            mask = flat.reshape(shape)
        return mask.astype(np.float32)

    if name in {"m3_block", "block", "m3"}:
        # Block missing: observe everything *outside* a contiguous missing block.
        missing = total - num_obs
        missing = max(1, min(total - 1, missing))

        d = len(shape)
        side = int(round(missing ** (1.0 / d)))
        side = max(1, side)

        block_sizes = [min(side, shape[i]) for i in range(d)]
        vol = int(np.prod(block_sizes))
        if vol < missing:
            last = block_sizes[-1]
            grow = int(np.ceil(missing / max(1, vol) * last))
            block_sizes[-1] = min(shape[-1], max(1, grow))
        elif vol > missing:
            last = block_sizes[-1]
            shrink = int(np.floor(missing / max(1, vol) * last))
            block_sizes[-1] = min(shape[-1], max(1, shrink))

        starts = []
        for dim, bs in zip(shape, block_sizes):
            if dim == bs:
                starts.append(0)
            else:
                starts.append(int(rng.integers(0, dim - bs + 1)))

        mask = np.ones(shape, dtype=np.float32)
        slices = tuple(slice(st, st + bs) for st, bs in zip(starts, block_sizes))
        mask[slices] = 0.0

        # Ensure exact count (optional fine-tune).
        cur = int(mask.sum())
        if cur > num_obs:
            ones = np.flatnonzero(mask.reshape(-1) > 0)
            drop = rng.choice(ones, size=(cur - num_obs), replace=False)
            flat = mask.reshape(-1)
            flat[drop] = 0.0
            mask = flat.reshape(shape)
        elif cur < num_obs:
            zeros = np.flatnonzero(mask.reshape(-1) == 0)
            add = rng.choice(zeros, size=(num_obs - cur), replace=False)
            flat = mask.reshape(-1)
            flat[add] = 1.0
            mask = flat.reshape(shape)

        return mask.astype(np.float32)

    raise ValueError(f"Unknown mask type: {cfg.name}")


def apply_mask(u: np.ndarray, mask: np.ndarray, noise_std: float = 0.0, *, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Apply mask (observed=1) and optional i.i.d. Gaussian noise on observed entries."""
    if rng is None:
        rng = np.random.default_rng(0)
    obs = u * mask
    if noise_std > 0:
        noise = rng.normal(loc=0.0, scale=noise_std, size=u.shape).astype(u.dtype)
        obs = obs + noise * mask
    return obs
