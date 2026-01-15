#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ai4pde_bench.data.diffusionpde import DiffusionPDEDataConfig, DiffusionPDEDataset
from ai4pde_bench.masks import MaskConfig, make_mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="third_party/DiffusionPDE/data")
    ap.add_argument("--pde", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "test", "training", "testing"])
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--mask", type=str, default="m1_random", choices=["m1_random", "m2_regular", "m3_block"])
    ap.add_argument("--obs_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    split = args.split
    if split == "training":
        split = "train"
    if split == "testing":
        split = "test"

    data_cfg = DiffusionPDEDataConfig(data_dir=args.data_dir, pde=args.pde, split=split)
    mask_cfg = MaskConfig(name=args.mask, obs_ratio=args.obs_ratio, seed=args.seed)
    ds = DiffusionPDEDataset(data_cfg, mask_cfg)

    sample = ds[args.index]
    u = sample["u"].numpy()  # (1, ...)
    shape = u.shape[1:]
    m = make_mask(shape, mask_cfg, rng=np.random.default_rng(mask_cfg.seed + args.index))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, m.astype(np.float32))
    print(f"[saved] mask shape={m.shape}, mean={m.mean():.4f} -> {out_path}")


if __name__ == "__main__":
    main()
