from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split

from ai4pde_bench.baselines.diffusionpde_runner import DiffusionPDERunnerConfig, run_diffusionpde
from ai4pde_bench.baselines.pinn import PINNConfig, solve_pinn_instance
from ai4pde_bench.baselines.supervised import SupervisedTrainConfig, train as train_supervised, evaluate as eval_supervised
from ai4pde_bench.data.diffusionpde import DiffusionPDEDataConfig, DiffusionPDEDataset
from ai4pde_bench.masks import MaskConfig
from ai4pde_bench.models import DeepONetConfig, DeepONetGrid, FNO1d, FNO2d, FNO3d
from ai4pde_bench.utils.seed import seed_all


def _make_dataloaders(cfg: DictConfig):
    # Train split
    data_cfg = DiffusionPDEDataConfig(
        data_dir=cfg.data.data_dir,
        pde=cfg.pde.data_key,
        split="train",
        coeff_path=cfg.data.get("coeff_path", None),
        sol_path=cfg.data.get("sol_path", None),
        max_instances=cfg.data.get("max_train_instances", None),
    )
    mask_cfg = MaskConfig(
        name=cfg.mask.name,
        obs_ratio=float(cfg.mask.obs_ratio),
        noise_std=float(cfg.mask.get("noise_std", 0.0)),
        seed=int(cfg.mask.get("seed", 0)),
    )
    full_train = DiffusionPDEDataset(data_cfg, mask_cfg)

    val_fraction = float(cfg.data.get("val_fraction", 0.1))
    n_val = max(1, int(round(len(full_train) * val_fraction)))
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        pin_memory=True,
    )

    # Test split
    test_data_cfg = DiffusionPDEDataConfig(
        data_dir=cfg.data.data_dir,
        pde=cfg.pde.data_key,
        split="test",
        coeff_path=cfg.data.get("coeff_path_test", None),
        sol_path=cfg.data.get("sol_path_test", None),
        max_instances=cfg.data.get("max_test_instances", None),
    )
    test_ds = DiffusionPDEDataset(test_data_cfg, mask_cfg)
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg.eval.batch_size),
        shuffle=False,
        num_workers=int(cfg.eval.num_workers),
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, test_ds


def _infer_grid_shape(test_ds: DiffusionPDEDataset) -> Tuple[int, ...]:
    sample = test_ds[0]
    u = sample["u"]  # (1, ...)
    return tuple(int(s) for s in u.shape[1:])


def _build_model(cfg: DictConfig, grid_shape: Tuple[int, ...]) -> torch.nn.Module:
    in_ch = 2 if bool(cfg.train.input_include_mask) else 1
    out_ch = 1

    if cfg.method.name == "fno":
        d = len(grid_shape)
        if d == 1:
            return FNO1d(in_ch, out_ch, width=cfg.method.width, modes=cfg.method.modes, depth=cfg.method.depth)
        if d == 2:
            return FNO2d(in_ch, out_ch, width=cfg.method.width, modes1=cfg.method.modes1, modes2=cfg.method.modes2, depth=cfg.method.depth)
        if d == 3:
            return FNO3d(in_ch, out_ch, width=cfg.method.width, modes1=cfg.method.modes1, modes2=cfg.method.modes2, modes3=cfg.method.modes3, depth=cfg.method.depth)
        raise ValueError(f"FNO does not support grid dim={d} for grid_shape={grid_shape}")

    if cfg.method.name == "deeponet":
        dcfg = DeepONetConfig(
            latent_dim=int(cfg.method.latent_dim),
            branch_hidden=tuple(int(x) for x in cfg.method.branch_hidden),
            trunk_hidden=tuple(int(x) for x in cfg.method.trunk_hidden),
            act=str(cfg.method.act),
        )
        return DeepONetGrid(in_ch, out_ch, grid_shape=grid_shape, cfg=dcfg)

    raise ValueError(f"Unknown method for supervised model build: {cfg.method.name}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Reproducibility
    seed_all(int(cfg.seed))

    # Hydra run dir
    run_dir = Path(os.getcwd())
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    # Handle DiffusionPDE baseline (external runner)
    if cfg.method.name == "diffusionpde":
        dp_cfg = DiffusionPDERunnerConfig(
            diffusionpde_dir=cfg.method.diffusionpde_dir,
            config_path=cfg.method.config_path,
            python=cfg.method.python,
        )
        run_diffusionpde(dp_cfg)
        return

    # Build data
    train_loader, val_loader, test_loader, test_ds = _make_dataloaders(cfg)
    grid_shape = _infer_grid_shape(test_ds)

    if cfg.method.name in {"fno", "deeponet"}:
        model = _build_model(cfg, grid_shape)

        if cfg.mode == "train":
            train_cfg = SupervisedTrainConfig(
                device=str(cfg.device),
                batch_size=int(cfg.train.batch_size),
                num_workers=int(cfg.train.num_workers),
                lr=float(cfg.train.lr),
                weight_decay=float(cfg.train.weight_decay),
                max_epochs=int(cfg.train.max_epochs),
                input_include_mask=bool(cfg.train.input_include_mask),
                amp=bool(cfg.train.amp),
            )
            metrics = train_supervised(model, train_loader, val_loader, train_cfg, out_dir=run_dir)
            print("TRAIN DONE:", metrics)
            return

        if cfg.mode == "eval":
            if cfg.method.get("checkpoint_path", None) is None:
                raise ValueError("For eval mode with supervised baselines, set method.checkpoint_path=...")

            ckpt = torch.load(cfg.method.checkpoint_path, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            model.to(cfg.device)

            metrics = eval_supervised(model, test_loader, device=str(cfg.device), include_mask=bool(cfg.train.input_include_mask))
            (run_dir / "eval_metrics.yaml").write_text(OmegaConf.to_yaml(metrics))
            print("EVAL:", metrics)
            return

        raise ValueError(f"Unknown mode={cfg.mode}")

    if cfg.method.name == "pinn":
        if cfg.mode != "eval":
            raise ValueError("PINN baseline is eval-only (per-instance optimization). Use mode=eval.")

        pinn_cfg = PINNConfig(
            device=str(cfg.device),
            lr=float(cfg.method.lr),
            steps=int(cfg.method.steps),
            obs_weight=float(cfg.method.obs_weight),
            pde_weight=float(cfg.method.pde_weight),
            boundary_weight=float(cfg.method.boundary_weight),
            nu=float(cfg.method.nu),
            hidden=tuple(int(x) for x in cfg.method.hidden),
            act=str(cfg.method.act),
            max_test_instances=int(cfg.method.max_test_instances),
        )

        rels = []
        for i in range(min(len(test_ds), pinn_cfg.max_test_instances)):
            batch = test_ds[i]
            out = solve_pinn_instance(cfg.pde.name, batch, pinn_cfg)
            rels.append(float(out["rel_l2"].item()))
        rel_mean = sum(rels) / max(1, len(rels))
        metrics = {"rel_l2_mean": rel_mean, "n": len(rels)}
        (run_dir / "eval_metrics.yaml").write_text(OmegaConf.to_yaml(metrics))
        print("PINN EVAL:", metrics)
        return

    raise ValueError(f"Unknown method.name={cfg.method.name}")


if __name__ == "__main__":
    main()
