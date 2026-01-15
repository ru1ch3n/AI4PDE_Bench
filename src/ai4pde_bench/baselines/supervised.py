from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ai4pde_bench.utils.checkpoint import save_checkpoint
from ai4pde_bench.utils.metrics import relative_l2_error


PathLike = Union[str, Path]


@dataclass
class SupervisedTrainConfig:
    device: str = "cpu"
    batch_size: int = 8
    num_workers: int = 0
    lr: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 50
    log_every: int = 50
    input_include_mask: bool = True
    amp: bool = False


def _make_inp(batch: Dict[str, torch.Tensor], include_mask: bool) -> torch.Tensor:
    u_obs = batch["u_obs"]
    mask = batch["mask"]
    if include_mask:
        return torch.cat([u_obs, mask], dim=1)
    return u_obs


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, include_mask: bool) -> Dict[str, float]:
    model.eval()
    errs = []
    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        x = _make_inp(batch, include_mask)
        y = batch["u"]
        pred = model(x)
        errs.append(relative_l2_error(pred, y).detach().cpu())
    errs = torch.cat(errs, dim=0)
    return {"rel_l2_mean": float(errs.mean().item()), "rel_l2_std": float(errs.std().item())}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    cfg: SupervisedTrainConfig,
    out_dir: PathLike,
) -> Dict[str, float]:
    device = torch.device(cfg.device)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    out_dir = Path(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"

    global_step = 0
    last_metrics: Dict[str, float] = {}
    for epoch in range(cfg.max_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.max_epochs}", leave=False)
        for batch in pbar:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            x = _make_inp(batch, cfg.input_include_mask)
            y = batch["u"]

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                pred = model(x)
                loss = torch.mean((pred - y) ** 2)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            global_step += 1
            pbar.set_postfix(loss=float(loss.item()))

        # checkpoint last
        save_checkpoint(last_path, model, opt, extra={"epoch": epoch, "global_step": global_step})

        # evaluate
        if val_loader is not None:
            metrics = evaluate(model, val_loader, device=str(device), include_mask=cfg.input_include_mask)
            last_metrics = metrics
            val = metrics["rel_l2_mean"]
            if val < best:
                best = val
                save_checkpoint(best_path, model, opt, extra={"epoch": epoch, "global_step": global_step, "best": best})

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps({"best_rel_l2": best, **last_metrics}, indent=2))
    return {"best_rel_l2": best, **last_metrics}
