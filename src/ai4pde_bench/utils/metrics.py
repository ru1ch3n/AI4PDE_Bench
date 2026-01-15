from __future__ import annotations

import torch


def relative_l2_error(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute relative L2 error per-sample.

    Args:
        pred, target: tensors of shape (B, C, ...)

    Returns:
        Tensor of shape (B,) with per-sample relative L2.
    """
    b = pred.shape[0]
    pred_f = pred.reshape(b, -1)
    targ_f = target.reshape(b, -1)
    num = torch.norm(pred_f - targ_f, dim=1)
    den = torch.norm(targ_f, dim=1).clamp_min(eps)
    return num / den
