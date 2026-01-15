# AI4PDE Partial-Observation Benchmark (Baselines)

A **config-driven benchmark harness** for reconstructing PDE solution fields from **partial observations**.

This repo aims to be:
- **Academic / readable**: clean structure, explicit configs, reproducible seeds, simple logging.
- **Reusable**: add PDEs, masks, and methods via small, isolated modules.
- **DiffusionPDE-compatible**: uses **DiffusionPDE** datasets and **their pretrained diffusion models** (downloaded at runtime; not bundled).

---

## Problem setting

For each PDE instance, the (ground-truth) solution field `u` is only partially observed:
\[
y = M u + \epsilon
\]
where `M` is a binary observation mask (partial observation operator) and `ε` is optional noise.

**Goal:** reconstruct the full solution field `u` from `y`.

---

## Baselines included (with references)

This repository provides a unified interface and configs for the following baselines:

- **DiffusionPDE** (pretrained; *no training here*): we call the official DiffusionPDE inference script and use their released pretrained models.  
  Reference: Huang et al., *DiffusionPDE: Generative PDE-Solving Under Partial Observation*, NeurIPS 2024 / arXiv:2406.17763.

- **FNO** (train yourself): supervised reconstructor mapping `(u_obs, mask) → u`.  
  Reference: Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations*, arXiv:2010.08895.

- **DeepONet** (train yourself): supervised reconstructor mapping `(u_obs, mask) → u`.  
  Reference: Lu et al., *DeepONet: Learning nonlinear operators…*, arXiv:1910.03193.

- **PINN** (eval-only; per-instance optimization): solves each test instance by minimizing observation mismatch + PDE residual (and optional boundary terms).  
  Reference: Raissi et al., *Physics-informed neural networks…*, J. Comput. Phys. 378 (2019).

> Note: FNO/DeepONet implementations here are **minimal reference baselines** intended for standardized benchmarking, not a reproduction of any single official training recipe.

---

## PDEs and masks

### PDE datasets (from DiffusionPDE)
- **Burgers**
- **Darcy**
- **Poisson**
- **Navier–Stokes** (vorticity / time-series as provided in DiffusionPDE)

### Masks (partial observation operators `M`)
- **M1**: random points (Bernoulli sampling)
- **M2**: regular grid subsampling (structured lattice)
- **M3**: block missing (inpainting-style contiguous holes)

Mask configs live in `configs/mask/` and are applied consistently across baselines.

---

# Quickstart

## 0) Requirements
- Python ≥ 3.9
- PyTorch ≥ 2.0

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
pip install -e .
