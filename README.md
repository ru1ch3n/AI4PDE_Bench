# AI4PDE Partial-Observation Benchmark (Baselines)

This repository is a **config-driven benchmark harness** for PDE reconstruction from **partial observations**.

It is designed to be:
- **Elegant / academic**: clear folder structure, experiment configs, reproducible seeds, clean logging.
- **Reusable**: add new PDEs, masks, and methods via small, isolated modules.
- **Compatible with DiffusionPDE**: uses DiffusionPDE datasets + their pretrained diffusion models (downloaded, not bundled).

## Baselines included
- **DiffusionPDE** (pretrained; *no training* here — we call their inference script)
- **FNO** (train yourself)
- **DeepONet** (train yourself)
- **PINN** (per-instance optimization)

> APGD is intentionally **not implemented** here. A placeholder config + module stub is provided so you can plug it in later.

## PDEs and masks
PDEs:
- Burgers
- Darcy
- Poisson
- Navier–Stokes (vorticity)

Masks (partial observation operators `M`):
- **M1**: random points
- **M2**: regular grid subsampling
- **M3**: block missing (inpainting-style)

## What the benchmark solves
For each PDE instance, we treat the **solution field** `u` as partially observed:
\[
y = M u + \epsilon
\]
and benchmark methods for reconstructing the full `u` from `y` (optionally leveraging PDE constraints).

This aligns with the “reconstruction/inference under partial observations” setting.

---

# Quickstart

## 1) Create an environment
Minimal (pip) setup:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

**Important:** run commands from the repo root with `PYTHONPATH=src` so the package can find the local configs:

```bash
export PYTHONPATH=src
```


## 2) Fetch DiffusionPDE (code)
We do **not** vendor DiffusionPDE in this repo. Instead, clone it into `third_party/`:

```bash
bash scripts/fetch_diffusionpde.sh
```

This will create:
```
third_party/DiffusionPDE/
```

## 3) Download datasets + pretrained diffusion models
DiffusionPDE provides Google Drive links for:
- `training.zip`
- `testing.zip`
- `pretrained-models.zip`

Download + unzip into the expected locations:

```bash
python scripts/download_diffusionpde_assets.py --all
```

Default locations:
- datasets: `third_party/DiffusionPDE/data/`
- pretrained diffusion models: `third_party/DiffusionPDE/pretrained-models/`

> If you prefer different paths, pass `--data_dir ...` and `--diffusionpde_dir ...` — see script help.

## 4) Run a baseline
All runs are driven by Hydra configs in `configs/`.

Example: train an FNO reconstructor for Poisson with mask M1:

```bash
python -m ai4pde_bench.run \
  mode=train \
  pde=poisson \
  mask=m1_random \
  method=fno
```

Then evaluate the trained checkpoint:

```bash
python -m ai4pde_bench.run \
  mode=eval \
  pde=poisson \
  mask=m1_random \
  method=fno \
  method.checkpoint_path=results/checkpoints/poisson/fno/m1_random/best.pt
```

Run DiffusionPDE baseline (pretrained):

```bash
python -m ai4pde_bench.run \
  mode=eval \
  pde=poisson \
  mask=m1_random \
  method=diffusionpde
```

---

# Folder structure

```
ai4pde_partial_obs_bench/
  configs/                 # Hydra configs (pde, mask, method, ...)
  scripts/                 # download + setup helpers
  src/ai4pde_bench/        # benchmark code
    data/                  # dataset loaders (DiffusionPDE)
    masks.py               # M1-M3 mask operators
    models/                # FNO + DeepONet minimal implementations
    baselines/             # DiffusionPDE runner + PINN
    run.py                 # main entrypoint (hydra)
  third_party/             # external code lives here (DiffusionPDE)
  data/                    # datasets live here (downloaded)
  results/                 # logs, metrics, checkpoints
```

---

# Notes / assumptions

1. **DiffusionPDE dataset files**
   DiffusionPDE distributes `.npy` datasets. File naming can differ across PDEs.
   This repo includes a **robust dataset discovery** layer:
   - It searches for reasonable `.npy` candidates under the given PDE folder.
   - You can always override paths explicitly in the config.

2. **Reproducibility**
   We set seeds for Python / NumPy / PyTorch per run, and write a `config.yaml` snapshot into the run directory.

3. **Licensing**
   - This repo’s *new code* is MIT (see `LICENSE`).
   - DiffusionPDE code + pretrained models have their own license (see upstream).

---

# Adding APGD later

A placeholder is provided:
- `configs/method/apgd_placeholder.yaml`
- `src/ai4pde_bench/baselines/apgd_placeholder.py`

You can implement APGD and register it in `src/ai4pde_bench/run.py`.
