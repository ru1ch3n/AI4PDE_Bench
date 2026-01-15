from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class DiffusionPDERunnerConfig:
    diffusionpde_dir: str = "third_party/DiffusionPDE"
    # Path *relative to diffusionpde_dir*
    config_path: Optional[str] = None
    python: str = "python3"


def run_diffusionpde(cfg: DiffusionPDERunnerConfig, *, extra_args: Optional[List[str]] = None) -> None:
    """Run DiffusionPDE's `generate_pde.py` with a given YAML config.

    This is a thin wrapper. It intentionally does not parse DiffusionPDE outputs,
    because upstream may change their output format.

    Expected setup:
      - third_party/DiffusionPDE/ exists (cloned)
      - pretrained models unzipped into the DiffusionPDE root (as per upstream)
      - datasets unzipped into DiffusionPDE's expected data path (as per upstream config)
    """
    dp_dir = Path(cfg.diffusionpde_dir)
    if not dp_dir.exists():
        raise FileNotFoundError(
            f"DiffusionPDE directory not found: {dp_dir}\nRun: bash scripts/fetch_diffusionpde.sh"
        )
    if cfg.config_path is None:
        raise ValueError(
            "cfg.method.config_path is required for DiffusionPDE runs. "

            "Example: configs/darcy-inverse.yaml (relative to the DiffusionPDE repo)."

        )
    config_path = dp_dir / cfg.config_path
    if not config_path.exists():
        raise FileNotFoundError(
            f"DiffusionPDE config not found: {config_path}\n"

            "Hint: check the file exists inside the cloned DiffusionPDE repo."

        )

    cmd = [cfg.python, "generate_pde.py", "--config", str(config_path)]
    if extra_args:
        cmd += list(extra_args)

    subprocess.run(cmd, cwd=str(dp_dir), check=True)
