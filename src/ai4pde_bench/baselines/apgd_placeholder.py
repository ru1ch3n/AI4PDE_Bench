"""APGD placeholder.

This repository intentionally does NOT implement APGD (you said you'll add it later).
The placeholder exists so the project structure stays consistent.

When you implement APGD:
- add your sampler / proximal operators here
- register it in ai4pde_bench/run.py (method switch)
- add configs under configs/method/apgd.yaml, etc.
"""

from dataclasses import dataclass


@dataclass
class APGDPlaceholderConfig:
    enabled: bool = False
