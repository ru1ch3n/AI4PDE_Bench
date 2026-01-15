#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="${ROOT_DIR}/third_party"
DP_DIR="${THIRD_PARTY}/DiffusionPDE"

mkdir -p "${THIRD_PARTY}"

if [[ -d "${DP_DIR}" ]]; then
  echo "[fetch_diffusionpde] DiffusionPDE already exists at: ${DP_DIR}"
  echo "If you want a clean re-clone, delete the folder first."
  exit 0
fi

echo "[fetch_diffusionpde] Cloning DiffusionPDE into ${DP_DIR}"
git clone --depth 1 https://github.com/jhhuangchloe/DiffusionPDE.git "${DP_DIR}"

echo "[fetch_diffusionpde] Done."
