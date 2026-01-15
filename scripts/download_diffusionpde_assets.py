#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import gdown


TRAIN_ID = "1z4ypsU3JdkAsoY9Px-JSw9RS2f5StNv5"
TEST_ID = "1HdkeCKMLvDN_keIBTijOFYrRcA3Quy0l"
MODELS_ID = "1w4V0o-nTjpHP_Xv32Rt_SgPGmVa9PwL_"


def _download(file_id: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[download] Exists: {out_path}")
        return
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[download] {url} -> {out_path}")
    gdown.download(url, str(out_path), quiet=False)


def _unzip(zip_path: Path, out_dir: Path) -> None:
    print(f"[unzip] {zip_path} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="third_party/DiffusionPDE/data", help="Where to place extracted training/testing data.")
    ap.add_argument("--diffusionpde_dir", type=str, default="third_party/DiffusionPDE", help="DiffusionPDE repo root (for pretrained models).")

    ap.add_argument("--train", action="store_true", help="Download + unzip training.zip")
    ap.add_argument("--test", action="store_true", help="Download + unzip testing.zip")
    ap.add_argument("--models", action="store_true", help="Download + unzip pretrained-models.zip")
    ap.add_argument("--all", action="store_true", help="Download everything")

    args = ap.parse_args()
    if args.all:
        args.train = args.test = args.models = True

    data_dir = Path(args.data_dir)
    dp_dir = Path(args.diffusionpde_dir)

    zip_dir = data_dir / "_zips"
    zip_dir.mkdir(parents=True, exist_ok=True)

    if args.train:
        train_zip = zip_dir / "training.zip"
        _download(TRAIN_ID, train_zip)
        _unzip(train_zip, data_dir)

    if args.test:
        test_zip = zip_dir / "testing.zip"
        _download(TEST_ID, test_zip)
        _unzip(test_zip, data_dir)

    if args.models:
        if not dp_dir.exists():
            raise FileNotFoundError(
                f"DiffusionPDE repo not found at {dp_dir}. Run: bash scripts/fetch_diffusionpde.sh"
            )
        models_zip = dp_dir / "pretrained-models.zip"
        _download(MODELS_ID, models_zip)
        _unzip(models_zip, dp_dir)

    print("[done]")


if __name__ == "__main__":
    main()
