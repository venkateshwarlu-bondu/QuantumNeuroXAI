from __future__ import annotations
import argparse
from src.utils.io import load_yaml
from src.datasets.build_manifest import build_tuh_manifest, build_chbmit_manifest, build_bci2a_manifest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["tuh", "chbmit", "bci2a"])
    args = parser.parse_args()

    cfg_path = {
        "tuh": "configs/dataset_tuh.yaml",
        "chbmit": "configs/dataset_chbmit.yaml",
        "bci2a": "configs/dataset_bci2a.yaml",
    }[args.dataset]

    cfg = load_yaml(cfg_path)

    if args.dataset == "tuh":
        df = build_tuh_manifest(cfg["raw_dir"], cfg["manifest_csv"])
    elif args.dataset == "chbmit":
        df = build_chbmit_manifest(cfg["raw_dir"], cfg["manifest_csv"])
    else:
        df = build_bci2a_manifest(cfg["raw_dir"], cfg["manifest_csv"])

    print(f"Saved {len(df)} records to {cfg['manifest_csv']}")

if __name__ == "__main__":
    main()
