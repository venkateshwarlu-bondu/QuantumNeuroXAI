from __future__ import annotations
import argparse
import pandas as pd
from tqdm import tqdm
from src.utils.io import load_yaml, save_csv
from src.preprocessing.pipeline import preprocess_recording
from src.datasets.splits import assign_splits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["tuh", "chbmit", "bci2a"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--global-config", default="configs/global.yaml")
    args = parser.parse_args()

    gcfg = load_yaml(args.global_config)
    cfg = load_yaml(args.config)

    df = pd.read_csv(cfg["manifest_csv"])
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        rows.extend(preprocess_recording(row.to_dict(), cfg))

    out = pd.DataFrame(rows)
    out = assign_splits(
        out,
        train_ratio=gcfg["splits"]["train_ratio"],
        val_ratio=gcfg["splits"]["val_ratio"],
        test_ratio=gcfg["splits"]["test_ratio"],
        subject_wise=gcfg["splits"]["subject_wise"],
        seed=gcfg["seed"],
    )
    save_csv(out, cfg["processed_manifest_csv"])
    print(f"Saved processed manifest to {cfg['processed_manifest_csv']} with {len(out)} segments")

if __name__ == "__main__":
    main()
