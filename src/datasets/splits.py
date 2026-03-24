from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split

def assign_splits(df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float, subject_wise: bool = True, seed: int = 42) -> pd.DataFrame:
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6
    out = df.copy()

    if subject_wise and "subject_id" in out.columns and out["subject_id"].nunique() > 1:
        subjects = out["subject_id"].astype(str).unique().tolist()
        train_sub, temp_sub = train_test_split(subjects, test_size=(1 - train_ratio), random_state=seed)
        rel_test = test_ratio / (val_ratio + test_ratio)
        val_sub, test_sub = train_test_split(temp_sub, test_size=rel_test, random_state=seed)
        split_map = {}
        for s in train_sub:
            split_map[s] = "train"
        for s in val_sub:
            split_map[s] = "val"
        for s in test_sub:
            split_map[s] = "test"
        out["split"] = out["subject_id"].map(split_map).fillna("train")
    else:
        idx = list(range(len(out)))
        train_idx, temp_idx = train_test_split(idx, test_size=(1 - train_ratio), random_state=seed)
        rel_test = test_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(temp_idx, test_size=rel_test, random_state=seed)
        out["split"] = "train"
        out.loc[val_idx, "split"] = "val"
        out.loc[test_idx, "split"] = "test"
    return out
