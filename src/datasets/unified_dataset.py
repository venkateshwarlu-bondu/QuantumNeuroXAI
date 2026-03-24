from __future__ import annotations
import json
import os
from typing import Optional, Callable
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class ProcessedEEGDataset(Dataset):
    def __init__(self, processed_manifest_csv: str, split: Optional[str] = None, transform: Optional[Callable] = None):
        self.df = pd.read_csv(processed_manifest_csv)
        if split is not None and "split" in self.df.columns:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        pack = np.load(row["tensor_path"], allow_pickle=True)
        x = pack["x"].astype(np.float32)
        y = int(pack["y"])
        meta_json = str(pack["meta_json"])
        meta = json.loads(meta_json)

        # Add model channel dimension for Conv2d: [1, C, F, T]
        x = torch.from_numpy(x).float().unsqueeze(0)
        y = torch.tensor(y, dtype=torch.long)

        if self.transform is not None:
            x = self.transform(x)

        return {"x": x, "y": y, "meta": meta}
