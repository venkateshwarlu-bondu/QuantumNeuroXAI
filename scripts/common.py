from __future__ import annotations
import os
from typing import Tuple
from torch.utils.data import DataLoader
from src.utils.io import load_yaml
from src.utils.config import deep_merge
from src.utils.device import get_device
from src.datasets.unified_dataset import ProcessedEEGDataset

def load_configs(global_cfg_path: str, dataset_cfg_path: str, model_cfg_path: str | None = None, xai_cfg_path: str | None = None):
    cfg = load_yaml(global_cfg_path)
    dcfg = load_yaml(dataset_cfg_path)
    cfg = deep_merge(cfg, dcfg)
    mcfg = load_yaml(model_cfg_path) if model_cfg_path else None
    xcfg = load_yaml(xai_cfg_path) if xai_cfg_path else None
    return cfg, dcfg, mcfg, xcfg

def build_loaders(processed_manifest_csv: str, batch_size: int, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = ProcessedEEGDataset(processed_manifest_csv, split="train")
    val_ds = ProcessedEEGDataset(processed_manifest_csv, split="val")
    test_ds = ProcessedEEGDataset(processed_manifest_csv, split="test")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

def infer_num_classes(dataset_cfg: dict) -> int:
    task = dataset_cfg["task"]
    if task["mode"] == "multiclass":
        return int(task.get("num_classes", 4))
    return 2
