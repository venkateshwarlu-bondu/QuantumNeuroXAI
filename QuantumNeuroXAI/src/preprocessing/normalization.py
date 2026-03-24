from __future__ import annotations
import numpy as np

def zscore_per_channel(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return ((x - mu) / (sd + eps)).astype(np.float32)

def robust_scale_per_channel(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    med = np.median(x, axis=-1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=-1, keepdims=True)
    return ((x - med) / (mad + eps)).astype(np.float32)
