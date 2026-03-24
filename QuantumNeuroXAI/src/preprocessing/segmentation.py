from __future__ import annotations
import numpy as np

def sliding_window_segments(x: np.ndarray, fs: float, window_sec: float, stride_sec: float) -> np.ndarray:
    window = int(round(window_sec * fs))
    stride = int(round(stride_sec * fs))
    if x.shape[-1] < window:
        return np.empty((0, x.shape[0], window), dtype=np.float32)
    segments = []
    for start in range(0, x.shape[-1] - window + 1, stride):
        segments.append(x[:, start:start + window])
    if not segments:
        return np.empty((0, x.shape[0], window), dtype=np.float32)
    return np.stack(segments).astype(np.float32)
