from __future__ import annotations
import numpy as np
from scipy.signal import stft

def compute_stft_tensor(segment: np.ndarray, fs: float, nperseg: int, noverlap: int) -> np.ndarray:
    # segment: [channels, samples]
    specs = []
    for ch in range(segment.shape[0]):
        _, _, z = stft(segment[ch], fs=fs, nperseg=nperseg, noverlap=noverlap, boundary=None, padded=False)
        specs.append(np.abs(z).astype(np.float32))
    return np.stack(specs).astype(np.float32)  # [channels, freq, time]

def compute_batch_stft(segments: np.ndarray, fs: float, nperseg: int, noverlap: int) -> np.ndarray:
    return np.stack([compute_stft_tensor(seg, fs, nperseg, noverlap) for seg in segments]).astype(np.float32)
