from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def _safe_filtfilt(b, a, x: np.ndarray) -> np.ndarray:
    if x.shape[-1] < max(len(a), len(b)) * 3:
        return x
    return filtfilt(b, a, x, axis=-1)

def bandpass_filter(x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 0.999)
    b, a = butter(order, [low, high], btype="band")
    return _safe_filtfilt(b, a, x).astype(np.float32)

def notch_filter(x: np.ndarray, fs: float, notch_hz: float = 50.0, q: float = 30.0) -> np.ndarray:
    if notch_hz <= 0 or notch_hz >= fs / 2:
        return x.astype(np.float32)
    b, a = iirnotch(notch_hz, q, fs=fs)
    return _safe_filtfilt(b, a, x).astype(np.float32)

def artifact_clip(x: np.ndarray, z_thresh: float = 6.0) -> np.ndarray:
    x = x.copy()
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True) + 1e-8
    z = (x - mu) / sd
    x = np.where(np.abs(z) > z_thresh, mu + np.sign(z) * z_thresh * sd, x)
    return x.astype(np.float32)
