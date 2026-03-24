from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample
import mne

def resample_signal(x: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    if int(original_fs) == int(target_fs):
        return x
    n_target = int(round(x.shape[-1] * target_fs / original_fs))
    return resample(x, n_target, axis=-1)

def read_with_mne(file_path: str) -> Tuple[np.ndarray, float, list[str]]:
    lower = file_path.lower()
    if lower.endswith(".edf") or lower.endswith(".bdf"):
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose="ERROR")
    elif lower.endswith(".gdf"):
        raw = mne.io.read_raw_gdf(file_path, preload=True, verbose="ERROR")
    elif lower.endswith(".fif"):
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose="ERROR")
    else:
        raise ValueError(f"MNE reader unsupported for file: {file_path}")
    x = raw.get_data().astype(np.float32)
    fs = float(raw.info["sfreq"])
    ch_names = list(raw.ch_names)
    return x, fs, ch_names

def read_mat_fallback(file_path: str) -> Tuple[np.ndarray, float, list[str], Dict[str, Any]]:
    obj = loadmat(file_path)
    meta: Dict[str, Any] = {"mat_keys": [k for k in obj.keys() if not k.startswith("__")]}
    candidate = None
    for k, v in obj.items():
        if k.startswith("__"):
            continue
        if hasattr(v, "shape") and len(v.shape) == 2 and min(v.shape) > 4:
            candidate = v
            break
    if candidate is None:
        raise ValueError(f"Could not infer signal matrix from MAT file: {file_path}")
    x = np.asarray(candidate, dtype=np.float32)
    if x.shape[0] > x.shape[1]:
        x = x.T
    fs = 250.0
    ch_names = [f"ch_{i:02d}" for i in range(x.shape[0])]
    return x, fs, ch_names, meta

def read_signal_any(file_path: str, target_sfreq: Optional[float] = None) -> Dict[str, Any]:
    lower = file_path.lower()
    meta: Dict[str, Any] = {"file_path": file_path}
    try:
        if lower.endswith((".edf", ".gdf", ".fif", ".bdf")):
            x, fs, ch_names = read_with_mne(file_path)
        elif lower.endswith(".mat"):
            x, fs, ch_names, mat_meta = read_mat_fallback(file_path)
            meta.update(mat_meta)
        elif lower.endswith(".npy"):
            x = np.load(file_path).astype(np.float32)
            if x.ndim != 2:
                raise ValueError("NPY EEG file must have shape [channels, time]")
            fs = float(target_sfreq or 256.0)
            ch_names = [f"ch_{i:02d}" for i in range(x.shape[0])]
        else:
            raise ValueError(f"Unsupported file extension: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to read EEG file {file_path}: {e}") from e

    if target_sfreq is not None and fs != target_sfreq:
        x = resample_signal(x, fs, target_sfreq)
        fs = float(target_sfreq)

    return {"x": x, "sfreq": fs, "ch_names": ch_names, "meta": meta}
