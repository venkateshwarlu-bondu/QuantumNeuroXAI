from __future__ import annotations
import json
import os
from typing import Any, Dict
import numpy as np
from scipy.ndimage import zoom

def resize_tensor(tf_tensor: np.ndarray, target_freq_bins: int, target_time_bins: int) -> np.ndarray:
    # tf_tensor: [channels, freq, time]
    c, f, t = tf_tensor.shape
    if f == target_freq_bins and t == target_time_bins:
        return tf_tensor.astype(np.float32)
    zoom_f = target_freq_bins / float(f)
    zoom_t = target_time_bins / float(t)
    out = zoom(tf_tensor, zoom=(1.0, zoom_f, zoom_t), order=1)
    return out.astype(np.float32)

def save_processed_tensor(save_path: str, tensor: np.ndarray, label: int, metadata: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, x=tensor.astype(np.float32), y=int(label), meta_json=json.dumps(metadata))
