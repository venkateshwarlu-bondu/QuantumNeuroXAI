from __future__ import annotations
import os
from typing import Dict, Any, List
import numpy as np
from src.datasets.readers import read_signal_any
from src.preprocessing.filters import bandpass_filter, notch_filter, artifact_clip
from src.preprocessing.normalization import zscore_per_channel, robust_scale_per_channel
from src.preprocessing.segmentation import sliding_window_segments
from src.preprocessing.time_frequency import compute_batch_stft
from src.preprocessing.tensor_projection import resize_tensor, save_processed_tensor

def _infer_label_from_record(dataset_name: str, record: Dict[str, Any]) -> int:
    path = str(record["file_path"]).lower()
    if dataset_name == "chbmit":
        return 1 if any(tok in path for tok in ["seiz", "sz", "ictal"]) else 0
    if dataset_name == "bci2a":
        # Best-effort multiclass fallback from file name
        for i, tok in enumerate(["left", "right", "foot", "tongue"]):
            if tok in path:
                return i
        return 0
    return 0

def preprocess_recording(record: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    read_cfg = cfg["read"]
    pre_cfg = cfg["preprocessing"]
    dataset_name = cfg["dataset_name"]

    pack = read_signal_any(record["file_path"], target_sfreq=read_cfg["target_sfreq"])
    x = pack["x"]
    fs = float(pack["sfreq"])
    ch_names = pack["ch_names"]

    x = bandpass_filter(x, fs, pre_cfg["bandpass_low"], pre_cfg["bandpass_high"])
    x = notch_filter(x, fs, pre_cfg["notch_hz"])
    x = artifact_clip(x, pre_cfg["artifact_z_thresh"])

    norm = pre_cfg.get("normalization", "zscore")
    if norm == "robust":
        x = robust_scale_per_channel(x)
    else:
        x = zscore_per_channel(x)

    segments = sliding_window_segments(x, fs, pre_cfg["window_sec"], pre_cfg["stride_sec"])
    if len(segments) == 0:
        return []

    tf_batch = compute_batch_stft(segments, fs, pre_cfg["stft_nperseg"], pre_cfg["stft_noverlap"])

    processed_rows: List[Dict[str, Any]] = []
    label = _infer_label_from_record(dataset_name, record)

    for i in range(tf_batch.shape[0]):
        tf_tensor = resize_tensor(tf_batch[i], pre_cfg["target_freq_bins"], pre_cfg["target_time_bins"])
        rel_dir = os.path.join(cfg["processed_dir"], str(record.get("subject_id", "unknown")))
        fname = f"{record.get('recording_id', 'rec')}_seg{i:04d}.npz"
        save_path = os.path.join(rel_dir, fname)
        metadata = {
            "dataset": dataset_name,
            "subject_id": str(record.get("subject_id", "unknown")),
            "session_id": str(record.get("session_id", "unknown")),
            "recording_id": str(record.get("recording_id", "unknown")),
            "segment_index": i,
            "sfreq": fs,
            "channels": ch_names,
            "source_file": record["file_path"],
        }
        save_processed_tensor(save_path, tf_tensor, label, metadata)
        processed_rows.append({
            "dataset": dataset_name,
            "subject_id": metadata["subject_id"],
            "session_id": metadata["session_id"],
            "recording_id": metadata["recording_id"],
            "segment_index": i,
            "label": label,
            "tensor_path": save_path,
        })
    return processed_rows
