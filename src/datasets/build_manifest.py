from __future__ import annotations
import os
import re
from pathlib import Path
from typing import List, Dict
import pandas as pd

SUPPORTED_EEG_EXTS = {".edf", ".gdf", ".fif", ".bdf", ".mat", ".npy"}

def _scan_files(raw_dir: str) -> List[Path]:
    files = []
    for root, _, names in os.walk(raw_dir):
        for name in names:
            path = Path(root) / name
            if path.suffix.lower() in SUPPORTED_EEG_EXTS:
                files.append(path)
    return sorted(files)

def _extract_subject_session(path: Path) -> tuple[str, str]:
    parts = [p.lower() for p in path.parts]
    subject = "unknown_subject"
    session = "unknown_session"
    for p in parts:
        if re.match(r"chb\d+", p) or re.match(r"s\d+", p) or re.match(r"a\d+", p):
            subject = p
        if "session" in p or re.match(r"run\d+", p):
            session = p
    return subject, session

def build_generic_manifest(raw_dir: str, out_csv: str, dataset_name: str) -> pd.DataFrame:
    rows: List[Dict] = []
    for path in _scan_files(raw_dir):
        subject_id, session_id = _extract_subject_session(path)
        rows.append({
            "dataset": dataset_name,
            "subject_id": subject_id,
            "session_id": session_id,
            "recording_id": path.stem,
            "file_path": str(path),
            "file_type": path.suffix.lower(),
            "sfreq": "",
            "n_channels": "",
            "label_info": "",
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df

def build_tuh_manifest(raw_dir: str, out_csv: str) -> pd.DataFrame:
    return build_generic_manifest(raw_dir, out_csv, "tuh")

def build_chbmit_manifest(raw_dir: str, out_csv: str) -> pd.DataFrame:
    return build_generic_manifest(raw_dir, out_csv, "chbmit")

def build_bci2a_manifest(raw_dir: str, out_csv: str) -> pd.DataFrame:
    return build_generic_manifest(raw_dir, out_csv, "bci2a")
