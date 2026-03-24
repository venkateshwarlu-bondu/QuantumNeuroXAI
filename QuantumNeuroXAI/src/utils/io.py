import json
import os
from typing import Any, Dict
import pandas as pd
import yaml

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)
