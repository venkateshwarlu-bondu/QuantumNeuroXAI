from __future__ import annotations
from typing import Dict, Any
from src.utils.io import save_json

def build_unified_report(signal_summary: Dict[str, Any], attention: Any, fused_attr: Any, quantum_rank: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prediction": prediction,
        "signal_level": signal_summary,
        "model_level": {
            "attention_weights": attention.tolist() if hasattr(attention, "tolist") else attention,
            "fused_feature_abs": fused_attr.tolist() if hasattr(fused_attr, "tolist") else fused_attr,
        },
        "quantum_level": quantum_rank,
    }

def save_explanation_report(report: Dict[str, Any], path: str) -> None:
    save_json(report, path)
