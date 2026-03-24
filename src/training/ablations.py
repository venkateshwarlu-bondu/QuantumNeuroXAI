from __future__ import annotations
from typing import Dict, Any, Callable

def list_ablation_variants() -> Dict[str, str]:
    return {
        "cnn_attention": "Classical baseline with CNN + AttentionRNN",
        "quantum_full": "QuantumNeuroXAI full hybrid model",
    }
