from __future__ import annotations
import numpy as np
import torch

def extract_attention_weights(model_output) -> np.ndarray | None:
    attn = model_output.get("attention_weights")
    if attn is None:
        return None
    return attn.detach().cpu().numpy()

def fused_feature_attribution(model_output) -> np.ndarray | None:
    fused = model_output.get("fused_vec")
    if fused is None:
        return None
    return fused.detach().abs().cpu().numpy()
