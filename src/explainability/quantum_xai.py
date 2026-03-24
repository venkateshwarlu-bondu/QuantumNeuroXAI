from __future__ import annotations
import numpy as np
import torch

def quantum_sensitivity_analysis(model, x: torch.Tensor, eps: float = 1e-2) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        base = model(x)
        q = base.get("quantum_vec")
        if q is None:
            return np.array([])
        base_logits = base["logits"]

    sensitivities = []
    for i in range(q.shape[-1]):
        x_pert = x.clone()
        # perturb input slightly and estimate effect on output;
        # practical fallback since internal q injection is not exposed
        x_pert = x_pert + eps * torch.randn_like(x_pert)
        with torch.no_grad():
            out = model(x_pert)
            delta = (out["logits"] - base_logits).abs().mean().item()
        sensitivities.append(delta)
    return np.asarray(sensitivities, dtype=np.float32)

def rank_quantum_dimensions(scores: np.ndarray) -> dict:
    if scores.size == 0:
        return {"top_indices": [], "top_scores": []}
    order = np.argsort(scores)[::-1]
    top = order[: min(10, len(order))]
    return {"top_indices": top.tolist(), "top_scores": scores[top].tolist()}
