from __future__ import annotations
import torch
import numpy as np

def input_saliency(model, x: torch.Tensor, target_class=None) -> np.ndarray:
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    out = model(x)
    logits = out["logits"]
    if logits.shape[-1] == 1:
        score = logits[:, 0].sum()
    else:
        if target_class is None:
            target_class = logits.argmax(dim=-1)
        score = logits[torch.arange(logits.shape[0]), target_class].sum()
    score.backward()
    sal = x.grad.detach().abs().cpu().numpy()
    return sal

def integrated_gradients_signal(model, x: torch.Tensor, baseline=None, steps: int = 32) -> np.ndarray:
    if baseline is None:
        baseline = torch.zeros_like(x)
    grads = []
    for alpha in torch.linspace(0, 1, steps, device=x.device):
        xi = baseline + alpha * (x - baseline)
        xi.requires_grad_(True)
        out = model(xi)
        logits = out["logits"]
        score = logits[:, 0].sum() if logits.shape[-1] == 1 else logits.max(dim=-1).values.sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        grads.append(xi.grad.detach())
    avg_grad = torch.stack(grads).mean(dim=0)
    ig = ((x - baseline) * avg_grad).abs().detach().cpu().numpy()
    return ig

def channel_band_summary(saliency_map: np.ndarray) -> dict:
    # input saliency shape: [batch, 1, channels, freq, time]
    s = saliency_map.mean(axis=(0, 1))
    ch_scores = s.mean(axis=(1, 2))
    freq_scores = s.mean(axis=(0, 2))
    time_scores = s.mean(axis=(0, 1))
    return {
        "channel_scores": ch_scores.tolist(),
        "frequency_scores": freq_scores.tolist(),
        "time_scores": time_scores.tolist(),
    }
