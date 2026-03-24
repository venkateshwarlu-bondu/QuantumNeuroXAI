from __future__ import annotations
import torch

def amplitude_encode(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(eps)
    return x / norm

def phase_encode(x: torch.Tensor, phase_scale: float = 3.14159) -> torch.Tensor:
    x_norm = torch.tanh(x)
    return phase_scale * x_norm

def combine_amplitude_phase(a: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return torch.cat([a * torch.cos(p), a * torch.sin(p)], dim=-1)
