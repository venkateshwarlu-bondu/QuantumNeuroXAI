from __future__ import annotations
import torch
import torch.nn as nn

class TrigFeatureMap(nn.Module):
    def __init__(self, dim: int, depth: int = 2):
        super().__init__()
        self.depth = depth
        self.mix = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.mix:
            z = layer(h)
            h = torch.sin(z) + torch.cos(z) + 0.5 * h
        return h

class EntanglementInspiredMixing(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pair = x * torch.roll(x, shifts=1, dims=-1)
        return self.proj(x + pair)
