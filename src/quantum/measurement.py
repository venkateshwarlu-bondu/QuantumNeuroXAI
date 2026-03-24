from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class MeasurementProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(self.fc(x), normalized_shape=(self.fc.out_features,))
