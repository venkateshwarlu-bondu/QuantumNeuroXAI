from __future__ import annotations
import torch
import torch.nn as nn

class QuantumClassicalFusion(nn.Module):
    def __init__(self, quantum_dim: int, classical_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(quantum_dim + classical_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = torch.cat([q, c], dim=-1)
        return self.out(self.gate(h))
