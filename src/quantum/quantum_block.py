from __future__ import annotations
import torch
import torch.nn as nn
from src.quantum.encoders import amplitude_encode, phase_encode, combine_amplitude_phase
from src.quantum.feature_map import TrigFeatureMap, EntanglementInspiredMixing
from src.quantum.measurement import MeasurementProjection

class QuantumFeatureBlock(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int, feature_depth: int, measure_dim: int, phase_scale: float = 3.14159):
        super().__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.phase_scale = phase_scale

        self.proj = nn.Linear(input_dim, proj_dim)
        qdim = proj_dim * 2
        self.feature_map = TrigFeatureMap(qdim, depth=feature_depth)
        self.entangle = EntanglementInspiredMixing(qdim)
        self.measure = MeasurementProjection(qdim, measure_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        a = amplitude_encode(h)
        p = phase_encode(h, self.phase_scale)
        q = combine_amplitude_phase(a, p)
        q = self.feature_map(q)
        q = self.entangle(q)
        q = self.measure(q)
        return q
