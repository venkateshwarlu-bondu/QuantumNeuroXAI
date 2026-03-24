from __future__ import annotations
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class TemporalCNN(nn.Module):
    def __init__(self, in_ch: int = 1, conv_channels=(16, 32, 64), kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        blocks = []
        prev = in_ch
        for ch in conv_channels:
            blocks.append(ConvBlock(prev, ch, kernel_size, dropout))
            prev = ch
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
