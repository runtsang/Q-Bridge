"""Enhanced classical model with residual connections and multi‑head feature extraction."""

from __future__ import annotations

import torch
import torch.nn as nn


class QuantumNATHybrid(nn.Module):
    """Hybrid classical model inspired by Quantum‑NAT with residuals and multi‑head feature extraction.

    Features
    --------
    - Multi‑head convolutional encoder that processes the input with different kernel sizes.
    - Residual connections between encoder blocks to ease gradient flow.
    - Optional depthwise‑separable convolutions for parameter efficiency.
    - Adaptive average pooling to a fixed spatial resolution regardless of input size.
    - Bottleneck fully‑connected head producing the 4‑dimensional output.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        num_heads: int = 3,
        base_channels: int = 8,
        use_depthwise: bool = False,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Multi‑head encoder
        self.encoders = nn.ModuleList()
        for i in range(num_heads):
            ks = 3 + 2 * i  # 3,5,7
            conv = nn.Conv2d(
                base_channels,
                base_channels,
                kernel_size=ks,
                padding=ks // 2,
                groups=base_channels if use_depthwise else 1,
                bias=False,
            )
            bn = nn.BatchNorm2d(base_channels)
            act = nn.ReLU(inplace=True)
            self.encoders.append(nn.Sequential(conv, bn, act))

        self.residual = nn.Conv2d(base_channels, base_channels, kernel_size=1, bias=False)

        self.pool = nn.AdaptiveAvgPool2d((2, 2))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(64, num_classes),
        )

        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        out = x
        for layer in self.encoders:
            out = layer(out) + out
        out = self.residual(out) + out
        out = self.pool(out)
        out = self.fc(out)
        return self.norm(out)


__all__ = ["QuantumNATHybrid"]
