"""QuantumNATGen – a deep, residual‑based CNN with optional quantum fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QFCModel(nn.Module):
    """
    Residual CNN with configurable depth and a skip connection.
    The architecture follows the original QFCModel but adds:
    * depth: number of conv blocks.
    * residual: add skip connections after each block.
    * fusion: method to combine classical output with a quantum embedding.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        depth: int = 2,
        residual: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.depth = depth
        self.residual = residual
        self.device = device

        layers = []
        in_ch = in_channels
        for i in range(depth):
            conv = nn.Conv2d(in_ch, 8 * (i + 1), kernel_size=3, padding=1)
            bn = nn.BatchNorm2d(conv.out_channels)
            relu = nn.ReLU(inplace=True)
            pool = nn.MaxPool2d(2)
            block = nn.Sequential(conv, bn, relu, pool)
            layers.append(block)
            in_ch = conv.out_channels
        self.features = nn.Sequential(*layers)

        # Assume 28x28 input; compute flattened size after pooling
        fc_input_dim = (28 // (2**depth)) ** 2 * in_ch
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.features:
            residual = out if self.residual else None
            out = block(out)
            if self.residual and residual is not None and residual.shape[1] == out.shape[1]:
                out = out + residual
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return self.norm(out)

    def fuse_with_quantum(self, x: torch.Tensor, quantum_emb: torch.Tensor) -> torch.Tensor:
        """
        Concatenate classical output with a quantum embedding and pass through a linear layer.
        """
        classical = self.forward(x)
        combined = torch.cat([classical, quantum_emb], dim=-1)
        return self.norm(combined)
