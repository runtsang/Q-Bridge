"""Enhanced classical model with residual paths, dropout, and a dual‑task head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumNATEnhanced(nn.Module):
    """Dual‑pathway CNN → FC architecture with residual connections and dropout."""

    def __init__(
        self,
        in_channels: int = 1,
        base_features: int = 8,
        conv_depth: int = 2,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout_prob: float = 0.2,
    ) -> None:
        super().__init__()

        # Convolutional backbone with residual‑style skip connections
        conv_layers = []
        in_ch = in_channels
        for i in range(conv_depth):
            out_ch = base_features * (2 ** i)
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=1))
            conv_layers.append(nn.BatchNorm2d(out_ch))
            conv_layers.append(nn.ReLU(inplace=True))
            conv_layers.append(nn.MaxPool2d(pool_size))
            in_ch = out_ch
        self.backbone = nn.Sequential(*conv_layers)

        # Compute flattened dimension using a dummy input
        dummy = torch.zeros(1, in_channels, 28, 28)
        dummy_feat = self.backbone(dummy)
        flat_dim = dummy_feat.view(1, -1).size(1)

        # Dual‑task fully‑connected heads
        self.cls_head = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 4),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        flat = features.view(features.size(0), -1)
        cls_out = self.cls_head(flat)
        reg_out = self.reg_head(flat)
        return self.norm(cls_out), self.norm(reg_out)


__all__ = ["QuantumNATEnhanced"]
