"""Extended Quantum-NAT classical backbone with residual CNN and transformer encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple 2‑D residual block with two conv layers and a skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride!= 1 or in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class QuantumNATModel(nn.Module):
    """Hybrid CNN‑Transformer backbone for image classification inspired by Quantum‑NAT."""

    def __init__(self, num_classes: int = 4, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        # Residual CNN backbone
        self.backbone = nn.Sequential(
            ResidualBlock(1, 16, stride=1),
            nn.MaxPool2d(2),
            ResidualBlock(16, 32, stride=2),
            nn.MaxPool2d(2),
        )
        # Flatten and feed into transformer encoder
        self.flatten = nn.Flatten()
        self.linear_proj = nn.Linear(32 * 7 * 7, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(d_model, num_classes),
            nn.BatchNorm1d(num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone
        features = self.backbone(x)          # shape: (B, 32, 7, 7)
        flattened = self.flatten(features)   # shape: (B, 32*7*7)
        proj = F.relu(self.linear_proj(flattened))  # shape: (B, d_model)
        # Transformer expects (S, B, D), we treat each sample as a sequence of length 1
        proj = proj.unsqueeze(0)  # (1, B, d_model)
        transformed = self.transformer(proj)  # (1, B, d_model)
        transformed = transformed.squeeze(0)  # (B, d_model)
        out = self.classifier(transformed)
        return out


__all__ = ["QuantumNATModel"]
