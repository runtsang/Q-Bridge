"""Hybrid regression model combining classical fully‑connected, convolutional, and transformer blocks."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class FCLayer(nn.Module):
    """Classical fully‑connected layer emulating the FCL example."""
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x)).mean(dim=1, keepdim=True)

class QuanvolutionFilter(nn.Module):
    """Convolutional filter inspired by the quanvolution example."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class TransformerBlock(nn.Module):
    """Simple transformer encoder block."""
    def __init__(self, embed_dim: int, num_heads: int, dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder_layer(x))

class HybridRegressionModel(nn.Module):
    """Hybrid regression model combining classical fully‑connected, quanvolution, and transformer blocks."""
    def __init__(
        self,
        num_features: int = 784,
        conv_out_channels: int = 4,
        transformer_heads: int = 4,
        transformer_ffn: int = 128,
    ):
        super().__init__()
        self.num_features = num_features
        self.fclayer = FCLayer(num_features)
        self.qfilter = QuanvolutionFilter(out_channels=conv_out_channels)
        self.transformer = TransformerBlock(
            embed_dim=conv_out_channels,
            num_heads=transformer_heads,
            dim_feedforward=transformer_ffn,
        )
        self.head = nn.Linear(conv_out_channels + 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, num_features)
        # reshape for convolution
        x_img = x.view(-1, 1, 28, 28)
        conv_features = self.qfilter(x_img)  # (batch, conv_out_channels * 14 * 14)
        seq_len = conv_features.size(1) // self.qfilter.conv.out_channels
        seq = conv_features.view(x.size(0), seq_len, self.qfilter.conv.out_channels)
        transformer_out = self.transformer(seq)  # (batch, seq_len, conv_out_channels)
        transformer_out = transformer_out.mean(dim=1)  # (batch, conv_out_channels)
        fcl_out = self.fclayer(x)  # (batch, 1)
        combined = torch.cat([transformer_out, fcl_out], dim=-1)  # (batch, conv_out_channels+1)
        return self.head(combined).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
