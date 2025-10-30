"""Enhanced classical model: a dual‑branch CNN‑Transformer encoder with regularization."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATEnhanced(nn.Module):
    """
    Classical backbone for Quantum‑NAT that combines a CNN feature extractor with
    a transformer encoder for global context.  The model can be paired with a
    quantum sub‑module (see the QML variant) for joint training.

    The network:
        - ConvBlock → ConvBlock → AvgPool
        - Flatten → TransformerEncoder (self‑attention)
        - Linear head → BatchNorm1d
    """
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 4,
                 hidden_dim: int = 128,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(2)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # (batch, 16, 7, 7)

        bsz = x.size(0)
        x = x.view(bsz, -1)  # (batch, 16*7*7)

        x_t = x.unsqueeze(0)  # (1, batch, 16*7*7)
        x_t = self.transformer(x_t)  # (1, batch, 16*7*7)
        x = x_t.squeeze(0)  # (batch, 16*7*7)

        out = self.fc(x)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
