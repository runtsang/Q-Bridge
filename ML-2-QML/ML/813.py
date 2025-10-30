"""Enhanced classical model with residual CNN and self‑attention based on Quantum‑NAT."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumNATEnhanced(nn.Module):
    """Residual CNN followed by a self‑attention block and a fully connected head."""

    def __init__(self) -> None:
        super().__init__()
        # Residual convolutional block
        self.res_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)

        # Self‑attention on the flattened feature map
        self.attn = nn.MultiheadAttention(
            embed_dim=32 * 14 * 14, num_heads=4, batch_first=True
        )

        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(32 * 14 * 14, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 4),
        )
        self.norm = nn.LayerNorm(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        x = self.res_conv(x)
        x = self.pool(x)
        x = x.view(bsz, -1)  # (batch, seq_len=1, embed_dim)
        x = x.unsqueeze(1)   # add sequence dimension
        attn_out, _ = self.attn(x, x, x)
        attn_out = attn_out.squeeze(1)  # remove sequence dimension
        out = self.fc(attn_out)
        return self.norm(out)


__all__ = ["QuantumNATEnhanced"]
