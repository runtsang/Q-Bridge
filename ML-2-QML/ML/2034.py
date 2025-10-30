"""Hybrid classical model with enhanced feature extraction for Quantum‑NAT."""
from __future__ import annotations

import torch
import torch.nn as nn

class QuantumNATGen240(nn.Module):
    """Classical CNN encoder followed by multi‑head self‑attention and a fully connected head."""
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.attention = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        feat = self.encoder(x)  # shape: (bsz, 32, 7, 7)
        bsz, c, h, w = feat.shape
        seq = feat.view(bsz, c, h * w).transpose(1, 2)  # (bsz, seq_len, c)
        attn_out, _ = self.attention(seq, seq, seq)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, c, h, w)
        flat = attn_out.view(bsz, -1)  # flatten
        out = self.fc(flat)
        return self.norm(out)
