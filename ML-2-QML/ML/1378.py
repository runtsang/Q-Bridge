"""Hybrid classical model with self‑attention and a quantum‑inspired head.

The original QFCModel was a simple CNN‑to‑FC architecture.  This
extension adds:
*   a multi‑head self‑attention block that processes the
    *features* before flattening, and
*   a final fully‑connected head that outputs four features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumNATEnhanced(nn.Module):
    """Hybrid classical model with self‑attention and a quantum‑inspired head."""

    def __init__(self) -> None:
        super().__init__()
        # Convolutional backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Self‑attention block
        self.attn = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
        # Fully‑connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)  # (bsz, 16, 7, 7)
        # Prepare sequence for attention: (bsz, seq_len, embed_dim)
        seq = features.permute(0, 2, 3, 1).reshape(bsz, -1, 16)  # seq_len = 49
        attn_output, _ = self.attn(seq, seq, seq)
        # Project back to feature map shape
        attn_output = attn_output.reshape(bsz, 7, 7, 16).permute(0, 3, 1, 2)
        # Flatten and pass through FC
        flattened = attn_output.reshape(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)


__all__ = ["QuantumNATEnhanced"]
