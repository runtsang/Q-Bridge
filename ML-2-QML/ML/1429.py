"""Hybrid classical model with advanced feature fusion and optional self‑attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumNATExtended(nn.Module):
    """A classical CNN‑based network that incorporates a self‑attention block and
    a quantum‑inspired fully‑connected head. The output is a 4‑dimensional feature
    vector suitable for downstream tasks.

    The architecture preserves the original four‑feature output while adding
    richer representations through attention and a lightweight variational
    MLP.  The module can be used as a drop‑in replacement for the original
    QFCModel.
    """

    def __init__(self, use_attention: bool = True, attention_heads: int = 4) -> None:
        super().__init__()
        self.use_attention = use_attention

        # Feature extractor – identical to the original CNN
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Optional self‑attention module
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=16 * 7 * 7, num_heads=attention_heads, batch_first=True
            )
        else:
            self.attention = None

        # Quantum‑inspired fully‑connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)

        if self.use_attention:
            # Multi‑head attention expects (batch, seq_len, embed_dim)
            # Here seq_len = 1, so we reshape accordingly.
            q = k = v = flattened.unsqueeze(1)  # (bsz, 1, embed_dim)
            attn_output, _ = self.attention(q, k, v)
            flattened = attn_output.squeeze(1)

        out = self.fc(flattened)
        return self.norm(out)


__all__ = ["QuantumNATExtended"]
