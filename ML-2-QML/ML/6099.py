"""Hybrid classical model combining classification and regression tasks.

The architecture extends the original Quantum‑NAT CNN with a second
fully‑connected head for regression, enabling multi‑task learning
and richer diagnostics during experimental comparison with the
quantum implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridNATModel(nn.Module):
    """CNN backbone with separate classification and regression heads."""

    def __init__(self, num_features: int = 16) -> None:
        super().__init__()
        # CNN encoder (identical to the original Quantum‑NAT feature extractor)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Classification head: 4‑way output
        self.class_head = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4),
        )
        # Regression head: single continuous output
        self.reg_head = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a tuple (class_logits, reg_output) where:
          * class_logits is batch‑size × 4 logits, batch‑normed
          * reg_output is batch‑size continuous values
        """
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        cls_out = self.class_head(flat)
        reg_out = self.reg_head(flat)
        return self.norm(cls_out), reg_out.squeeze(-1)


__all__ = ["HybridNATModel"]
