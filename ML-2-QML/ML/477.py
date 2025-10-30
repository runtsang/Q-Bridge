"""Enhanced classical Quanvolution with multi‑branch feature extraction.

The model replaces the single 2‑D convolution with a small residual
network that learns two parallel pathways – a 3×3 and a 5×5 kernel –
before merging.  The resulting richer representation is then
flattened and passed through a linear head.  The architecture is
fully compatible with the MNIST input shape used in the seed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Quanvolution__gen110(nn.Module):
    """Classical multi‑branch quanvolutional network."""

    def __init__(self) -> None:
        super().__init__()
        # Branch 1: 3×3 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        # Branch 2: 5×5 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        # Merge and refine
        self.merge = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.25),
        )
        # Classifier head
        self.classifier = nn.Linear(16 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Two parallel pathways
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        # Concatenate along channel dimension
        out = torch.cat([out1, out2], dim=1)
        # Merge and refine
        out = self.merge(out)
        # Flatten
        out = out.view(bsz, -1)
        # Log‑softmax logits
        logits = self.classifier(out)
        return F.log_softmax(logits, dim=-1)


__all__ = ["Quanvolution__gen110"]
