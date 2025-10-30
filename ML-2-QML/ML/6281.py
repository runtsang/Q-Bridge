"""Enhanced classical CNN with dual-task heads and optional feature fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class QFCModel(nn.Module):
    """Classical CNN with optional feature‑fusion and two classification heads.

    The network processes a single‑channel image.  After the convolutional
    backbone it can optionally fuse a learnable projection of the
    intermediate feature map before the fully‑connected layers.  Two
    heads are exposed: ``head1`` outputs 4 logits (original task) and
    ``head2`` outputs 2 logits (auxiliary task).  The heads share the
    same hidden representation, which allows joint training.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 8,
        fusion: bool = False,
        hidden_dim: int = 64,
        head1_dim: int = 4,
        head2_dim: int = 2,
    ) -> None:
        super().__init__()
        self.fusion = fusion

        # Convolutional backbone
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Optional fusion layer: a 1×1 conv that projects the
        # intermediate feature map to a lower‑dimensional space.
        if fusion:
            self.fusion_layer = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)
        else:
            self.fusion_layer = nn.Identity()

        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear((base_channels * 2) * 7 * 7, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Two classification heads
        self.head1 = nn.Linear(hidden_dim, head1_dim)
        self.head2 = nn.Linear(hidden_dim, head2_dim)

        self.norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits for head1 and head2."""
        bsz = x.shape[0]
        feats = self.features(x)
        fused = self.fusion_layer(feats)
        flat = fused.view(bsz, -1)
        h = self.fc(flat)
        h = self.norm(h)

        out1 = self.head1(h)
        out2 = self.head2(h)
        return out1, out2

__all__ = ["QFCModel"]
