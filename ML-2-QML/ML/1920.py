"""Classical hybrid classifier using a pretrained ResNet backbone.

This module implements a binary classifier that replaces the shallow
dense head of the original seed with a ResNet‑18 backbone.  The
final linear layer outputs a single logit which is passed through a
sigmoid to obtain a probability.  The design is intentionally
simple yet powerful enough to serve as a strong baseline for
comparison with the quantum‑augmented counterpart.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNetHybridHead(nn.Module):
    """Linear head that maps ResNet features to a binary logit."""

    def __init__(self, in_features: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class HybridClassifier(nn.Module):
    """Pretrained ResNet‑18 backbone followed by a sigmoid head."""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False) -> None:
        super().__init__()
        self.backbone = models.resnet18(weights="DEFAULT" if pretrained else None)
        # Capture the feature dimension before replacing the final layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = ResNetHybridHead(in_features=in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features).squeeze(-1)
        probs = torch.sigmoid(logits)
        return torch.stack([probs, 1 - probs], dim=-1)


__all__ = ["HybridClassifier"]
