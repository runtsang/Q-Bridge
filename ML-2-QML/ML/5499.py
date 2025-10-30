from __future__ import annotations

import torch
from torch import nn
import numpy as np
from typing import Iterable, Tuple


class ConvGen396(nn.Module):
    """
    Classical convolutional filter that can be used as a drop‑in replacement
    for the quanvolution layer.  It exposes the same ``run`` API as the
    original ``Conv.py`` seed and optionally clips its weights to keep
    them in a numerically stable regime.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        clip_weights: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.clip_weights = clip_weights

        # 1‑channel convolution – the filter is learnable
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        if self.clip_weights:
            self._clip_weights()

    def _clip_weights(self) -> None:
        """Clamp the convolution weights to the range [-5, 5]."""
        with torch.no_grad():
            self.conv.weight.clamp_(-5.0, 5.0)
            self.conv.bias.clamp_(-5.0, 5.0)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning a single scalar activation.
        """
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

    def run(self, data: np.ndarray) -> float:
        """
        Compatibility wrapper matching the legacy ``Conv.run`` signature.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        return self.forward(tensor).item()

    @staticmethod
    def build_classifier(
        num_features: int,
        depth: int,
        hidden: int = 64,
    ) -> nn.Sequential:
        """
        Small fully‑connected classifier that can be paired with the filter.
        Mirrors the structure used in the hybrid binary‑classification example.
        """
        layers: list[torch.nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 2))
        return nn.Sequential(*layers)


def Conv() -> ConvGen396:
    """Factory function matching the legacy API."""
    return ConvGen396()
