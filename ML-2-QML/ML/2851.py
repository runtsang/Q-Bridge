"""Hybrid classical neural network combining fully connected and convolution‑like blocks."""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple


class HybridFCQCNN(nn.Module):
    """
    A hybrid neural network that emulates a quantum convolutional neural network
    but implemented entirely in PyTorch.  The architecture is inspired by the
    QCNN model and the fully connected layer from the FCL example.
    """

    def __init__(
        self,
        input_features: int = 8,
        conv_features: Tuple[int,...] = (16, 16, 12, 8, 4, 4),
        output_features: int = 1,
        activation: nn.Module = nn.Tanh(),
    ) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(input_features, conv_features[0]),
            activation,
        )

        # Convolution‑like blocks
        self.conv_blocks = nn.ModuleList()
        in_feat = conv_features[0]
        for out_feat in conv_features[1:]:
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Linear(in_feat, out_feat),
                    activation,
                )
            )
            in_feat = out_feat

        # Final fully connected layer
        self.head = nn.Linear(in_feat, output_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the hybrid network."""
        x = self.feature_map(x)
        for block in self.conv_blocks:
            x = block(x)
        return torch.sigmoid(self.head(x))

    def run(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Convenience method that mimics the interface of the original FCL
        implementation.  It accepts a batch of input vectors and returns
        the network output.
        """
        return self.forward(input_data)


def HybridFCQCNNFactory() -> HybridFCQCNN:
    """Factory that returns a pre‑configured HybridFCQCNN instance."""
    return HybridFCQCNN()


__all__ = ["HybridFCQCNN", "HybridFCQCNNFactory"]
