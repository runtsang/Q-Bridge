"""Hybrid classical estimator combining convolution, MLP, and classification."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import sigmoid


class ConvFilter(nn.Module):
    """Drop‑in convolutional filter emulating a quantum quanvolution layer."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        # Single‑channel 2‑D convolution (1→1)
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data: Tensor of shape (batch, 1, H, W) containing grayscale images.

        Returns:
            Tensor of shape (batch,) containing the mean activation after sigmoid
            thresholding.  The value is suitable as a scalar feature for the
            downstream network.
        """
        logits = self.conv(data)
        activations = sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3])


class HybridEstimatorQNN(nn.Module):
    """
    Classical estimator that:
      * Convolves the input image with ConvFilter.
      * Passes the flattened feature through a configurable MLP.
      * Appends a 2‑class classification head.

    The architecture mirrors the quantum counterpart in terms of
    depth and layer counts, enabling direct comparison.
    """

    def __init__(
        self,
        input_dim: int = 2,
        depth: int = 2,
        kernel_size: int = 2,
        conv_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size, conv_threshold)

        # Feed‑forward regression network
        layers = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, in_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

        # Classification head
        self.classifier = nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        If *x* has 4 dimensions, it is treated as a batch of images and
        passed through the ConvFilter.  The resulting scalar per sample
        is then fed to the MLP and classification head.

        Returns:
            logits: Tensor of shape (batch, 2)
        """
        if x.dim() == 4:
            x = self.conv(x).unsqueeze(1)  # (batch, 1)
        out = self.network(x)
        logits = self.classifier(out)
        return logits

    def weight_sizes(self) -> list[int]:
        """Return the number of learnable parameters for each linear layer."""
        sizes = []
        for m in self.modules():
            if isinstance(m, nn.Linear):
                sizes.append(m.weight.numel() + m.bias.numel())
        return sizes


def EstimatorQNN() -> HybridEstimatorQNN:
    """Factory function matching the original EstimatorQNN interface."""
    return HybridEstimatorQNN()


__all__ = ["HybridEstimatorQNN", "EstimatorQNN"]
