"""Hybrid convolution and fully connected layer with classical implementation."""

from __future__ import annotations

import torch
from torch import nn
import numpy as np

class HybridConvFC(nn.Module):
    """
    Combines a learnable convolutional filter with a threshold gate and a
    fully connected layer.  The convolution outputs a scalar activation that
    is then linearly combined by the dense layer.  This mirrors the
    architecture of the quantum filter followed by a quantum fully connected
    layer, but remains fully classical.

    Scaling paradigm: combination of convolution + dense.
    """

    def __init__(self,
                 kernel_size: int = 3,
                 conv_threshold: float = 0.0,
                 fc_output_dim: int = 1,
                 fc_features: int = 10) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_threshold = conv_threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        # Flattened convolution output fed into a linear layer
        self.fc = nn.Linear(kernel_size * kernel_size, fc_output_dim)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            data (torch.Tensor): 2D array of shape (kernel_size, kernel_size)

        Returns:
            torch.Tensor: scalar output
        """
        tensor = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        gated = torch.sigmoid(logits - self.conv_threshold)
        flat = gated.view(1, -1)
        out = self.fc(flat)
        return out

    def run(self, data: np.ndarray) -> float:
        """
        Convenience method that accepts a NumPy array and returns a scalar.
        """
        with torch.no_grad():
            tensor = torch.as_tensor(data, dtype=torch.float32)
            return self.forward(tensor).item()


__all__ = ["HybridConvFC"]
