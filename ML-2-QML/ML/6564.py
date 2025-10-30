"""Hybrid convolution + fully‑connected layer implemented in PyTorch.

The class mimics a quanvolution followed by a parameterised fully‑connected
quantum layer, but is fully classical.  It is a drop‑in replacement for the
original Conv.py and FCL.py modules while exposing a unified interface.

Usage
-----
>>> layer = HybridLayer(kernel_size=3, threshold=0.1)
>>> out = layer.run(data, thetas=[0.5])
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

class HybridLayer(nn.Module):
    """
    A single PyTorch module that first applies a 2‑D convolution and then
    performs a parameterised fully‑connected transformation.  The interface
    matches the original Conv.py / FCL.py modules: ``run(data, thetas)``.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        n_fc_features: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        # The fully‑connected layer only consumes a single scalar that
        # represents the mean of the supplied thetas.
        self.fc = nn.Linear(1, 1)

    def run(self, data: np.ndarray, thetas: list[float] | None = None) -> float:
        """
        Forward pass that emulates a quanvolution followed by a fully‑connected
        quantum layer.

        Args:
            data: 2‑D array of shape (kernel_size, kernel_size).
            thetas: Iterable of floats used to parameterise the fully‑connected
                    mapping.  If omitted a neutral value of 0.0 is used.

        Returns:
            A single scalar output.
        """
        # Convolutional part
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        conv_mean = activations.mean().item()

        # Fully‑connected part
        if thetas is None:
            thetas = [0.0]
        theta_mean = torch.tensor([np.mean(thetas)], dtype=torch.float32)
        fc_out = torch.tanh(self.fc(theta_mean)).item()

        # Simple fusion of the two signals
        return (conv_mean + fc_out) / 2.0

__all__ = ["HybridLayer"]
