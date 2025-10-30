"""Hybrid Quanvolution Estimator implemented purely with PyTorch.

This module defines a classical neural network that mimics the structure of a
quantum convolutional filter followed by a quantum estimator network.  The
convolutional layer extracts local 2×2 patches and maps them to a 4‑dimensional
feature vector, exactly as in the original quanvolution example.  The
subsequent feed‑forward head is a lightweight fully‑connected network that
resembles the EstimatorQNN architecture (2→8→4→1) but is adapted for
classification.

The design allows a direct comparison between the classical approximation
and the quantum implementation that follows in :mod:`qml_code`.  The
class is fully differentiable and can be trained with standard PyTorch
optimizers.

Typical usage::

    model = HybridQuanvolutionEstimator()
    logits = model(x)   # ``x`` is a batch of 1‑channel 28×28 images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQuanvolutionEstimator(nn.Module):
    """Classical hybrid of a quanvolution filter and a feed‑forward head."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # 1-channel 28×28 -> 4‑channel 14×14 via 2×2 stride filter
        self.filter = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # Flatten: 4*14*14 = 784 features
        self.head = nn.Sequential(
            nn.Linear(4 * 14 * 14, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, num_classes).
        """
        features = self.filter(x).view(x.size(0), -1)
        logits = self.head(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionEstimator"]
