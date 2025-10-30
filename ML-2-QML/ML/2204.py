import torch
from torch import nn
import numpy as np


class ConvFilter(nn.Module):
    """
    Classical 2×2 convolutional filter that emulates the quantum filter
    used in the QML counterpart.  The filter is a single‑channel
    convolution followed by a sigmoid activation and a mean reduction
    to a scalar feature.  The threshold parameter is kept for
    compatibility with the original Conv.py seed.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, H, W).  Only the top‑left
            2×2 patch is used to mimic the quantum filter's local
            receptive field.

        Returns
        -------
        torch.Tensor
            Scalar feature per batch element, shape (batch, 1).
        """
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3]).unsqueeze(-1)


class EstimatorQNN(nn.Module):
    """
    Classical feed‑forward regressor that mirrors the EstimatorQNN
    architecture but uses the ConvFilter as a drop‑in replacement for
    the quantum quanvolution.  The network is fully differentiable
    and can be trained with standard PyTorch optimizers.
    """
    def __init__(self, hidden_dims=[8, 4]) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size=2, threshold=0.0)
        self.fc = nn.Sequential(
            nn.Linear(1, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classical estimator.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, H, W).  The convolution
            operates on the top‑left 2×2 patch; the resulting scalar
            is fed into the fully‑connected layers.

        Returns
        -------
        torch.Tensor
            Regression output of shape (batch, 1).
        """
        # Extract top‑left 2×2 patch
        patch = x[:, :, :2, :2]
        conv_out = self.conv(patch)
        return self.fc(conv_out)


__all__ = ["EstimatorQNN"]
