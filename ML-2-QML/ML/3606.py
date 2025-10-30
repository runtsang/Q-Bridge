"""Hybrid classical estimator that uses a convolutional filter and a feed‑forward regressor.

The class can be used as a drop‑in replacement for the original EstimatorQNN, but
adds a 2‑D convolutional preprocessing stage inspired by the Conv seed."""
import torch
from torch import nn
import numpy as np

class ConvFilter(nn.Module):
    """2‑D convolutional filter inspired by the Conv seed."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data expected shape (..., kernel_size, kernel_size)
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[-2, -1])  # mean over spatial dims

class EstimatorQNN(nn.Module):
    """Classical estimator that first applies a ConvFilter and then a regression head."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size, threshold)
        self.regressor = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            4‑D tensor of shape (batch, 1, kernel_size, kernel_size)

        Returns
        -------
        torch.Tensor
            Predicted scalar per example
        """
        conv_out = self.conv(inputs).unsqueeze(-1)  # shape (batch, 1)
        return self.regressor(conv_out)

__all__ = ["EstimatorQNN"]
