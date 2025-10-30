"""Hybrid estimator combining classical convolutional preprocessing with a small MLP.

The class exposes a `run` method that accepts a 2‑D array of shape
(k, k) and returns a scalar regression output.  Internally it uses a
`ConvFilter` to emulate a quantum quanvolution, then feeds the resulting
scalar into an `EstimatorNN`.  The design is intentionally lightweight
and fully compatible with the original EstimatorQNN example.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Any

__all__ = ["HybridEstimator"]


class ConvFilter(nn.Module):
    """
    Classical 2‑D convolutional filter that emulates a quantum quanvolution.

    Parameters
    ----------
    kernel_size : int
        Size of the square kernel.  The filter is applied to a single‑channel
        input of shape (kernel_size, kernel_size).
    threshold : float
        Bias threshold applied before the sigmoid activation.  Values below
        this threshold are suppressed.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        # 1‑channel in → 1‑channel out
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: Any) -> float:
        """
        Apply the convolution and return the mean sigmoid activation.

        Parameters
        ----------
        data : array‑like
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            The mean activation over the single output channel.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()


class EstimatorNN(nn.Module):
    """
    Minimal feed‑forward regressor used after the convolutional filter.

    The network accepts a single scalar input (the output of `ConvFilter`)
    and produces a scalar regression value.
    """

    def __init__(self, hidden_sizes: tuple[int,...] = (8, 4)) -> None:
        super().__init__()
        layers = []
        in_features = 1
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.Tanh())
            in_features = h
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)


class HybridEstimator:
    """
    Hybrid classical estimator that combines a convolutional filter with a
    small MLP.  The public API mirrors the original EstimatorQNN example.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolutional kernel.
    threshold : float, default 0.0
        Threshold used in the convolutional filter.
    hidden_sizes : tuple[int,...], default (8, 4)
        Sizes of hidden layers in the MLP.
    device : str | torch.device, default 'cpu'
        Device on which the model runs.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        hidden_sizes: tuple[int,...] = (8, 4),
        device: Any = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.conv = ConvFilter(kernel_size, threshold).to(self.device)
        self.estimator = EstimatorNN(hidden_sizes).to(self.device)

    def run(self, data: Any) -> float:
        """
        Run a single forward pass.

        Parameters
        ----------
        data : array‑like
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Regression output.
        """
        feature = self.conv.run(data)
        # Convert feature to tensor and forward through the MLP
        input_tensor = torch.tensor([feature], device=self.device, dtype=torch.float32)
        with torch.no_grad():
            output = self.estimator(input_tensor)
        return output.item()

    # Compatibility shim for the original EstimatorQNN function
    @staticmethod
    def EstimatorQNN() -> "HybridEstimator":
        return HybridEstimator()
