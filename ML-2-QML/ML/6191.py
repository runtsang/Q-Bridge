"""
HybridNet – Classical implementation of a convolution + fully‑connected layer.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn
import numpy as np


class ConvFilter(nn.Module):
    """
    A lightweight 2‑D convolutional filter that mimics the behaviour of a
    quanvolution layer.  The filter is parameterised by a learnable weight
    matrix and an optional threshold that controls activation clipping.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: np.ndarray, thetas: Iterable[float] | None = None) -> float:
        """
        Forward pass of the filter.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).
        thetas : Iterable[float] | None
            Optional parameters that overwrite the convolutional weight.
            If provided, the weight tensor is reshaped to match the kernel.

        Returns
        -------
        float
            Mean activation after sigmoid clipping.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)

        if thetas is not None:
            weight = torch.tensor(list(thetas), dtype=torch.float32)
            weight = weight.view(1, 1, self.kernel_size, self.kernel_size)
            self.conv.weight.data = weight

        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()


class FullyConnectedLayer(nn.Module):
    """
    A simple linear layer that emulates the final quantum fully‑connected
    circuit.  Parameters are supplied as a 1‑D iterable.
    """
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Forward pass of the fully‑connected layer.

        Parameters
        ----------
        thetas : Iterable[float]
            Parameters that overwrite the linear weights and bias.

        Returns
        -------
        np.ndarray
            The scalar output wrapped in a 1‑element array.
        """
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        # Overwrite weights & bias
        self.linear.weight.data = torch.tensor([[values[0]]], dtype=torch.float32)
        self.linear.bias.data = torch.tensor([values[1]], dtype=torch.float32)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()


class HybridNet(nn.Module):
    """
    Hybrid network that chains a convolutional filter and a fully‑connected
    layer.  The API mirrors the quantum counterpart by exposing a `run`
    method that accepts separate parameter sets for each block.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 n_features: int = 1,
                 threshold: float = 0.0) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size, threshold)
        self.fc   = FullyConnectedLayer(n_features)

    def run(self,
            data: np.ndarray,
            conv_thetas: Iterable[float] | None = None,
            fc_thetas: Iterable[float] = (1.0, 0.0)) -> np.ndarray:
        """
        Forward pass through the hybrid network.

        Parameters
        ----------
        data : np.ndarray
            Input image patch of shape (kernel_size, kernel_size).
        conv_thetas : Iterable[float] | None
            Parameters for the convolutional block.
        fc_thetas : Iterable[float]
            Parameters for the fully‑connected block.

        Returns
        -------
        np.ndarray
            The scalar output of the fully‑connected layer.
        """
        conv_out = self.conv.run(data, conv_thetas)
        # Reshape conv output to match the linear layer's expected input shape
        fc_input = [conv_out, 0.0]  # dummy bias term
        return self.fc.run(fc_input + list(fc_thetas))


__all__ = ["HybridNet"]
