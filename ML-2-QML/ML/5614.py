"""Hybrid classical regression model inspired by multiple seed examples.

This module provides a purely classical implementation that mimics quantum
behaviour using stand‑in layers: a convolution filter, a fully‑connected
quantum‑like layer, and a sampler network.  The design follows the
structure of the original `QuantumRegression.py` but replaces all quantum
operations with efficient PyTorch primitives while still exposing the
same public API for downstream experiments.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import nn
from torch.utils.data import Dataset
from typing import Iterable

# ----------------------------------------------------------------------
# Data generation – identical to the quantum seed
# ----------------------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# ----------------------------------------------------------------------
# Classical stand‑ins for quantum layers
# ----------------------------------------------------------------------
def Conv():
    """Return a convolution filter that emulates a quantum filter."""
    class ConvFilter(nn.Module):
        def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.threshold = threshold
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        def run(self, data) -> float:
            tensor = torch.as_tensor(data, dtype=torch.float32)
            tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
            logits = self.conv(tensor)
            activations = torch.sigmoid(logits - self.threshold)
            return activations.mean().item()

    return ConvFilter()

def SamplerQNN():
    """A lightweight sampler network that mimics a quantum sampler."""
    class SamplerModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 4),
                nn.Tanh(),
                nn.Linear(4, 2),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return torch.nn.functional.softmax(self.net(inputs), dim=-1)

    return SamplerModule()

def FCL():
    """Classical fully‑connected layer that mimics a quantum fully‑connected layer."""
    class FullyConnectedLayer(nn.Module):
        def __init__(self, n_features: int = 1) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach().numpy()

    return FullyConnectedLayer()

# ----------------------------------------------------------------------
# Hybrid classical regression model
# ----------------------------------------------------------------------
class HybridRegressor(nn.Module):
    """
    A hybrid regression model that stitches together classical stand‑ins
    for quantum components.  It preserves the API of the original
    quantum regression example while providing an entirely classical
    implementation that is suitable for rapid prototyping and unit
    testing.
    """

    def __init__(self, num_features: int, kernel_size: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = Conv()          # classical convolution filter
        self.fcl = FCL()            # classical fully‑connected layer
        self.sampler = SamplerQNN() # classical sampler network
        self.head = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of feature vectors with shape (batch, num_features).

        Returns
        -------
        torch.Tensor
            Predicted scalar for each sample in the batch.
        """
        batch = x.shape[0]
        # reshape to a square kernel (kernel_size x kernel_size)
        data = x.view(batch, self.kernel_size, self.kernel_size)

        # Convolution filter – returns a scalar per sample
        conv_out = torch.tensor([self.conv.run(d.cpu().numpy()) for d in data])

        # Fully‑connected layer – returns a scalar per sample
        fcl_out = torch.tensor([self.fcl.run([float(v)]) for v in conv_out])

        # Prepare sampler input: [fcl_out, 1 - fcl_out]
        sampler_input = torch.stack([torch.tensor([v, 1.0 - v]) for v in fcl_out], dim=1)
        sampler_out = self.sampler(sampler_input)  # softmax probabilities

        # Take probability of class 0 as the feature for the head
        probs = sampler_out[:, 0]

        return self.head(probs.unsqueeze(-1)).squeeze(-1)

__all__ = ["HybridRegressor", "RegressionDataset", "generate_superposition_data"]
