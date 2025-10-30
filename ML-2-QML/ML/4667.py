"""Hybrid fraud detection model combining classical fully connected layers, a quantum-inspired filter, and a simple sampler.

This module implements a purely classical neural network that mimics the structure of the quantum counterparts from the reference seeds.  The `QuantumFilter` layer simulates the effect of a 2‑qubit quantum kernel by applying a random linear embedding to each 2×2 patch of the input tensor.  The `SamplerNetwork` reproduces the behaviour of the `SamplerQNN` quantum circuit using a tiny feed‑forward net that outputs a probability distribution via soft‑max.  The final `FraudDetectionHybridModel` stitches these components together and ends with a linear classifier.

The design intentionally mirrors the quantum version in the QML module so that the same high‑level API can be used interchangeably in a classical or quantum setting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable


@dataclass
class LayerParams:
    """Container for layer hyper‑parameters used in the classical model."""
    weight_init: float = 1.0
    bias_init: float = 0.0


class QuantumFilter(nn.Module):
    """
    Classical approximation of a 2‑qubit quantum kernel applied to 2×2 patches.
    The filter is a learnable linear layer per patch that mimics the action of a
    variational circuit with a fixed random seed.  The weights are initialized
    from a normal distribution and are not clipped to keep the behaviour
    comparable to the quantum counterpart.
    """
    def __init__(self, patch_size: int = 2, out_channels: int = 4) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, H, W)
        patches = self.conv(x)
        return patches.view(x.size(0), -1)


class SamplerNetwork(nn.Module):
    """
    Lightweight classifier that mimics the behaviour of a quantum sampler.
    The network consists of a single hidden layer followed by a soft‑max
    output that yields a probability distribution over two classes.
    """
    def __init__(self, in_features: int = 2, hidden: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)


class FraudDetectionHybridModel(nn.Module):
    """
    End‑to‑end classical model that combines the quantum‑filter,
    sampler, and a final linear classifier.

    Parameters
    ----------
    filter_out : int
        Number of output channels produced by the `QuantumFilter`.
    sampler_hidden : int
        Size of the hidden layer in the `SamplerNetwork`.
    num_classes : int
        Number of target classes (default: 2 for fraud / non‑fraud).
    """
    def __init__(
        self,
        filter_out: int = 4,
        sampler_hidden: int = 4,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.quantum_filter = QuantumFilter(out_channels=filter_out)
        self.sampler = SamplerNetwork(in_features=filter_out, hidden=sampler_hidden)
        self.classifier = nn.Linear(filter_out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected shape: (batch, 1, H, W) – e.g. 28x28 MNIST images
        features = self.quantum_filter(x)
        sampled = self.sampler(features)
        logits = self.classifier(sampled)
        return logits


__all__ = ["FraudDetectionHybridModel", "QuantumFilter", "SamplerNetwork", "LayerParams"]
