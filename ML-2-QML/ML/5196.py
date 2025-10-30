"""Hybrid classical classifier combining conv, sampler, and fully connected layers.

The `build_classifier_circuit` function returns a torch.nn.Module network along with
encoding metadata, weight sizes, and observable indices.  The design allows optional
inclusion of a quanvolution filter, a sampler QNN, and a fully‑connected layer,
enabling a smooth transition from purely classical to hybrid workflows.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QuanvolutionFilter(nn.Module):
    """Simple 2×2 image patch convolution that emulates a quantum kernel."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)


class SamplerModule(nn.Module):
    """Classical sampler network mimicking the quantum sampler."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)


class FCL(nn.Module):
    """Fully connected layer that can be replaced by a quantum counterpart."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation


class QuantumClassifierModelGen:
    """Hybrid classical classifier that optionally stacks quanvolution, sampler and FCL."""
    def __init__(self,
                 num_features: int,
                 depth: int,
                 use_quanvolution: bool = True,
                 use_sampler: bool = True,
                 use_fcl: bool = True) -> None:
        self.num_features = num_features
        self.depth = depth
        self.use_quanvolution = use_quanvolution
        self.use_sampler = use_sampler
        self.use_fcl = use_fcl
        self.network = self._build_network()
        self.encoding = list(range(num_features))
        self.weight_sizes = self._compute_weight_sizes()
        self.observables = list(range(2))

    def _build_network(self) -> nn.Module:
        layers: list[nn.Module] = []
        in_dim = self.num_features
        if self.use_quanvolution:
            layers.append(QuanvolutionFilter())
            in_dim = 4 * 14 * 14
        if self.use_sampler:
            layers.append(SamplerModule())
            in_dim = 2
        if self.use_fcl:
            layers.append(FCL(1))
            in_dim = 1
        layers.append(nn.Linear(in_dim, 2))
        return nn.Sequential(*layers)

    def _compute_weight_sizes(self) -> list[int]:
        sizes: list[int] = []
        for module in self.network:
            if isinstance(module, nn.Linear):
                sizes.append(module.weight.numel() + module.bias.numel())
            elif isinstance(module, FCL):
                sizes.append(module.linear.weight.numel() + module.linear.bias.numel())
        return sizes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_classifier_circuit(num_features: int,
                             depth: int,
                             use_quanvolution: bool = True,
                             use_sampler: bool = True,
                             use_fcl: bool = True) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Return a classifier network and metadata."""
    model = QuantumClassifierModelGen(num_features, depth,
                                      use_quanvolution, use_sampler, use_fcl)
    return model.network, model.encoding, model.weight_sizes, model.observables


__all__ = [
    "build_classifier_circuit",
    "QuantumClassifierModelGen",
    "QuanvolutionFilter",
    "SamplerModule",
    "FCL",
]
