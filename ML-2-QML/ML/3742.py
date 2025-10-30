"""Hybrid classical classifier combining the classical and photonic‑style design patterns from the seeds."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
from torch import nn


class _ScaleShift(nn.Module):
    """Applies element‑wise scaling and shift to a tensor.

    The parameters are buffers so they are not optimized by default, mirroring the
    photonic layer implementation in the FraudDetection seed.
    """
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("scale", torch.ones(dim))
        self.register_buffer("shift", torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x * self.scale + self.shift


class HybridClassifier(nn.Module):
    """
    Classical feed‑forward network with per‑layer scaling/shift buffers.

    Parameters
    ----------
    num_features : int
        Dimension of the input feature vector.
    depth : int
        Number of hidden layers.
    clip : bool, optional
        If True, clamp weight and bias values to [-5, 5] after initialization,
        emulating the clipping logic in the FraudDetection seed.
    """
    def __init__(self, num_features: int, depth: int, clip: bool = True) -> None:
        super().__init__()
        self.num_features = num_features
        layers: List[nn.Module] = []

        in_dim = num_features
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            activation = nn.Tanh()
            scale_shift = _ScaleShift(num_features)

            if clip:
                with torch.no_grad():
                    linear.weight.clamp_(-5.0, 5.0)
                    linear.bias.clamp_(-5.0, 5.0)

            layers.append(nn.Sequential(linear, activation, scale_shift))
            in_dim = num_features

        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


def build_classifier_circuit(
    num_features: int,
    depth: int,
    clip: bool = True,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct the classical classifier and return auxiliary metadata.

    The returned metadata mirrors the quantum interface: an encoding list,
    a list of weight‑size counts, and a list of observable indices.
    """
    # Instantiate the model
    model = HybridClassifier(num_features, depth, clip=clip)

    # Encoding: we simply expose the indices of the input features
    encoding = list(range(num_features))

    # Weight sizes: iterate over all parameters of the network
    weight_sizes: List[int] = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            weight_sizes.append(module.weight.numel() + module.bias.numel())

    # Observables: for a binary classifier we expose two output neurons
    observables = [0, 1]
    return model, encoding, weight_sizes, observables


__all__ = ["HybridClassifier", "build_classifier_circuit"]
