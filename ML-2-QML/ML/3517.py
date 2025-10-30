"""Hybrid classical classifier mirroring the quantum interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List
import random

import torch
import torch.nn as nn


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a linear layer with optional clipping of weights and biases."""
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


def _random_params() -> FraudLayerParameters:
    """Generate a random FraudLayerParameters instance."""
    return FraudLayerParameters(
        bs_theta=random.uniform(-5, 5),
        bs_phi=random.uniform(-5, 5),
        phases=(random.uniform(-5, 5), random.uniform(-5, 5)),
        squeeze_r=(random.uniform(-5, 5), random.uniform(-5, 5)),
        squeeze_phi=(random.uniform(-5, 5), random.uniform(-5, 5)),
        displacement_r=(random.uniform(-5, 5), random.uniform(-5, 5)),
        displacement_phi=(random.uniform(-5, 5), random.uniform(-5, 5)),
        kerr=(random.uniform(-1, 1), random.uniform(-1, 1)),
    )


class QuantumClassifierModel:
    """Classical feed‑forward classifier that mimics the interface of its quantum counterpart.

    The network is constructed by stacking a sequence of layers derived from
    :class:`FraudLayerParameters`.  The first layer is un‑clipped (free to learn
    arbitrary values), while subsequent layers are clipped to keep the
    parameters within a physically meaningful range, similar to the photonic
    implementation in the QML branch.
    """

    @staticmethod
    def build_classifier_circuit_from_params(
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        num_features: int = 2,
    ) -> Tuple[nn.Sequential, List[int], List[int], List[int]]:
        """
        Build a sequential PyTorch model that mirrors the layered structure used
        in the quantum implementation.

        Parameters
        ----------
        input_params
            Parameters for the first (un‑clipped) layer.
        layers
            Iterable of parameters for the remaining layers.
        num_features
            Dimensionality of the input feature vector.  Defaults to 2.
        """
        modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(layer, clip=True) for layer in layers)

        # Final classification head
        modules.append(nn.Linear(num_features, 2))

        network = nn.Sequential(*modules)

        # Metadata analogous to the QML circuit
        encoding = list(range(num_features))
        weight_sizes = [
            module.linear.weight.numel() + module.linear.bias.numel()
            for module in modules
            if isinstance(module, nn.Linear)
        ]
        observables = list(range(2))

        return network, encoding, weight_sizes, observables

    @staticmethod
    def build_classifier_circuit(
        num_features: int, depth: int
    ) -> Tuple[nn.Sequential, List[int], List[int], List[int]]:
        """
        Convenience wrapper that generates random parameters for a network
        of the requested depth.

        Parameters
        ----------
        num_features
            Dimensionality of the input feature vector.
        depth
            Total number of layers (including the first un‑clipped layer).
        """
        if depth < 1:
            raise ValueError("Depth must be at least 1.")
        input_params = _random_params()
        layers = [_random_params() for _ in range(depth - 1)]
        return QuantumClassifierModel.build_classifier_circuit_from_params(
            input_params, layers, num_features
        )


__all__ = ["FraudLayerParameters", "QuantumClassifierModel"]
