"""Hybrid fraud detection model combining classical neural network layers with a quantum-inspired variational classifier.

The module exposes a single class `FraudDetectionHybrid` that encapsulates:
  * a classical feature extractor built from photonic-inspired `FraudLayerParameters`;
  * a standard feed‑forward classifier mirroring the quantum variant;
  * a convenience `forward` method for end‑to‑end inference.

The hybrid design allows the classical part to learn robust feature embeddings
while the classifier can be swapped with a quantum backend if desired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List

import torch
from torch import nn


# --------------------------------------------------------------------------- #
# Classical photonic‑style layer definitions
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
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
            return outputs * self.scale + self.shift

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
# Classical classifier mirroring the quantum interface
# --------------------------------------------------------------------------- #
def build_classifier_circuit(
    num_features: int,
    depth: int,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Construct a feed‑forward classifier and metadata similar to the quantum variant."""
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


# --------------------------------------------------------------------------- #
# Hybrid model exposing a unified interface
# --------------------------------------------------------------------------- #
class FraudDetectionHybrid:
    """
    Hybrid fraud detection pipeline that combines a classical feature extractor
    with a classifier mirroring the quantum architecture.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Depth of the feed‑forward classifier.
    input_params : FraudLayerParameters
        Parameters for the initial photonic‑style layer.
    layers : Iterable[FraudLayerParameters]
        Parameters for subsequent photonic‑style layers.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        self.feature_extractor = build_fraud_detection_program(input_params, layers)
        self.classifier, self.encoding, self.weight_sizes, self.observables = (
            build_classifier_circuit(num_features, depth)
        )
        self.model = nn.Sequential(self.feature_extractor, self.classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """End‑to‑end inference."""
        return self.model(x)


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "build_classifier_circuit",
    "FraudDetectionHybrid",
]
