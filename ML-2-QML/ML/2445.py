"""Hybrid classical‑quantum fraud detection and classification model.

This module builds a PyTorch model that first processes the input through a
photonic‑inspired fraud‑detection sub‑network and then feeds the result into a
parameterised feed‑forward classifier.  The design mirrors the structure of the
original `QuantumClassifierModel.py` and `FraudDetection.py` seeds while
adding a clear hybrid interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  Photonic‑inspired fraud‑detection layer
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clamp a value to a symmetric bound."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a single fraud‑detection layer with optional clipping."""
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


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch fraud‑detection program."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
#  Classical classifier
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Construct a feed‑forward classifier and metadata."""
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
#  Hybrid interface
# --------------------------------------------------------------------------- #
def build_hybrid_classifier(
    num_features: int,
    depth: int,
    fraud_params: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """
    Combine the fraud‑detection sub‑network with the classical classifier.

    Parameters
    ----------
    num_features : int
        Input dimensionality expected by the classifier (should match the
        output of the fraud sub‑network).
    depth : int
        Number of hidden layers in the classifier.
    fraud_params : Iterable[FraudLayerParameters]
        Parameters for the fraud‑detection layers.  The first element is used
        as the input layer; the rest are stacked as hidden layers.
    """
    if not fraud_params:
        raise ValueError("At least one FraudLayerParameters instance is required.")

    fraud_model = build_fraud_detection_program(fraud_params[0], fraud_params[1:])
    classifier_model, _, _, _ = build_classifier_circuit(num_features, depth)
    return nn.Sequential(fraud_model, classifier_model)


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "build_classifier_circuit",
    "build_hybrid_classifier",
]
