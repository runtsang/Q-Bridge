"""Hybrid quantum‑classical classifier with a classical backbone and optional photonic fraud‑detection sub‑module.

The public API mirrors the classical build‑function from the seed, but the focus is on
- a multi‑layer perceptron backbone with configurable depth and hidden size,
- an optional fraud‑detection sub‑module built from parametrised layers,
- and metadata that can be shared with the quantum counterpart for joint training.

The module exposes:
- `FraudLayerParameters` – dataclass describing a single fraud layer,
- `build_fraud_detection_program` – builds a PyTorch sequential model,
- `build_classifier_circuit` – builds the full hybrid classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected fraud layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to the range [-bound, bound]."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a single fraud layer as a PyTorch module."""
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
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential fraud‑detection model mirroring the photonic structure."""
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


def build_classifier_circuit(
    num_features: int,
    depth: int,
    hidden_size: int | None = None,
    use_fraud: bool = False,
    fraud_input: FraudLayerParameters | None = None,
    fraud_layers: Iterable[FraudLayerParameters] | None = None,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Build a hybrid classical classifier.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of hidden layers in the MLP backbone.
    hidden_size : int | None, optional
        Width of each hidden layer; defaults to ``num_features``.
    use_fraud : bool, optional
        If ``True`` prepend a fraud‑detection sub‑module at the front.
    fraud_input : FraudLayerParameters | None, optional
        Parameters for the first fraud layer (required if ``use_fraud``).
    fraud_layers : Iterable[FraudLayerParameters] | None, optional
        Parameters for the remaining fraud layers.

    Returns
    -------
    model : nn.Module
        The fully constructed PyTorch model.
    encoding : Iterable[int]
        Indices mapping each input feature to a position in the model.
    weight_sizes : Iterable[int]
        Number of trainable parameters in each layer, useful for joint optimisation.
    observables : List[int]
        Integer indices of the output logits (0 and 1 for binary classification).
    """
    if hidden_size is None:
        hidden_size = num_features

    layers: List[nn.Module] = []

    # Optional fraud‑detection sub‑module
    if use_fraud:
        if fraud_input is None or fraud_layers is None:
            raise ValueError("``fraud_input`` and ``fraud_layers`` must be provided when ``use_fraud`` is True")
        fraud_module = build_fraud_detection_program(fraud_input, fraud_layers)
        layers.append(fraud_module)

    # MLP backbone
    in_dim = num_features
    for _ in range(depth):
        linear = nn.Linear(in_dim, hidden_size)
        layers.append(linear)
        layers.append(nn.ReLU())
        in_dim = hidden_size

    # Output layer
    head = nn.Linear(in_dim, 2)
    layers.append(head)

    model = nn.Sequential(*layers)

    # Metadata
    encoding = list(range(num_features))
    weight_sizes = [m.weight.numel() + m.bias.numel() for m in model if isinstance(m, nn.Linear)]
    observables = [0, 1]

    return model, encoding, weight_sizes, observables
