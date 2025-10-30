"""Hybrid classical classifier built from feed‑forward layers and optional fraud‑style custom layers.

The implementation merges the simple depth‑wise architecture from the original
`QuantumClassifierModel` with the parameterised `FraudLayerParameters` used in
the photonic fraud‑detection example.  The returned tuple mirrors the quantum
interface so that the same helper functions can be called from either side of
the hybrid stack.

The API matches the anchor file:

    build_classifier_circuit(num_features: int, depth: int,
                            custom_layers: Optional[Iterable[FraudLayerParameters]] = None) -> Tuple[nn.Sequential,
                                                                                                  Iterable[int],
                                                                                                  List[int],
                                                                                                  List[int]]

The function returns:

    * the PyTorch model,
    * an encoding list (indexes of input features),
    * the number of trainable parameters per layer,
    * a list of output class indices (``[0, 1]``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Fraud‑style layer definition – copied from the photonic example
# --------------------------------------------------------------------------- #
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
    """Clip a scalar to the interval ``[-bound, bound]``."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Construct a linear → tanh → affine‑scale layer from *params*."""
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
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
# Main classifier builder – combines simple feed‑forward and fraud layers
# --------------------------------------------------------------------------- #
def build_classifier_circuit(
    num_features: int,
    depth: int,
    *,
    custom_layers: Optional[Iterable[FraudLayerParameters]] = None,
) -> Tuple[nn.Sequential, Iterable[int], List[int], List[int]]:
    """
    Construct a PyTorch classifier that can be used interchangeably with the
    quantum helper interface.

    Parameters
    ----------
    num_features:
        Number of input features / qubits.
    depth:
        Number of hidden layers when ``custom_layers`` is None.
    custom_layers:
        Optional iterable of :class:`FraudLayerParameters`.  If supplied the
        network consists of one fraud‑style layer for each element followed by
        a final linear output layer.  This mirrors the photonic fraud‑detection
        network and demonstrates how domain‑specific layers can be embedded in
        a classical pipeline.

    Returns
    -------
    model:
        ``torch.nn.Sequential`` instance.
    encoding:
        List of feature indices used by the quantum variant.
    weight_sizes:
        Number of trainable parameters per layer of the network.
    observables:
        Output class indices (``[0, 1]``) to keep the quantum interface
        compatible.
    """
    # --------------------------------------------------------------------- #
    # Build the network
    # --------------------------------------------------------------------- #
    if custom_layers is None:
        layers: List[nn.Module] = []
        in_dim = num_features
        weight_sizes: List[int] = []
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features)
            layers.extend([linear, nn.ReLU()])
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())
        model = nn.Sequential(*layers)
    else:
        # Build fraud‑style layers
        modules: List[nn.Module] = []
        # First layer is not clipped to expose raw photonic parameters
        modules.append(_layer_from_params(next(iter(custom_layers)), clip=False))
        for layer in custom_layers:
            modules.append(_layer_from_params(layer, clip=True))
        modules.append(nn.Linear(2, 1))
        model = nn.Sequential(*modules)
        # Compute weight sizes for reporting
        weight_sizes = [
            sum(p.numel() for p in m.parameters())
            for m in modules
        ]

    encoding = list(range(num_features))
    observables = [0, 1]

    return model, encoding, weight_sizes, observables


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "build_classifier_circuit",
]
