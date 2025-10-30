"""Hybrid classical fraud detection classifier.

Combines the photonic‑inspired layer construction from the original
`FraudDetection.py` with the feed‑forward head of the quantum classifier
to form a two‑stage model suitable for fraud detection experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected layer used in the classical branch."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to a symmetric interval."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Instantiate a single `nn.Module` from the photonic parameters."""
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi], [params.squeeze_r[0], params.squeeze_r[1]]],
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
        def __init__(self):
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(inputs))
            return out * self.scale + self.shift

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model that mirrors the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class FraudDetectionClassifier(nn.Module):
    """Hybrid classical fraud‑detection classifier.

    The backbone consists of a stack of photonic‑inspired layers created
    from `FraudLayerParameters`.  A lightweight feed‑forward head,
    identical to the quantum classifier reference, produces a 2‑class
    output.  The class exposes `encoding_indices` and `weight_sizes`
    for API parity with the quantum counterpart.
    """
    def __init__(self,
                 fraud_params: Iterable[FraudLayerParameters],
                 head_features: int = 2) -> None:
        super().__init__()
        params_iter = iter(fraud_params)
        input_params = next(params_iter)
        self.backbone = build_fraud_detection_program(input_params, params_iter)
        self.head = nn.Sequential(
            nn.Linear(2, head_features),
            nn.ReLU(),
            nn.Linear(head_features, 2),
        )
        self.weight_sizes = self._compute_weight_sizes()

    def _compute_weight_sizes(self) -> List[int]:
        return [p.numel() for p in self.parameters()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.head(x)

    @property
    def encoding_indices(self) -> List[int]:
        """Indices used for encoding – kept for API parity."""
        return list(range(self.backbone[0].linear.in_features))

    @property
    def observables(self) -> List[str]:
        """No classical observables; placeholder for API compatibility."""
        return []

__all__ = ["FraudLayerParameters", "FraudDetectionClassifier"]
