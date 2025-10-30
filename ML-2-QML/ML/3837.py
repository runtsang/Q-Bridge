"""Hybrid classical classifier inspired by quantum and photonic fraud detection models.

The class exposes a feed‑forward network that mirrors the layered structure of the
photonic implementation while preserving the depth‑controlled architecture of the
incremental data‑uploading quantum classifier.  Parameters are provided through
``FraudLayerParameters``; if none are supplied the network is built with
trainable PyTorch linear layers only.

The implementation remains fully classical (NumPy / PyTorch) and can be
plugged into any standard training pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Optional, Tuple

import torch
from torch import nn, Tensor


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

        def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules: list[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class HybridQuantumClassifier:
    """
    Classical implementation of the hybrid quantum–photonic classifier.

    Parameters
    ----------
    num_features : int
        Number of input features (must be 2 to match the fraud‑detection
        photonic layer design).
    depth : int
        Number of stacked photonic layers.
    fraud_params : Iterable[FraudLayerParameters] | None, optional
        Explicit parameters for each photonic layer.  If omitted, the class
        generates a random sequence of parameters, suitable for training
        from scratch.
    """
    def __init__(
        self,
        num_features: int,
        depth: int,
        fraud_params: Optional[Iterable[FraudLayerParameters]] = None,
    ) -> None:
        self.num_features = num_features
        self.depth = depth
        if fraud_params is None:
            # Generate a random set of parameters for each layer
            fraud_params = [
                FraudLayerParameters(
                    bs_theta=torch.randn(1).item(),
                    bs_phi=torch.randn(1).item(),
                    phases=(torch.randn(1).item(), torch.randn(1).item()),
                    squeeze_r=(torch.randn(1).item(), torch.randn(1).item()),
                    squeeze_phi=(torch.randn(1).item(), torch.randn(1).item()),
                    displacement_r=(torch.randn(1).item(), torch.randn(1).item()),
                    displacement_phi=(torch.randn(1).item(), torch.randn(1).item()),
                    kerr=(torch.randn(1).item(), torch.randn(1).item()),
                )
                for _ in range(depth)
            ]
        self.fraud_params = list(fraud_params)
        self.classifier = self._build_classifier()

    def _build_classifier(self) -> nn.Sequential:
        """Construct the PyTorch sequential model."""
        # The first layer uses the "input" parameters; for simplicity
        # we reuse the first FraudLayerParameters object for all layers.
        input_params = self.fraud_params[0]
        return build_fraud_detection_program(input_params, self.fraud_params[1:])

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the classifier."""
        return self.classifier(x)

    def parameters(self):
        """Return the underlying PyTorch parameters."""
        return self.classifier.parameters()


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "HybridQuantumClassifier",
]
