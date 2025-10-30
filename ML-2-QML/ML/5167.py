"""Hybrid classical‑quantum model inspired by Quantum‑NAT, Fraud‑Detection, and Conv filters."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable


# -------------------------------------------------------------
# 1. Classical Conv filter (drop‑in replacement for quanvolution)
# -------------------------------------------------------------
class ConvFilter(nn.Module):
    """Simple 2‑D convolution followed by a sigmoid activation."""

    def __init__(self, kernel_size: int = 3, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations


# -------------------------------------------------------------
# 2. Fraud‑Detection inspired dense head
# -------------------------------------------------------------
@dataclass
class FraudLayerParameters:
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
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a dense sequence mimicking the photonic fraud‑detection stack."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# -------------------------------------------------------------
# 3. Hybrid sigmoid head (classical analogue of a quantum expectation)
# -------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation with an optional shift."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:  # type: ignore[override]
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None


class Hybrid(nn.Module):
    """Linear head followed by a differentiable sigmoid."""

    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)


# -------------------------------------------------------------
# 4. Main hybrid model
# -------------------------------------------------------------
class HybridQuantumNAT(nn.Module):
    """Convolutional backbone + fraud‑like dense head + optional hybrid sigmoid."""

    def __init__(self, n_classes: int = 1, shift: float = 0.0) -> None:
        super().__init__()
        # Convolutional backbone (same as QFCModel)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Reduce to 2‑dimensional representation
        self.fc_reduce = nn.Linear(16 * 7 * 7, 2)

        # Fraud‑detection inspired dense stack
        input_params = FraudLayerParameters(
            bs_theta=0.5,
            bs_phi=0.3,
            phases=(0.1, -0.1),
            squeeze_r=(0.2, 0.2),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(0.4, 0.4),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
        layers = [
            FraudLayerParameters(
                bs_theta=0.6,
                bs_phi=0.4,
                phases=(0.2, -0.2),
                squeeze_r=(0.3, 0.3),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(0.5, 0.5),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
            for _ in range(2)
        ]
        self.fraud_head = build_fraud_detection_program(input_params, layers)

        # Optional hybrid sigmoid head
        self.hybrid = Hybrid(1, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        reduced = self.fc_reduce(flattened)
        fraud_output = self.fraud_head(reduced)  # shape (bsz, 1)
        probs = self.hybrid(fraud_output)
        return probs


__all__ = ["HybridQuantumNAT", "HybridFunction", "Hybrid", "FraudLayerParameters", "build_fraud_detection_program"]
