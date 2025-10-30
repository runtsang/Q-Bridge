"""Combined classical backbone for a hybrid quantum‑classical binary classifier.

The module integrates:
* A convolutional feature extractor (from ClassicalQuantumBinaryClassification).
* Fraud‑detection style dense layers (from FraudDetection).
* An RBF kernel feature (from QuantumKernelMethod).
* Optional attachment of a quantum head (placeholder for QML).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable

# --------------------------------------------------------------------------- #
# Fraud‑detection style dense layers (re‑implemented for compatibility)
# --------------------------------------------------------------------------- #
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
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]],
                          dtype=torch.float32)
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    out_features: int = 84,
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, out_features))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# RBF kernel (re‑implemented for compatibility)
# --------------------------------------------------------------------------- #
class RBFKernel(nn.Module):
    """Radial‑basis‑function kernel used as an auxiliary feature."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

# --------------------------------------------------------------------------- #
# Hybrid classifier
# --------------------------------------------------------------------------- #
class HybridQuantumClassifier(nn.Module):
    """Classical backbone with an optional quantum head."""
    def __init__(self) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.5),
        )
        # Reduce to 2‑dimensional embedding before fraud layers
        self.fc_reduce = nn.Linear(60, 2)
        # Fraud‑detection style dense layers
        input_params = FraudLayerParameters(
            bs_theta=0.5, bs_phi=0.3,
            phases=(0.1, -0.1),
            squeeze_r=(0.2, 0.2),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(1.0, 1.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
        self.fraud = build_fraud_detection_program(input_params, [], out_features=84)
        # RBF kernel feature
        self.kernel = RBFKernel(gamma=0.5)
        self.kernel_vector = nn.Parameter(torch.randn(84))
        # Final linear layer
        self.fc = nn.Linear(85, 1)
        # Placeholder for a quantum head
        self.quantum_head = None

    def set_quantum_head(self, quantum_head: nn.Module) -> None:
        """Attach a differentiable quantum head."""
        self.quantum_head = quantum_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc_reduce(x)
        x = self.fraud(x)
        k = self.kernel(x, self.kernel_vector)
        logits = self.fc(torch.cat([x, k], dim=-1))
        if self.quantum_head is not None:
            logits = self.quantum_head(logits)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "RBFKernel",
    "HybridQuantumClassifier",
]
