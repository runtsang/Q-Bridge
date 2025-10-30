"""Hybrid classical-quantum binary classifier with fraud‑detection inspired layers.

This module builds a CNN front‑end, a dense fraud‑detection style head, and a
quantum expectation head.  It extends the original ClassicalQuantumBinaryClassification
design by replacing the simple linear head with a sequence of
parameter‑controlled layers derived from the photonic fraud‑detection
example.  The quantum interface remains fully differentiable via a custom
autograd function that uses the parameter‑shift rule.

The model can be trained end‑to‑end with standard PyTorch optimisers; the
quantum part is executed on a Qiskit Aer simulator.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected layer in the fraud detection head."""
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
    """Create a single dense layer that mimics a photonic layer."""
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
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class FraudHybridQCNet(nn.Module):
    """CNN → fraud‑detection dense head → quantum expectation head."""
    def __init__(
        self,
        fraud_params: Tuple[FraudLayerParameters,...],
        quantum_shift: float = 3.141592653589793 / 2,
    ) -> None:
        super().__init__()
        # Convolutional front end (identical to the original QCNet)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Dense fraud‑detection head
        self.fraud_head = build_fraud_detection_program(fraud_params[0], fraud_params[1:])

        # Quantum head (defined in qml module)
        from. import QuantumCircuit, Hybrid  # local import to avoid circular deps
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.quantum = Hybrid(
            n_qubits=self.fraud_head[-1].out_features,
            backend=backend,
            shots=200,
            shift=quantum_shift,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = self.fraud_head(x)
        # quantum expectation head
        quantum_out = self.quantum(x).T  # shape (batch, 1)
        probs = torch.cat((quantum_out, 1 - quantum_out), dim=-1)
        return probs


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudHybridQCNet",
]
