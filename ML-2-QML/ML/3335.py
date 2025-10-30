"""Hybrid fraud detection model combining classical neural network layers with a qubit-based fully connected quantum sub‑layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute


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


class QuantumFeature(nn.Module):
    """Small qubit circuit that produces a single expectation value."""

    def __init__(self, theta: float, shots: int = 1024):
        super().__init__()
        self.theta = theta
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

        self.circuit = QuantumCircuit(1)
        self.circuit.h(0)
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()

    def forward(self, _x: torch.Tensor) -> torch.Tensor:
        job = execute(self.circuit, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(self.circuit)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array(list(counts.keys()), dtype=int)
        expectation = np.sum(states * probs)
        return torch.tensor([expectation], dtype=_x.dtype, device=_x.device)


class FraudLayer(nn.Module):
    """Classical linear layer augmented with a quantum feature."""

    def __init__(self, params: FraudLayerParameters, clip: bool = False) -> None:
        super().__init__()
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

        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)

        self.activation = nn.Tanh()
        self.scale = torch.tensor(params.displacement_r, dtype=torch.float32)
        self.shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

        # Quantum sub‑layer
        self.quantum = QuantumFeature(params.kerr[0] if params.kerr else 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        out = out * self.scale + self.shift
        out = out + self.quantum(x)
        return out


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> FraudLayer:
    return FraudLayer(params, clip=clip)


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class FraudDetectorHybrid:
    """Hybrid fraud detection model."""

    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> None:
        self.model = build_fraud_detection_program(input_params, layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectorHybrid"]
