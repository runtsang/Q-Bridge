"""Pennylane implementation of the fraud detection hybrid circuit."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pennylane as qml
import numpy as np
import torch


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


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


class FraudDetectionModel:
    """Hybrid quantum-classical fraud detection model using Pennylane."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        n_qubits: int = 2,
        shots: int = 1024,
    ):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
        self.input_params = input_params
        self.layers = list(layers)
        self._build_qnode()

    def _apply_layer(self, params: FraudLayerParameters, clip: bool):
        theta = params.bs_theta if not clip else _clip(params.bs_theta, 5.0)
        phi = params.bs_phi if not clip else _clip(params.bs_phi, 5.0)
        for i in range(self.n_qubits):
            qml.RZ(params.phases[i], wires=i)
            qml.RX(theta, wires=i)
            qml.RZ(phi, wires=i)
        for i in range(self.n_qubits):
            r = params.squeeze_r[i] if not clip else _clip(params.squeeze_r[i], 5.0)
            phi_s = params.squeeze_phi[i]
            qml.Rot(r, phi_s, 0.0, wires=i)
        for i in range(self.n_qubits):
            d = params.displacement_r[i] if not clip else _clip(params.displacement_r[i], 5.0)
            phi_d = params.displacement_phi[i]
            qml.PhaseShift(d, wires=i)
        for i in range(self.n_qubits):
            k = params.kerr[i] if not clip else _clip(params.kerr[i], 1.0)
            qml.PhaseShift(k, wires=i)

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor):
            for i, val in enumerate(inputs):
                qml.RY(val, wires=i)
            self._apply_layer(self.input_params, clip=False)
            for layer in self.layers:
                self._apply_layer(layer, clip=True)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self._qnode = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self._qnode(x)
        probs = 0.5 * (torch.tensor(raw) + 1.0)
        return probs.mean().unsqueeze(0)

__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
