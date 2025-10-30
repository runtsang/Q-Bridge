"""Quantum‑classical fraud detection hybrid using Pennylane and Strawberry Fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pennylane as qml
import pennylane.numpy as np
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate


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
    return max(-bound, min(bound, value))


def _apply_layer_qml(
    wires: list[int],
    params: FraudLayerParameters,
    *,
    clip: bool,
) -> None:
    qml.BSgate(params.bs_theta, params.bs_phi, wires=[wires[0], wires[1]])
    for i, phase in enumerate(params.phases):
        qml.Rgate(phase, wires=wires[i])
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.Sgate(r if not clip else _clip(r, 5), phi, wires=wires[i])
    qml.BSgate(params.bs_theta, params.bs_phi, wires=[wires[0], wires[1]])
    for i, phase in enumerate(params.phases):
        qml.Rgate(phase, wires=wires[i])
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.Dgate(r if not clip else _clip(r, 5), phi, wires=wires[i])
    for i, k in enumerate(params.kerr):
        qml.Kgate(k if not clip else _clip(k, 1), wires=wires[i])


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Create a Strawberry Fields program for the hybrid fraud detection model."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer_qml([q[0], q[1]], input_params, clip=False)
        for layer in layers:
            _apply_layer_qml([q[0], q[1]], layer, clip=True)
    return program


class FraudDetectionHybrid:
    """Hybrid quantum‑classical fraud detection model with Pennylane QNode."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        dev: qml.Device | None = None,
        shots: int = 5000,
    ) -> None:
        if dev is None:
            dev = qml.device("strawberryfields.fock", wires=2, cutoff_dim=10)
        self.dev = dev
        self.shots = shots
        self.input_params = input_params
        self.layers = list(layers)

        @qml.qnode(self.dev, interface="torch")
        def circuit(x: np.ndarray) -> np.ndarray:
            # Encode classical input as displacement gates
            for i, val in enumerate(x):
                qml.Dgate(val, wires=i)
            # Apply the photonic circuit
            _apply_layer_qml([0, 1], self.input_params, clip=False)
            for layer in self.layers:
                _apply_layer_qml([0, 1], layer, clip=True)
            # Measure photon‑number expectation via Pauli‑Z
            return [qml.expval(qml.PauliZ(i)) for i in range(2)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the hybrid circuit on a batch of inputs."""
        return torch.stack(
            [self.circuit(x[i].detach().numpy()) for i in range(x.shape[0])], dim=0
        )
