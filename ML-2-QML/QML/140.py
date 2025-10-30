from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import pennylane as qml
import numpy as np
import torch


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer (re‑used in the quantum circuit)."""
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


def _apply_layer(device: qml.Device, params: FraudLayerParameters, *, clip: bool) -> None:
    """Parameterised gate pattern mimicking the classical layer."""
    # Basic rotations
    qml.Rot(params.bs_theta, params.bs_phi, 0.0, wires=0)
    qml.Rot(params.bs_theta, params.bs_phi, 0.0, wires=1)

    # Phase shifts
    for i, phase in enumerate(params.phases):
        qml.RZ(phase, wires=i)

    # Squeezing‑like rotations (treated as extra RZ gates)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.RZ(r if not clip else _clip(r, 5), wires=i)

    # Entangling gate
    qml.CNOT(wires=[0, 1])

    # Repeat phase shifts
    for i, phase in enumerate(params.phases):
        qml.RZ(phase, wires=i)

    # Displacement‑like rotations
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.RZ(r if not clip else _clip(r, 5), wires=i)

    # Kerr‑like rotations
    for i, k in enumerate(params.kerr):
        qml.RZ(k if not clip else _clip(k, 1), wires=i)


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    shots: int = 1024,
) -> qml.QNode:
    """Create a Pennylane QNode that encodes a 2‑dimensional input and applies the layered circuit."""
    dev = qml.device("default.qubit", wires=2, shots=shots)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs: torch.Tensor) -> torch.Tensor:
        # Encode classical input as rotation angles
        qml.RX(inputs[0], wires=0)
        qml.RX(inputs[1], wires=1)

        _apply_layer(dev, input_params, clip=False)
        for layer in layers:
            _apply_layer(dev, layer, clip=True)

        # Return expectation values of Pauli‑Z on both qubits
        return torch.stack([qml.expval(qml.PauliZ(i)) for i in range(2)])

    return circuit


__all__ = ["FraudLayerParameters", "build_fraud_detection_program"]
