"""Hybrid fraud detection engine – quantum implementation using PennyLane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
from pennylane import numpy as np


@dataclass
class FraudDetectionParams:
    """Parameters for a single variational layer and global hyper‑parameters."""

    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]
    dropout: float = 0.0
    clip: bool = False
    weight_clip: float | None = None
    bias_clip: float | None = None


def _clip(value: float, bound: float) -> float:
    """Clamp a scalar value to a symmetric interval."""
    return max(-bound, min(bound, value))


def _apply_layer(params: FraudDetectionParams, *, clip: bool) -> None:
    """Encode a classical layer as a sequence of parameterised gates."""
    # Rotation gates encode the phase and squeezing parameters
    for i, phase in enumerate(params.phases):
        qml.RZ(phase, wires=i)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        qml.RY(_clip(r, 5) if clip else r, wires=i)
        qml.RZ(_clip(phi, 5) if clip else phi, wires=i)
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        qml.RX(_clip(r, 5) if clip else r, wires=i)
        qml.RZ(_clip(phi, 5) if clip else phi, wires=i)
    for i, k in enumerate(params.kerr):
        qml.RZ(_clip(k, 1) if clip else k, wires=i)
    # Simple entangling layer
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 0])


def _build_circuit(params_list: Iterable[FraudDetectionParams]) -> qml.QNode:
    """Create a PennyLane QNode that implements the hybrid variational circuit."""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(inputs: np.ndarray) -> np.ndarray:
        # Encode classical inputs as X rotations
        for i in range(2):
            qml.RX(inputs[i], wires=i)

        # First layer – unclipped
        _apply_layer(params_list[0], clip=False)

        # Subsequent layers – clipped
        for params in list(params_list)[1:]:
            _apply_layer(params, clip=True)

        # Measurement – Pauli‑Z expectation value on first qubit
        return qml.expval(qml.PauliZ(0))

    return circuit


class FraudDetectionEngine:
    """Encapsulates a quantum fraud‑detection variational circuit."""

    def __init__(self, params_list: Sequence[FraudDetectionParams]) -> None:
        self.params_list = params_list
        self.circuit = _build_circuit(params_list)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return the raw circuit output."""
        return self.circuit(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return a sigmoid‑transformed probability."""
        return 1 / (1 + np.exp(-self.forward(x)))  # sigmoid

__all__ = ["FraudDetectionParams", "FraudDetectionEngine"]
