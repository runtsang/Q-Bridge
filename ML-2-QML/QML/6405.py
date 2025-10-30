from __future__ import annotations

import pennylane as qml
import numpy as np
import torch
from dataclasses import dataclass
from typing import Iterable


@dataclass
class FraudLayerParameters:
    """Parameters that mirror the original photonic layer but are now
    interpreted as initialization values for a variational ansatz."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


class FraudDetectionHybrid:
    """Variational quantum circuit for fraud detection.

    The circuit encodes two classical features via RZ rotations,
    applies a layered ansatz of RX/RZ rotations and CNOT entangling gates,
    and returns a fraud probability derived from the expectation of
    Pauli‑Z on the first qubit.
    """
    def __init__(self, n_qubits: int = 2, layers: int = 2):
        self.n_qubits = n_qubits
        self.layers = layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.params = np.random.randn(layers, n_qubits, 3)  # theta, phi, unused
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, x: np.ndarray, params: np.ndarray):
        # Encode classical data
        for i in range(self.n_qubits):
            qml.RZ(x[i], wires=i)
        # Variational ansatz
        for l in range(self.layers):
            for i in range(self.n_qubits):
                qml.RY(params[l, i, 0], wires=i)
                qml.RZ(params[l, i, 1], wires=i)
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            # Entangling layer
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
        return qml.expval(qml.PauliZ(0))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raw = self.qnode(x, self.params)
        # Map expectation value [-1, 1] to probability [0, 1]
        return (raw + 1) / 2

    @staticmethod
    def from_params(
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> "FraudDetectionHybrid":
        """Create a quantum circuit from the photonic parameters.
        The parameters are used to initialise the first layer of the
        variational ansatz for demonstration purposes."""
        obj = FraudDetectionHybrid()
        # Map bs_theta to first layer RY, bs_phi to RZ on qubit 0
        obj.params[0, 0, 0] = input_params.bs_theta
        obj.params[0, 0, 1] = input_params.bs_phi
        return obj


__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
