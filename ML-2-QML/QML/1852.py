"""Quantum neural network with 2‑qubit entangled ansatz for regression."""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from typing import List

class EstimatorQNNHybrid:
    """
    A 2‑qubit variational quantum circuit for regression.

    The circuit consists of:
    - Input‑encoding layers: Ry rotations that encode `x[0]` and `x[1]` on each qubit.
    - Entanglement via a CNOT gate.
    - Parameterised weight layers: Rz rotations on each qubit.
    The expectation value of Pauli‑Z on both qubits is returned as the output.

    Attributes
    ----------
    device : qml.Device
        QPU simulator.
    params : np.ndarray
        Trainable parameters (weights).
    """

    def __init__(self, wires: int = 2, shots: int = 1000) -> None:
        self.device = qml.device("default.qubit", wires=wires, shots=shots)
        # 4 weight parameters: one per qubit for each of the two weight rotations
        self.params = np.random.randn(4)

    def circuit(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Variational circuit returning the expectation value of Pauli‑Z."""
        @qml.qnode(self.device)
        def qnode():
            # Input encoding
            qml.Ry(inputs[0], wires=0)
            qml.Ry(inputs[1], wires=1)
            # Entanglement
            qml.CNOT(wires=[0, 1])
            # Weight rotations
            qml.Rz(weights[0], wires=0)
            qml.Rz(weights[1], wires=1)
            # Second layer of entanglement
            qml.CNOT(wires=[1, 0])
            # Second weight rotations
            qml.Rz(weights[2], wires=0)
            qml.Rz(weights[3], wires=1)
            # Measurement
            return qml.expval(qml.PauliZ(0)) + qml.expval(qml.PauliZ(1))

        return qnode()

    def predict(self, inputs: np.ndarray) -> float:
        """Return the regression output for a single input sample."""
        return float(self.circuit(inputs, self.params))

    @staticmethod
    def default() -> "EstimatorQNNHybrid":
        """Convenience constructor with default settings."""
        return EstimatorQNNHybrid()

__all__ = ["EstimatorQNNHybrid"]
