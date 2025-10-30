"""
AdvancedSamplerQNN: A variational quantum sampler built with Pennylane.

The circuit uses two qubits, parameterised rotations, and a single entangling layer.
It returns a QNode that can be sampled or used to compute expectation values.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from typing import Tuple


class AdvancedSamplerQNN:
    """
    Variational quantum sampler with a two‑qubit circuit.
    """

    def __init__(self, dev_name: str = "default.qubit", shots: int = 1024) -> None:
        """
        Parameters
        ----------
        dev_name : str
            Pennylane device name (e.g., "default.qubit", "qiskit", etc.).
        shots : int
            Number of measurement shots for sampling.
        """
        self.dev = qml.device(dev_name, wires=2, shots=shots)
        self._build_qnode()

    def _build_qnode(self) -> None:
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
            """
            Parameterised circuit that accepts two input angles and four trainable weights.
            """
            # Input encoding
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)

            # Entangling layer
            qml.CNOT(wires=[0, 1])

            # Trainable rotations
            for i, wire in enumerate([0, 1]):
                qml.RY(weights[i], wires=wire)
            qml.CNOT(wires=[0, 1])
            for i, wire in enumerate([0, 1], start=2):
                qml.RY(weights[i], wires=wire)

            return qml.sample(qml.PauliZ(wires=0))

        self.circuit = circuit

    def forward(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Execute the circuit and return the sampled measurement outcomes.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape (2,) with input rotation angles.
        weights : np.ndarray
            Array of shape (4,) with trainable rotation angles.

        Returns
        -------
        np.ndarray
            Sampled measurement results (±1) of shape (shots,).
        """
        return self.circuit(inputs, weights)

    def expectation(self, inputs: np.ndarray, weights: np.ndarray) -> float:
        """
        Compute the expectation value of Z on qubit 0.

        Parameters
        ----------
        inputs : np.ndarray
            Input angles.
        weights : np.ndarray
            Trainable weights.

        Returns
        -------
        float
            Expectation value ⟨Z⟩.
        """
        return np.mean(self.forward(inputs, weights))

__all__ = ["AdvancedSamplerQNN"]
