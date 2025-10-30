"""Variational quantum regressor with entanglement and adaptive measurement.

This module implements EstimatorQNNGen using Pennylane. The circuit
consists of 3 qubits, parameterised Ry rotations, CNOT entangling layers
and a measurement of a Pauli‑Y observable. The class exposes an `evaluate`
method that accepts classical inputs and returns the expectation value.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np


class EstimatorQNNGen:
    """
    Quantum neural network estimator.

    Attributes
    ----------
    device : qml.Device
        Quantum device used for simulation.
    qnode : qml.QNode
        Parameterised quantum circuit.
    """

    def __init__(self, shots: int = 1000) -> None:
        """
        Initialise the quantum estimator.

        Parameters
        ----------
        shots : int
            Number of measurement shots for expectation estimation.
        """
        self.device = qml.device("default.qubit", wires=3, shots=shots)

        @qml.qnode(self.device, interface="autograd")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
            # Input embedding
            for i, wire in enumerate(range(3)):
                qml.RY(inputs[i], wires=wire)

            # Variational block
            for _ in range(2):  # two repetitions
                for wire in range(3):
                    qml.RY(weights[wire], wires=wire)
                    qml.RZ(weights[wire + 3], wires=wire)
                # Entangling layer
                for wire in range(2):
                    qml.CNOT(wires=[wire, wire + 1])

            # Measurement of Y observable on all qubits
            return qml.expval(qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliY(2))

        self.qnode = circuit

    def evaluate(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Compute the expectation value for given inputs and weights.

        Parameters
        ----------
        inputs : np.ndarray
            Classical input vector of shape (3,).
        weights : np.ndarray
            Variational parameters of shape (6,).

        Returns
        -------
        np.ndarray
            Expectation value of the Pauli‑Y product observable.
        """
        return self.qnode(inputs, weights)

    def __call__(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Alias to `evaluate` for functional usage.

        Parameters
        ----------
        inputs : np.ndarray
            Classical input vector.
        weights : np.ndarray
            Variational parameters.

        Returns
        -------
        np.ndarray
            Expectation value.
        """
        return self.evaluate(inputs, weights)


__all__ = ["EstimatorQNNGen"]
