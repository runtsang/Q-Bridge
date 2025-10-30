"""
Hybrid quantum sampler employing a depth‑controlled parameterised ansatz
with entangling layers.  The sampler can handle arbitrary input dimensionality
and returns probability distributions via statevector sampling.

The circuit uses Pennylane and a default qubit device.  It is designed to be
plugged into classical pipelines or trained end‑to‑end with gradient‑based
optimisers.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml


class AdvancedSamplerQNN:
    """
    Quantum sampler with configurable depth and entangling structure.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the classical input vector.
    num_qubits : int
        Number of qubits in the circuit (>= input_dim).
    depth : int
        Number of variational layers.
    device : pennylane.Device, optional
        Pennylane device to run the circuit on.  Defaults to the
        'default.qubit' simulator.
    """
    def __init__(
        self,
        input_dim: int = 2,
        num_qubits: int = 2,
        depth: int = 3,
        device: qml.Device | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.num_qubits = num_qubits
        self.depth = depth
        self.device = device or qml.device("default.qubit", wires=num_qubits)

        # Trainable variational weights
        self.weights = np.random.randn(depth, num_qubits)

    def _circuit(self, inputs: np.ndarray) -> None:
        """Internal circuit construction using Pennylane."""
        # Encode classical inputs with Ry rotations
        for i in range(self.num_qubits):
            qml.RY(inputs[i], wires=i)

        # Variational layers with entanglement
        for d in range(self.depth):
            for i in range(self.num_qubits):
                qml.RY(self.weights[d, i], wires=i)
            # Full‑connect entangling block (CNOT ladder)
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.num_qubits - 1, 0])

    def probs(self, inputs: np.ndarray) -> np.ndarray:
        """
        Return the probability distribution over computational basis states.

        Parameters
        ----------
        inputs : np.ndarray
            Classical input vector of shape (input_dim,).

        Returns
        -------
        np.ndarray
            Probabilities of shape (2 ** num_qubits,).
        """
        @qml.qnode(self.device)
        def circuit(inp: np.ndarray) -> np.ndarray:
            self._circuit(inp)
            return qml.probs()

        return circuit(inputs)

    def sample(self, inputs: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Sample from the quantum state and estimate the probability distribution.

        Parameters
        ----------
        inputs : np.ndarray
            Classical input vector of shape (input_dim,).
        shots : int
            Number of measurement shots.

        Returns
        -------
        np.ndarray
            Estimated probabilities of shape (2 ** num_qubits,).
        """
        @qml.qnode(self.device)
        def circuit(inp: np.ndarray) -> np.ndarray:
            self._circuit(inp)
            return qml.sample()

        samples = circuit(inputs)
        probs = np.bincount(samples, minlength=2 ** self.num_qubits) / shots
        return probs


__all__ = ["AdvancedSamplerQNN"]
