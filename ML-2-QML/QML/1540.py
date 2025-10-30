"""Quantum variational circuit mimicking a fully‑connected layer using Pennylane."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class FullyConnectedLayer:
    """
    A variational quantum circuit that emulates a dense layer.  The circuit
    applies a parameterised RY rotation to each qubit, entangles the qubits
    with a simple CNOT chain, and measures the expectation of the Z operator
    on each qubit.  The weighted sum of these expectations yields a scalar
    output similar to a classical linear transformation.

    Parameters
    ----------
    n_qubits : int, default=1
        Number of qubits (input features).
    device : str, default='default.qubit'
        Pennylane device name.
    shots : int, default=100
        Number of shots used for sampling.
    """

    def __init__(self, n_qubits: int = 1, device: str = "default.qubit", shots: int = 100) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.dev = qml.device(device, wires=n_qubits, shots=shots)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params):
            # Prepare the initial state
            for w in range(n_qubits):
                qml.Hadamard(w)
            # Parameterised rotations
            for w, p in enumerate(params):
                qml.RY(p, w)
            # Simple entangling layer
            for w in range(n_qubits - 1):
                qml.CNOT(w, w + 1)
            # Measure expectation of Z on each qubit
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the variational circuit.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable containing the input parameters (features).

        Returns
        -------
        np.ndarray
            Scalar output similar to a classical linear layer.
        """
        params = np.array(list(thetas), dtype=float)
        # Pad or truncate to match n_qubits
        if params.shape[0]!= self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} parameters, got {params.shape[0]}")
        expectations = self.circuit(params)
        # Simple weighted sum (all weights = 1 for demonstration)
        output = np.sum(expectations)
        return np.array([output])


__all__ = ["FullyConnectedLayer"]
