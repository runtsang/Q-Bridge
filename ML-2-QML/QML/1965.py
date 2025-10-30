"""SamplerQNN__gen160 – A Pennylane variational sampler with ancilla read‑out.

The circuit uses two data qubits and a single ancilla qubit.  Data qubits
receive a rotation‑based embedding of the input features; the ancilla
encodes the weight parameters via a controlled‑rotation.  After a series
of entangling gates the circuit is measured with a Pauli‑Z expectation on
the ancilla, producing a probability distribution that is mapped to two
outcomes via a softmax‑like transformation.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

# Device for sampling; using default simulator for quick prototyping
dev = qml.device("default.qubit", wires=3, shots=1024)


class SamplerQNN__gen160:
    """
    Variational QNN sampler.

    Attributes
    ----------
    num_qubits : int
        Number of data qubits (default 2).
    params : np.ndarray
        Trainable weight parameters (shape (4,)).
    """

    def __init__(self, num_qubits: int = 2) -> None:
        self.num_qubits = num_qubits
        # Initialize four trainable rotation angles
        self.params = np.random.uniform(0, 2 * np.pi, size=(4,))

    def _embed(self, inputs: np.ndarray) -> None:
        """Data embedding via RY rotations."""
        for i in range(self.num_qubits):
            qml.RY(inputs[i], wires=i)

    def _variational_block(self) -> None:
        """Entangling and parameterised rotations."""
        for i in range(self.num_qubits):
            qml.RY(self.params[i], wires=i)
        # Entangle data qubits
        qml.CNOT(wires=[0, 1])
        # Couple ancilla to data qubits
        qml.CNOT(wires=[0, 2])
        qml.CNOT(wires=[1, 2])
        # Final rotations on ancilla
        qml.RY(self.params[2], wires=2)
        qml.RY(self.params[3], wires=2)

    @qml.qnode(dev)
    def circuit(self, inputs: np.ndarray) -> np.ndarray:
        """Quantum circuit that outputs a probability of measuring |0> on ancilla."""
        self._embed(inputs)
        self._variational_block()
        # Measure ancilla in computational basis
        return qml.probs(wires=2)

    def sample(self, inputs: np.ndarray) -> np.ndarray:
        """
        Execute the circuit and return a two‑class probability distribution.

        Parameters
        ----------
        inputs : array-like of shape (2,)
            Two‑dimensional input features.

        Returns
        -------
        np.ndarray of shape (2,)
            Softmax‑like probabilities for two outcomes.
        """
        probs = self.circuit(inputs)  # shape (2,) – prob of |0>, |1>
        # Map raw probabilities to a two‑class distribution (already normalized)
        return probs

    def update_params(self, grads: np.ndarray, lr: float = 0.01) -> None:
        """
        Gradient‑based update of the trainable parameters.

        Parameters
        ----------
        grads : array-like of shape (4,)
            Gradient of loss w.r.t. parameters.
        lr : float
            Learning rate.
        """
        self.params -= lr * grads

    def __repr__(self) -> str:
        return f"SamplerQNN__gen160(num_qubits={self.num_qubits})"


__all__ = ["SamplerQNN__gen160"]
