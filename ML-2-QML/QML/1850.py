"""Quantum sampler built with Pennylane and a variational circuit."""
from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from typing import Tuple


class SamplerQNN:
    """
    A variational sampler that returns a probability distribution over
    a 2‑qubit system.  The circuit uses two layers of RY rotations
    followed by a full‑SWAP entanglement block and a final rotation
    layer.  The `sample` method uses Pennylane's `sample` measurement
    to draw samples from the quantum state.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit (default 2).
    weight_dim : int
        Number of trainable parameters in the rotation layers.
    """

    def __init__(self, num_qubits: int = 2, weight_dim: int = 6) -> None:
        self.num_qubits = num_qubits
        self.weight_dim = weight_dim
        dev = qml.device("default.qubit", wires=num_qubits)
        self._qnode = qml.QNode(self._circuit, dev, interface="autograd", diff_method="backprop")

    def _circuit(self, inputs: Tuple[float, float], weights: np.ndarray) -> np.ndarray:
        """Variational circuit with input embedding and trainable weights."""
        # Input embedding: encode the 2‑dimensional classical input as
        # rotation angles on the first two qubits.
        qml.RY(inputs[0], wires=0)
        qml.RY(inputs[1], wires=1)

        # Entangling layer
        for i in range(self.num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

        # Trainable rotation layer
        for i in range(self.weight_dim):
            qml.RY(weights[i], wires=i % self.num_qubits)

        # Final SWAP to mix amplitudes
        for i in range(self.num_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[i + 1, i])
            qml.CNOT(wires=[i, i + 1])

        return qml.probs(wires=range(self.num_qubits))

    def sample(self, inputs: Tuple[float, float], weights: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Draw samples from the quantum state defined by `inputs` and `weights`.

        Parameters
        ----------
        inputs : Tuple[float, float]
            Classical 2‑dimensional input vector.
        weights : np.ndarray
            1‑D array of trainable parameters.
        shots : int, optional
            Number of measurement shots.

        Returns
        -------
        np.ndarray
            Sample counts for each basis state in lexicographic order.
        """
        probs = self._qnode(inputs, weights)
        # Use the built‑in probability distribution to sample
        return np.random.choice(len(probs), size=shots, p=probs)

    def __call__(self, inputs: Tuple[float, float], weights: np.ndarray) -> np.ndarray:
        """
        Alias for :meth:`sample` to keep the original functional style.
        """
        return self.sample(inputs, weights)


def SamplerQNN_factory() -> SamplerQNN:
    """
    Factory returning an instance of :class:`SamplerQNN`.  The name is
    distinct from the classical version to avoid import clashes, yet
    the function signature aligns with the original seed.
    """
    return SamplerQNN()


__all__ = ["SamplerQNN_factory"]
