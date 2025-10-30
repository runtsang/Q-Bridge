"""Quantum classifier built with PennyLane, featuring an entangling ansatz and a hybrid classical readout.

The class implements a QNode that can be vectorized over a batch of inputs and returns
logits in the same shape as the classical counterpart.  The interface is kept compatible
with the original seed: the class exposes encoding, weight_sizes, and observables attributes.
"""

import pennylane as qml
import pennylane.numpy as np
from typing import Iterable, List

class QuantumClassifierModel:
    """
    Variational quantum classifier.

    Parameters
    ----------
    num_qubits : int
        Number of qubits / features.
    depth : int
        Number of variational layers.
    device : str, optional
        PennyLane device name (default 'default.qubit').
    wires : Iterable[int], optional
        Wire indices; defaults to range(num_qubits).
    """
    def __init__(self, num_qubits: int, depth: int, device: str = "default.qubit", wires: Iterable[int] = None):
        self.num_qubits = num_qubits
        self.depth = depth
        self.wires = wires or list(range(num_qubits))
        self.dev = qml.device(device, wires=self.wires)

        # Variational parameters
        self.weights = np.random.randn(depth, num_qubits) * 0.01

        # Classical readout head: linear layer mapping expectation values to logits
        self.readout = np.random.randn(2, num_qubits) * 0.1
        self.bias = np.random.randn(2) * 0.01

        self._qnode = qml.QNode(self._circuit, self.dev, interface="autograd")

    def _circuit(self, x, weights):
        # Data encoding
        for i, wire in enumerate(self.wires):
            qml.RX(x[i], wire=wire)

        # Variational layers
        for layer in range(self.depth):
            for q in self.wires:
                qml.RY(weights[layer][q], q)
            # Entangling pattern: cyclic CZ
            for q in range(self.num_qubits - 1):
                qml.CZ(self.wires[q], self.wires[q + 1])
            qml.CZ(self.wires[-1], self.wires[0])

        # Measurements: expectation of Z on each qubit
        return [qml.expval(qml.PauliZ(w)) for w in self.wires]

    def __call__(self, x_batch: np.ndarray) -> np.ndarray:
        """
        Forward pass over a batch of inputs.

        Parameters
        ----------
        x_batch : np.ndarray of shape (batch, num_qubits)
            Feature vectors.

        Returns
        -------
        logits : np.ndarray of shape (batch, 2)
            Raw scores for the two classes.
        """
        preds = np.array([self._qnode(x, self.weights) for x in x_batch])
        logits = preds @ self.readout.T + self.bias
        return logits

    @property
    def weight_sizes(self) -> List[int]:
        """Return number of trainable parameters per variational layer."""
        return [self.num_qubits] * self.depth

    @property
    def observables(self) -> List[int]:
        """Indices of output logits."""
        return [0, 1]

__all__ = ["QuantumClassifierModel"]
