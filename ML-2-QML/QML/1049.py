"""Variational quantum self‑attention using Pennylane.

The circuit produces attention scores as expectation values of Pauli‑Z
measurements on a set of qubits.  The interface mirrors the classical
module: a `run` method that accepts rotation and entanglement parameters
and returns a probability distribution over the input sequence.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from typing import Tuple

class QuantumSelfAttentionModule:
    """
    Quantum self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits representing the sequence length.
    dev : pennylane.Device
        Quantum device used for simulation or hardware execution.
    """

    def __init__(self, n_qubits: int, dev: qml.Device):
        self.n_qubits = n_qubits
        self.dev = dev
        self.qnode = qml.QNode(self._circuit, dev)

    def _circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a parameter‑shiftable circuit that encodes the inputs
        and applies rotation/entanglement layers.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (attention_scores, values)
        """
        # Encode inputs as amplitude on computational basis
        for i, val in enumerate(inputs):
            qml.RY(val, wires=i)

        # Rotation layer
        for i in range(self.n_qubits):
            qml.RX(rotation_params[3 * i], wires=i)
            qml.RY(rotation_params[3 * i + 1], wires=i)
            qml.RZ(rotation_params[3 * i + 2], wires=i)

        # Entanglement layer (controlled‑X)
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

        # Additional entanglement via parameters
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
            qml.RZ(entangle_params[i], wires=i + 1)

        # Measure expectation values of Z on each qubit
        scores = [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        # Values are simply the raw input embeddings for demonstration
        values = inputs
        return np.array(scores), values

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Execute the circuit and return attention weights.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation angles for each qubit (size 3 * n_qubits).
        entangle_params : np.ndarray
            Entanglement angles (size n_qubits - 1).
        inputs : np.ndarray
            Input sequence embeddings (size n_qubits).

        Returns
        -------
        np.ndarray
            Normalised attention weights (softmax over Z expectation values).
        """
        scores, _ = self.qnode(rotation_params, entangle_params, inputs)
        # Convert expectation values to probabilities
        probs = np.exp(scores) / np.sum(np.exp(scores))
        return probs


def SelfAttention() -> QuantumSelfAttentionModule:
    """
    Factory function returning a quantum attention module pre‑configured
    for 4 qubits using the default Aer simulator.
    """
    dev = qml.device("default.qubit", wires=4)
    return QuantumSelfAttentionModule(n_qubits=4, dev=dev)


__all__ = ["QuantumSelfAttentionModule", "SelfAttention"]
