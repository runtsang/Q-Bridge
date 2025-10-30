"""Hybrid quantum‑classical self‑attention using Pennylane."""
from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.templates import StronglyEntanglingLayers
from typing import Callable


class SelfAttentionModule:
    """
    Variational self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits per attention head.
    num_layers : int, optional
        Depth of the variational circuit.
    dev : Callable, optional
        PennyLane device (defaults to AerSimulator).
    """

    def __init__(self, n_qubits: int, num_layers: int = 3, dev: Callable = None):
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        self.dev = dev or qml.device("default.qubit", wires=n_qubits, shots=1024)

        # Trainable parameters for the variational layers
        self.params = pnp.random.uniform(0, 2 * np.pi, (num_layers, n_qubits, 3))

        # Quantum node returning probability amplitudes
        self.qnode = qml.QNode(self._circuit, self.dev, interface="autograd")

    def _circuit(self, inputs: np.ndarray, rotation_params: np.ndarray, entangle_params: np.ndarray):
        """
        Build a parameterised circuit that applies rotation gates based on the input
        and additional trainable parameters, then entangles qubits.

        Parameters
        ----------
        inputs : np.ndarray
            Input values (shape: n_qubits).
        rotation_params : np.ndarray
            External rotation angles (shape: n_qubits).
        entangle_params : np.ndarray
            External entanglement angles (shape: n_qubits-1).
        """
        # Encode the input as rotations
        for i in range(self.n_qubits):
            qml.RX(inputs[i] + rotation_params[i], wires=i)

        # Variational layers
        StronglyEntanglingLayers(self.params, wires=range(self.n_qubits))

        # Additional entanglement from external parameters
        for i in range(self.n_qubits - 1):
            qml.CRX(entangle_params[i], wires=[i, i + 1])

        return qml.probs(wires=range(self.n_qubits))

    def run(
        self,
        inputs: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> np.ndarray:
        """
        Execute the circuit and return a probability distribution that serves as attention scores.

        Parameters
        ----------
        inputs : np.ndarray
            Input vector of shape (n_qubits,).
        rotation_params : np.ndarray
            External rotation angles (shape: n_qubits).
        entangle_params : np.ndarray
            External entanglement angles (shape: n_qubits-1).

        Returns
        -------
        np.ndarray
            Attention probability vector of shape (n_qubits,).
        """
        probs = self.qnode(inputs, rotation_params, entangle_params)
        return probs


__all__ = ["SelfAttentionModule"]
