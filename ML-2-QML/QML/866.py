"""Quantum self‑attention using Pennylane variational circuit.

The circuit encodes each token’s embedding into rotation angles, applies
parameterised entangling layers, and measures Pauli‑Z expectation values
to approximate attention scores.  It supports batched execution on any
Pennylane device (default: "default.qubit") and can be trained variationally
by updating the rotation and entanglement parameters.
"""

import pennylane as qml
import numpy as np

class SelfAttention:
    """
    Variational quantum self‑attention.

    Parameters
    ----------
    n_qubits : int, default 4
        Number of qubits per token (should match the embedding dimension).
    n_layers : int, default 2
        Number of variational layers in the circuit.
    """
    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Initialise parameters randomly
        self.rotation_params = np.random.uniform(0, 2 * np.pi,
                                                 (n_layers, n_qubits, 3))
        self.entangle_params = np.random.uniform(0, 2 * np.pi,
                                                 (n_layers, n_qubits - 1))

    def _circuit(self, rot_params, ent_params):
        """Variational sub‑circuit applied at each layer."""
        for q in range(self.n_qubits):
            qml.RX(rot_params[q, 0], wires=q)
            qml.RY(rot_params[q, 1], wires=q)
            qml.RZ(rot_params[q, 2], wires=q)
        for q in range(self.n_qubits - 1):
            qml.CNOT(wires=[q, q + 1])
        for q in range(self.n_qubits - 1):
            qml.RZ(ent_params[q], wires=q)
            qml.CNOT(wires=[q, q + 1])

    def _qnode(self, inputs, rot_params, ent_params):
        """Quantum node that encodes the input and runs the variational layers."""
        for q in range(self.n_qubits):
            qml.RX(inputs[q], wires=q)
        for layer in range(self.n_layers):
            self._circuit(rot_params[layer], ent_params[layer])
        return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]

    def run(self, inputs: np.ndarray,
            rotation_params: np.ndarray = None,
            entangle_params: np.ndarray = None,
            shots: int = 1024) -> np.ndarray:
        """
        Execute the circuit on a batch of input embeddings.

        Parameters
        ----------
        inputs : np.ndarray
            Array of shape (batch, n_qubits) containing token embeddings.
        rotation_params : np.ndarray, optional
            Custom rotation parameters of shape (n_layers, n_qubits, 3).
        entangle_params : np.ndarray, optional
            Custom entanglement parameters of shape (n_layers, n_qubits-1).
        shots : int, default 1024
            Number of measurement shots.

        Returns
        -------
        np.ndarray
            Expectation values of shape (batch, n_qubits) serving as
            attention‑like scores for each token.
        """
        if rotation_params is None:
            rotation_params = self.rotation_params
        if entangle_params is None:
            entangle_params = self.entangle_params

        qnode = qml.QNode(self._qnode, device=self.dev, interface="numpy", shots=shots)

        batch_out = []
        for sample in inputs:
            batch_out.append(qnode(sample, rotation_params, entangle_params))
        return np.array(batch_out)

__all__ = ["SelfAttention"]
