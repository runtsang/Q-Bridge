"""Quantum self‑attention block implemented with Pennylane.

The circuit applies a layer of Ry/Rz rotations per qubit (controlled
by ``rotation_params``) followed by a chain of controlled‑RX gates
parameterised by ``entangle_params``.  The expectation value of
Pauli‑Z on each qubit is interpreted as an attention weight and is
used to compute a weighted sum of the classical inputs.
"""

import pennylane as qml
import numpy as np
import torch

class SelfAttention:
    """Quantum self‑attention module."""

    def __init__(self, n_qubits: int = 4, dev_name: str = "default.qubit"):
        """
        Parameters
        ----------
        n_qubits : int
            Number of qubits used to encode the attention weights.
        dev_name : str
            Pennylane device name.
        """
        self.n_qubits = n_qubits
        self.dev = qml.device(dev_name, wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(rotation_params, entangle_params):
            """Variational circuit producing a state whose Z‑expectations
            encode attention weights."""
            # Rotation layer
            for i in range(self.n_qubits):
                qml.RY(rotation_params[3 * i], wires=i)
                qml.RZ(rotation_params[3 * i + 1], wires=i)
                qml.RX(rotation_params[3 * i + 2], wires=i)

            # Entanglement chain
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])

            # Return expectation values of Pauli‑Z
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            inputs: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Execute the quantum circuit and compute a weighted sum of the
        classical inputs.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (3 * n_qubits,) – angles for the rotation layer.
        entangle_params : np.ndarray
            Shape (n_qubits - 1,) – angles for the controlled‑RX gates.
        inputs : np.ndarray
            Shape (batch, seq_len, embed_dim).  Only the first
            ``n_qubits`` dimensions are used as attention weights.
        shots : int
            Number of shots for the measurement (ignored by the
            default simulator but kept for API compatibility).

        Returns
        -------
        np.ndarray
            Weighted sum of the inputs, shape (batch, seq_len, embed_dim).
        """
        if rotation_params.shape!= (3 * self.n_qubits,):
            raise ValueError(f"rotation_params must have shape ({3 * self.n_qubits},)")
        if entangle_params.shape!= (self.n_qubits - 1,):
            raise ValueError(f"entangle_params must have shape ({self.n_qubits - 1},)")

        # Run circuit on all batches in parallel using torch tensors
        batch, seq_len, embed_dim = inputs.shape
        # Expand parameters to match batch size
        rot = torch.tensor(rotation_params, dtype=torch.float32)
        ent = torch.tensor(entangle_params, dtype=torch.float32)

        # Compute expectation values
        z_expect = self.circuit(rot, ent).detach().numpy()
        # Normalise to obtain a probability distribution
        weights = np.exp(z_expect)  # positive
        weights /= weights.sum(axis=0, keepdims=True)

        # Broadcast weights to match input shape
        weights = weights.reshape(1, 1, self.n_qubits)
        weighted_inputs = inputs[..., :self.n_qubits] * weights
        # Pad remaining dimensions with zeros if embed_dim > n_qubits
        if embed_dim > self.n_qubits:
            pad = np.zeros((batch, seq_len, embed_dim - self.n_qubits))
            weighted_inputs = np.concatenate([weighted_inputs, pad], axis=-1)

        return weighted_inputs
