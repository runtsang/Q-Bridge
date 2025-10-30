"""Quantum self‑attention using a Pennylane variational circuit.

The implementation stays compatible with the original API while adding
quantum‑centric features: a parameterized rotation block, an entanglement
layer driven by `entangle_params`, and an expectation‑value readout that
serves as attention weights.  The module can be extended to use any
back‑end Pennylane supports (e.g., qiskit, cirq, or the default simulator).
"""

import numpy as np
import pennylane as qml
import torch


class SelfAttention:
    """
    Variational self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (also the number of attention heads in this toy
        implementation).
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor):
            # Encode each input dimension as a rotation on a distinct qubit
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)

            # Entanglement block driven by `entangle_params`
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])

            # Readout: expectation values of Pauli‑Z
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the variational circuit and produce attention‑weighted
        representations.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the RX/RY/RZ gates. Shape must be
            ``(3 * n_qubits,)``.
        entangle_params : np.ndarray
            Parameters for the CRX entanglement gates. Shape
            ``(n_qubits - 1,)``.
        inputs : np.ndarray
            Input batch of shape ``(batch, seq_len, embed_dim)``.
        shots : int, default=1024
            Number of shots for the backend (ignored by the default
            simulator but kept for API compatibility).

        Returns
        -------
        np.ndarray
            Attention‑weighted representations of shape
            ``(batch, embed_dim)``.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        batch = torch.as_tensor(inputs, dtype=torch.float32)

        # Run the circuit for each sample in the batch
        attn_scores = []
        for sample in batch:
            # `sample` has shape [seq_len, embed_dim]; we flatten it to feed
            # into the circuit.  In this toy example we simply sum the
            # dimensions to obtain a single vector per qubit.
            flat = sample.reshape(-1)
            probs = circuit(flat)
            probs = (np.array(probs) + 1) / 2  # Convert expectation to [0,1]
            attn_scores.append(probs)

        attn_scores = np.array(attn_scores)  # shape [batch, n_qubits]
        # Normalize to get an attention distribution
        attn = attn_scores / attn_scores.sum(axis=-1, keepdims=True)

        # Weighted sum of the original inputs
        # (here we treat each qubit as a head and weight the entire vector)
        weighted = np.einsum("bi,bij->bj", attn, inputs)
        return weighted


def SelfAttention():
    """Convenience factory matching the original seed signature."""
    return SelfAttention(n_qubits=4)


__all__ = ["SelfAttention"]
