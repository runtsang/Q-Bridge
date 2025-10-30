"""Hybrid self‑attention using Pennylane parameterized circuits."""
import pennylane as qml
import torch
import numpy as np


class SelfAttentionQML:
    """Quantum‑classical self‑attention that learns attention logits via expectation values."""

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.wires = list(range(n_qubits))
        self.dev = qml.device("default.qubit", wires=self.wires)

    def _circuit(self, rotation_params, entangle_params, inputs):
        # Encode inputs as rotations
        for i in range(self.n_qubits):
            qml.RX(rotation_params[3 * i], wires=self.wires[i])
            qml.RY(rotation_params[3 * i + 1], wires=self.wires[i])
            qml.RZ(rotation_params[3 * i + 2], wires=self.wires[i])
        # Entangle adjacent qubits
        for i in range(self.n_qubits - 1):
            qml.CRX(entangle_params[i], wires=[self.wires[i], self.wires[i + 1]])
        # Return expectation values of Z as logits
        return [qml.expval(qml.Z(w)) for w in self.wires]

    def _qnode(self, rotation_params, entangle_params, inputs):
        return qml.QNode(
            lambda *args: self._circuit(*args),
            self.dev,
            interface="torch",
            diff_method="backprop",
        )(rotation_params, entangle_params, inputs)

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute hybrid attention output.

        Parameters
        ----------
        rotation_params : array of shape (3 * n_qubits,)
            Rotation angles for RX, RY, RZ on each qubit.
        entangle_params : array of shape (n_qubits - 1,)
            Parameters for the CRX entangling gates.
        inputs : torch.Tensor, shape (batch, seq_len, embed_dim)
            Input embeddings to be attended.

        Returns
        -------
        torch.Tensor
            Attended representation of shape (batch, seq_len, embed_dim).
        """
        batch_size, seq_len, embed_dim = inputs.shape
        outputs = []
        for b in range(batch_size):
            sample_out = []
            for t in range(seq_len):
                # Map embedding to rotation angles (simple linear mapping)
                rot = torch.tensor(rotation_params, dtype=torch.float32)
                ent = torch.tensor(entangle_params, dtype=torch.float32)
                logits = self._qnode(rot, ent, inputs[b, t])
                logits = torch.tensor(logits, dtype=torch.float32)
                attn_weights = torch.softmax(logits, dim=0)
                weighted = torch.sum(attn_weights.unsqueeze(-1) * inputs[b, t], dim=0)
                sample_out.append(weighted)
            outputs.append(torch.stack(sample_out))
        return torch.stack(outputs)


__all__ = ["SelfAttentionQML"]
