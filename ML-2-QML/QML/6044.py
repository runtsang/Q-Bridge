import pennylane as qml
import torch
import numpy as np
import math

class SelfAttention:
    """
    Quantum self‑attention block implemented with Pennylane.
    The circuit consists of a parameterised rotation layer followed by a
    chain of controlled‑RX gates.  The expectation values of Pauli‑Z on
    each qubit are interpreted as unnormalised attention scores.
    """

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Trainable parameters
        self.q_params = torch.nn.Parameter(torch.randn(n_qubits * 3))
        self.ent_params = torch.nn.Parameter(torch.randn(n_qubits - 1))

        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, q_params, ent_params):
        # Rotation layer
        for i in range(self.n_qubits):
            qml.RX(q_params[3 * i], wires=i)
            qml.RY(q_params[3 * i + 1], wires=i)
            qml.RZ(q_params[3 * i + 2], wires=i)
        # Entangling layer
        for i in range(self.n_qubits - 1):
            qml.CRX(ent_params[i], wires=[i, i + 1])
        # Return expectation values of Pauli‑Z on each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the circuit and use the expectation values as attention
        weights to compute a weighted sum of the input embeddings.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation angles (ignored – the circuit parameters are learned
            during training).
        entangle_params : np.ndarray
            Entanglement angles (ignored).
        inputs : np.ndarray
            Input batch of shape (batch, seq_len, embed_dim).  The
            ``seq_len`` must match ``n_qubits``.
        shots : int, optional
            Number of shots for the simulator (ignored by the
            differentiable backend).  Default is 1024.

        Returns
        -------
        np.ndarray
            Weighted sum of the input embeddings, shape
            (batch, embed_dim).
        """
        # Compute raw attention scores from the quantum circuit
        raw_scores = self.qnode(self.q_params, self.ent_params)
        # ``raw_scores`` has shape (batch, n_qubits)
        raw_scores = torch.tensor(raw_scores, dtype=torch.float32)
        attn_weights = torch.softmax(raw_scores, dim=-1)

        # Broadcast weights over the embedding dimension
        weights = attn_weights.unsqueeze(-1)  # (batch, seq_len, 1)
        inputs_t = torch.tensor(inputs, dtype=torch.float32)
        weighted = torch.sum(inputs_t * weights, dim=1)  # (batch, embed_dim)
        return weighted.detach().numpy()

__all__ = ["SelfAttention"]
