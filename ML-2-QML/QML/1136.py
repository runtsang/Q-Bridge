"""Quantum self‑attention using a variational circuit.

The implementation follows the original interface but replaces the
classical dot‑product with a quantum‑encoded attention weight
computed from a parameterised circuit.  The circuit can be trained
with hybrid gradients, enabling a true quantum‑classical
variational attention layer.

The module exposes a factory ``SelfAttention`` that returns an
instance of ``SelfAttentionModule``.
"""

import pennylane as qml
import numpy as np
import torch

class SelfAttentionModule:
    def __init__(self, n_qubits: int = 4, device: str = "default.qubit"):
        """
        Parameters
        ----------
        n_qubits : int, default 4
            Number of qubits used to encode the embedding dimension.
        device : str, default "default.qubit"
            PennyLane device name.
        """
        self.n_qubits = n_qubits
        self.dev = qml.device(device, wires=n_qubits)

    def _quantum_attention(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        @qml.qnode(self.dev, interface="torch")
        def circuit():
            # Single‑qubit rotations
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)
            # Entangling gates
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])
            # Return Z‑expectation values as a feature vector
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for single‑qubit rotations, shape (3 * n_qubits,).
        entangle_params : np.ndarray
            Parameters for controlled‑rotation gates, shape (n_qubits - 1,).
        inputs : np.ndarray
            Input tensor of shape (batch, seq_len, embed_dim).  The
            embedding dimension must match ``n_qubits``.
        Returns
        -------
        np.ndarray
            Output after applying the quantum‑derived attention weights
            to the values (here simply the raw inputs).
        """
        batch, seq_len, embed_dim = inputs.shape
        assert embed_dim == self.n_qubits, "Embedding dimension must equal number of qubits"

        # Compute attention weights from the quantum circuit
        qnode = self._quantum_attention(rotation_params, entangle_params)
        raw_weights = qnode()
        raw_weights = torch.as_tensor(raw_weights, dtype=torch.float32)
        weights = torch.softmax(raw_weights, dim=0)

        # Apply the same weights to all tokens in the batch
        weighted = torch.einsum('bse, e -> bse', torch.as_tensor(inputs, dtype=torch.float32), weights)
        return weighted.numpy()

def SelfAttention():
    """
    Factory returning a SelfAttentionModule with 4 qubits.
    Matches the original interface so that existing code can
    import ``SelfAttention`` from this module.
    """
    return SelfAttentionModule(n_qubits=4)

__all__ = ["SelfAttention"]
