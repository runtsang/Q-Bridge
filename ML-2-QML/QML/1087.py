"""Quantum self‑attention using a variational circuit."""

import numpy as np
import pennylane as qml
import torch

class QuantumSelfAttention:
    """Variational self‑attention with parameterized rotations and entanglement."""
    def __init__(self, n_qubits: int = 4, num_layers: int = 2):
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(rotation_params, entangle_params):
            # Layered rotations
            for layer in range(self.num_layers):
                for q in range(self.n_qubits):
                    qml.RX(rotation_params[layer, q, 0], wires=q)
                    qml.RY(rotation_params[layer, q, 1], wires=q)
                    qml.RZ(rotation_params[layer, q, 2], wires=q)
                # Entangling CNOT chain
                for q in range(self.n_qubits - 1):
                    qml.CNOT(wires=[q, q+1])
                # Parameterized entanglement
                for q in range(self.n_qubits - 1):
                    qml.CRX(entangle_params[layer, q], wires=[q, q+1])
            # Expectation of PauliZ on each qubit gives attention logits
            return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]

        self.circuit = circuit

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024) -> np.ndarray:
        """
        Parameters
        ----------
        rotation_params : np.ndarray
            Shape (num_layers, n_qubits, 3)
        entangle_params : np.ndarray
            Shape (num_layers, n_qubits-1)
        shots : int
            Number of measurement shots.

        Returns
        -------
        np.ndarray
            Shape (n_qubits,) – normalized attention probabilities.
        """
        rot = torch.tensor(rotation_params, dtype=torch.float32)
        ent = torch.tensor(entangle_params, dtype=torch.float32)
        logits = self.circuit(rot, ent)
        logits = torch.stack(logits).detach().numpy()
        probs = np.exp(logits) / np.exp(logits).sum()
        return probs

def SelfAttention(n_qubits: int = 4, num_layers: int = 2):
    """Factory returning a QuantumSelfAttention instance."""
    return QuantumSelfAttention(n_qubits, num_layers)

__all__ = ["SelfAttention"]
