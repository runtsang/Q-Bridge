"""Quantum self‑attention built with Pennylane, providing differentiable attention logits."""

import pennylane as qml
import numpy as np
import torch

class SelfAttentionGen299Quantum:
    """
    Variational quantum circuit implementing a self‑attention style block.
    Parameters
    ----------
    n_qubits: int, default 4
        Number of qubits (must match embed_dim).
    """
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(rotation_params: torch.Tensor,
                    entangle_params: torch.Tensor,
                    inputs: torch.Tensor):
            # Encode classical inputs as rotations on each qubit
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            # Apply variational rotation block
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)
            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])
            # Measure expectation values of PauliZ
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
            training: bool = False) -> np.ndarray:
        """
        Execute the circuit and return attention logits.
        Parameters
        ----------
        rotation_params: np.ndarray
            Parameters for single‑qubit rotations (length 3*n_qubits).
        entangle_params: np.ndarray
            Parameters for controlled‑RX gates (length n_qubits-1).
        inputs: np.ndarray
            Classical input vector of length n_qubits.
        training: bool, default False
            If True, gradients w.r.t. parameters are retained.
        Returns
        -------
        np.ndarray
            Attention logits of shape (n_qubits,).
        """
        rot_t = torch.tensor(rotation_params, dtype=torch.float32, requires_grad=training)
        ent_t = torch.tensor(entangle_params, dtype=torch.float32, requires_grad=training)
        inp_t = torch.tensor(inputs, dtype=torch.float32)
        out = self.circuit(rot_t, ent_t, inp_t)
        return out.detach().numpy() if not training else out

def SelfAttention() -> SelfAttentionGen299Quantum:
    """
    Factory that returns a quantum self‑attention instance.
    """
    return SelfAttentionGen299Quantum(n_qubits=4)

__all__ = ["SelfAttentionGen299Quantum", "SelfAttention"]
