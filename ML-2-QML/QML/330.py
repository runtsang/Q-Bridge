import pennylane as qml
import numpy as np
import torch

class SelfAttention:
    """Quantum‑enhanced self‑attention block."""
    def __init__(self, n_qubits: int, dev: qml.Device | None = None):
        self.n_qubits = n_qubits
        self.dev = dev or qml.device("default.qubit", wires=n_qubits)
        @qml.qnode(self.dev)
        def circuit(rot_params, ent_params):
            for i in range(self.n_qubits):
                qml.RY(rot_params[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CRX(ent_params[i], wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        self.circuit = circuit
    
    def run(self, inputs: np.ndarray,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray) -> np.ndarray:
        """Compute quantum‑enhanced attention."""
        raw_weights = self.circuit(rotation_params, entangle_params)
        probs = np.exp(raw_weights) / np.sum(np.exp(raw_weights))
        batch, seq_len, embed_dim = inputs.shape
        weights = np.tile(probs, (batch, seq_len, 1))
        out = np.sum(weights[..., :seq_len] * inputs, axis=1, keepdims=True)
        out = np.tile(out, (1, seq_len, 1))
        return out
