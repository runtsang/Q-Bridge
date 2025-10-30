"""
Quantum self‑attention with a variational ansatz.
Uses Pennylane to build a parameterised circuit that mimics
the classical attention pattern via qubit entanglement.
"""

import pennylane as qml
import numpy as np

class SelfAttentionModule:
    """
    Variational quantum self‑attention.
    Parameters:
        qubits (int): Number of qubits (must be >= 2).
        num_layers (int): Number of variational layers.
    """
    def __init__(self, qubits: int = 4, num_layers: int = 2):
        self.qubits = qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=qubits)

        # Parameter shapes
        self.rotation_shape = (num_layers, qubits, 3)   # RX, RY, RZ per layer
        self.entangle_shape = (num_layers, qubits - 1)  # CRX per layer

        # Initialise parameters
        self.rotation_params = np.random.uniform(0, 2*np.pi,
                                                 size=self.rotation_shape)
        self.entangle_params = np.random.uniform(0, 2*np.pi,
                                                 size=self.entangle_shape)

        self.circuit = self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(rotation, entangle):
            # Apply variational layers
            for layer in range(self.num_layers):
                for q in range(self.qubits):
                    qml.RX(rotation[layer, q, 0], wires=q)
                    qml.RY(rotation[layer, q, 1], wires=q)
                    qml.RZ(rotation[layer, q, 2], wires=q)
                for q in range(self.qubits - 1):
                    qml.CRX(entangle[layer, q], wires=[q, q + 1])
            # Return expectation values of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.qubits)]
        return circuit

    def run(self, shots: int = 1024) -> np.ndarray:
        """
        Execute the circuit and return expectation values.
        """
        expvals = self.circuit(self.rotation_params, self.entangle_params)
        return np.array(expvals)

    def get_attention_matrix(self, shots: int = 1024) -> np.ndarray:
        """
        Convert the qubit expectation values into a soft‑max attention matrix.
        Each qubit represents a token; the matrix encodes pairwise similarity.
        """
        expvals = self.run(shots)
        # Map expectation values to logits
        logits = (expvals - expvals.mean()) / (expvals.std() + 1e-8)
        # Compute pairwise dot‑product similarity
        logits = logits.reshape(1, -1)
        scores = np.matmul(logits, logits.T) / np.sqrt(self.qubits)
        attn = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
        return attn

    def train_step(self, optimizer, loss_fn):
        """
        Perform a single parameter‑shift gradient descent step.
        """
        optimizer.zero_grad()
        output = self.circuit(self.rotation_params, self.entangle_params)
        loss = loss_fn(output)
        loss.backward()
        optimizer.step()
        return loss.item()

__all__ = ["SelfAttentionModule"]
