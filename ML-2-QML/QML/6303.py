"""Hybrid quantum‑classical binary classifier using PennyLane.

This module implements a variational circuit with multiple layers
and integrates it with PyTorch via a qnode. The quantum circuit
encodes input features with RY rotations and uses a parameter‑shift
gradient for automatic differentiation. The output is a probability
distribution over two classes.
"""

import pennylane as qml
import torch

class HybridQuantumBinaryClassifier:
    """Hybrid quantum‑classical binary classifier with a PennyLane variational circuit."""

    def __init__(self, n_qubits: int = 4, n_layers: int = 2, device: str = "default.qubit", shots: int = 1024):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        self.dev = qml.device(device, wires=n_qubits, shots=shots)
        # Initialize parameters for the variational ansatz
        self.params = torch.nn.Parameter(torch.randn(n_layers * n_qubits))
        # Define the qnode
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs: torch.Tensor, params: torch.Tensor):
            # Encode input features into the qubits using RY rotations
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            # Variational ansatz: repeated layers of single‑qubit rotations and CNOTs
            idx = 0
            for _ in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(params[idx], wires=i)
                    idx += 1
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            # Measure expectation of PauliZ on the first qubit
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Flatten the input if it contains more than one dimension
        flat_inputs = inputs.view(inputs.size(0), -1)
        # Compute expectation values
        exp_vals = self.circuit(flat_inputs, self.params)
        # Convert expectation to probability using sigmoid
        probs = torch.sigmoid(exp_vals)
        return torch.stack([probs, 1 - probs], dim=-1)

__all__ = ["HybridQuantumBinaryClassifier"]
