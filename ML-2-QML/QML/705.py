"""Hybrid QCNN with amplitude encoding, fully entangled ansatz, and classical readout.

The model extends the original QNN by using Pennylane for a compact variational circuit
that prepares an amplitude‑encoded state, applies two layers of entangling RX–CNOT blocks,
and then maps the expectation values through a lightweight classical neural network.
This architecture provides a richer quantum feature space while keeping the overall
model differentiable and trainable end‑to‑end in PyTorch.
"""

import pennylane as qml
import torch
from torch import nn

class QCNNGen375(nn.Module):
    """Hybrid quantum‑classical QCNN model."""

    def __init__(
        self,
        n_qubits: int = 8,
        ansatz_layers: int = 2,
        classical_hidden: int = 16,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.ansatz_layers = ansatz_layers
        self.device = qml.device("default.qubit", wires=n_qubits)

        # Trainable variational parameters
        self.var_params = nn.Parameter(
            torch.randn(ansatz_layers, n_qubits)
        )

        # Classical readout network
        self.classical = nn.Sequential(
            nn.Linear(n_qubits, classical_hidden),
            nn.Tanh(),
            nn.Linear(classical_hidden, 1),
        )

        # Define the quantum circuit as a Pennylane qnode
        def circuit(x, weights):
            # Amplitude embedding of the 8‑dimensional input
            qml.AmplitudeEmbedding(
                features=x, wires=range(n_qubits), normalize=True
            )
            # Entangled ansatz
            for layer in range(ansatz_layers):
                for i in range(n_qubits):
                    qml.RX(weights[layer, i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(n_qubits):
                    qml.RX(weights[layer, i], wires=i)
            # Expectation values of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = qml.qnode(self.device, interface="torch")(circuit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is a 1‑D tensor with length n_qubits
        qout = self.circuit(x, self.var_params)
        logits = torch.sigmoid(self.classical(qout))
        return logits

def build_QCNNGen375() -> QCNNGen375:
    """Factory returning the configured hybrid QCNN model."""
    return QCNNGen375()

__all__ = ["QCNNGen375"]
