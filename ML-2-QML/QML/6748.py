import pennylane as qml
import torch
from torch import nn
from typing import List

class QuantumDecoder(nn.Module):
    """Parameter‑driven variational decoder using Pennylane."""
    def __init__(self, latent_dim: int, output_dim: int, device: torch.device | None = None) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.device = device or torch.device("cpu")

        # Variational parameters: a square matrix of shape (latent_dim, latent_dim)
        self.weights = nn.Parameter(torch.randn(latent_dim, latent_dim))

        # Classical mapping from quantum outputs to the desired output dimension
        self.classical_map = nn.Linear(latent_dim, output_dim)

        # Pennylane device and qnode
        self.dev = qml.device("default.qubit", wires=latent_dim, shots=None)
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method="backprop")

    def _circuit(self, latent: torch.Tensor, weights: torch.Tensor) -> List[torch.Tensor]:
        """Variational circuit that encodes the latent vector and applies a trainable layer."""
        # Encode latent as rotation angles
        for i in range(self.latent_dim):
            qml.RX(latent[i], wires=i)

        # Entangling layer
        for i in range(self.latent_dim - 1):
            qml.CNOT(wires=[i, i + 1])

        # Parameterized rotation layer
        for i in range(self.latent_dim):
            qml.RZ(weights[i, i], wires=i)

        # Expectation values of Pauli‑Z as outputs
        return [qml.expval(qml.PauliZ(i)) for i in range(self.latent_dim)]

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a batch of latent vectors into reconstructions."""
        # latent shape: (batch, latent_dim)
        quantum_out = []
        for lat in latent:
            q_out = self.qnode(lat, self.weights)
            quantum_out.append(q_out)
        quantum_out = torch.stack(quantum_out)  # shape (batch, latent_dim)
        return self.classical_map(quantum_out)  # shape (batch, output_dim)

__all__ = ["QuantumDecoder"]
