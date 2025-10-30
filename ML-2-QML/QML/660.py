import pennylane as qml
import torch
from torch import nn
from typing import Tuple

__all__ = ["QuantumLatent"]


class QuantumLatent(nn.Module):
    """Quantum latent representation implemented with Pennylane."""

    def __init__(self, latent_dim: int, depth: int = 2, device: str = "default.qubit"):
        super().__init__()
        self.latent_dim = latent_dim
        self.depth = depth
        self.dev = qml.device(device, wires=latent_dim)
        self.params = nn.Parameter(torch.randn(depth, latent_dim, 3))
        self._qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, x: torch.Tensor, *params: torch.Tensor):
        # Encode classical features into quantum state
        for i in range(self.latent_dim):
            qml.RY(x[i], wires=i)
        # Variational ansatz
        qml.templates.StronglyEntanglingLayers(params, wires=range(self.latent_dim))
        # Measurements
        return [qml.expval(qml.PauliZ(i)) for i in range(self.latent_dim)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, latent_dim)
        batch_size = x.shape[0]
        out = torch.stack([self._qnode(x[i], self.params) for i in range(batch_size)])
        return out
