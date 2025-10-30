"""Quantum autoencoder using Pennylane."""

import pennylane as qml
import numpy as np
import torch
from torch import Tensor
from typing import Callable

class Autoencoder:
    """A variational quantum autoencoder that returns a latent vector."""

    def __init__(self, latent_dim: int = 3, n_qubits: int = 5, device: str = "default.qubit", shots: int = 1000):
        self.latent_dim = latent_dim
        self.n_qubits = n_qubits
        self.device = device
        self.shots = shots
        # create a quantum device
        self.dev = qml.device(device, wires=n_qubits, shots=shots)
        # parameters for the variational circuit
        self.params = np.random.randn(n_qubits * 3) * 0.01
        # compile the QNode
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, x: Tensor, params: Tensor) -> Tensor:
        """Circuit that encodes input x and applies variational layers."""
        # Encode input as rotation angles
        for i in range(len(x)):
            qml.RY(x[i], wires=i)
        # Variational layers
        for i in range(self.n_qubits):
            qml.RZ(params[3 * i], wires=i)
            qml.RX(params[3 * i + 1], wires=i)
            qml.RZ(params[3 * i + 2], wires=i)
        # Entangle
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        # Measure expectation of Z on first latent_dim qubits to produce latent vector
        return [qml.expval(qml.PauliZ(i)) for i in range(self.latent_dim)]

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the quantum circuit on input x."""
        return self.qnode(x, self.params)

    def decode(self, z: Tensor) -> Tensor:
        """Simple linear decoder for demonstration."""
        return z @ torch.eye(z.shape[-1], z.shape[-1])

    def loss(self, x: Tensor, target: Tensor) -> Tensor:
        """Return a loss combining reconstruction and a quantum regularizer."""
        z = self.forward(x)
        recon = self.decode(z)
        recon_loss = torch.mean((recon - target) ** 2)
        reg = torch.mean(z ** 2)
        return recon_loss + 0.1 * reg

    def train(self, data: Tensor, epochs: int = 50, lr: float = 0.01):
        opt = torch.optim.Adam([self.params], lr=lr)
        for epoch in range(epochs):
            opt.zero_grad()
            loss = self.loss(data, data)
            loss.backward()
            opt.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: loss={loss.item():.4f}")

    def quantum_regularizer(self, z: Tensor) -> Tensor:
        """Return a quantum penalty for latent vectors z."""
        return torch.mean(z ** 2)

__all__ = ["Autoencoder"]
