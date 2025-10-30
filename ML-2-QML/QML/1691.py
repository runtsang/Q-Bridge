"""Quantum autoencoder using Pennylane with a trainable variational circuit."""

import pennylane as qml
import torch
import numpy as np
from torch import nn


class AutoencoderQNN:
    """A simple quantum autoencoder implemented with Pennylane."""

    def __init__(self, n_qubits: int, reps: int = 3, lr: float = 0.01, epochs: int = 200):
        self.n_qubits = n_qubits
        self.reps = reps
        self.lr = lr
        self.epochs = epochs

        # Device and weights
        self.dev = qml.device("default.qubit", wires=n_qubits)
        # Initialize trainable weights as a torch tensor
        self.weight_params = torch.randn(n_qubits * reps, dtype=torch.float32, requires_grad=True)

        # QNode interface set to 'torch' for autograd support
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

        # Optimiser
        self.optimizer = torch.optim.Adam([self.weight_params], lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def _circuit(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Quantum circuit that encodes input `x` and outputs probabilities."""
        # Feature map: simple RY rotation per qubit
        for i in range(self.n_qubits):
            qml.RY(x[i], wires=i)

        # Variational ansatz
        idx = 0
        for _ in range(self.reps):
            for i in range(self.n_qubits):
                qml.RZ(weights[idx], wires=i)
                idx += 1
            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        # Output: probability of |0> on each qubit
        return torch.stack([qml.probs(wires=i) for i in range(self.n_qubits)], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the quantum circuit and return the output probabilities."""
        return self.qnode(x, self.weight_params)

    def fit(self, data: np.ndarray | torch.Tensor) -> list[float]:
        """Train the QNN to reconstruct the input data."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data.astype(np.float32))
        history: list[float] = []

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for x in data:
                x = x.to(torch.float32)
                self.optimizer.zero_grad()
                output = self.forward(x)
                loss = self.loss_fn(output, x)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(data)
            history.append(epoch_loss)
        return history

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent representation (probabilities) for input `x`."""
        return self.forward(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent vector back to data space (identity for this simple model)."""
        # In this toy example the decoder is identical to the forward pass
        return latent


__all__ = ["AutoencoderQNN"]
