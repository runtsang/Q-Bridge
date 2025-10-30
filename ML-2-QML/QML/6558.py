"""Variational quantum auto‑encoder built with Pennylane.

The quantum encoder maps a classical vector to a latent quantum state; a classical decoder reconstructs the input from the measurement outcomes. The circuit is trained end‑to‑end with autograd.
"""

import pennylane as qml
import torch
from torch import nn
from typing import List, Optional

class QuantumAutoencoder(nn.Module):
    """Hybrid quantum‑classical auto‑encoder."""
    def __init__(
        self,
        n_qubits: int,
        latent_dim: int,
        reps: int = 2,
        device: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.latent_dim = latent_dim
        self.reps = reps
        self.dev = qml.device(device, wires=n_qubits)

        # Parameterised quantum circuit
        self.qnode = qml.QNode(self._quantum_encoder, self.dev, interface="torch")

        # Classical decoder
        decoder_layers: List[nn.Module] = []
        in_dim = latent_dim
        hidden = n_qubits * 2
        decoder_layers.append(nn.Linear(in_dim, hidden))
        decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(hidden, n_qubits))
        self.decoder = nn.Sequential(*decoder_layers)

    def _quantum_encoder(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Encodes the input vector into a quantum state and measures."""
        for i in range(self.n_qubits):
            qml.Hadamard(wires=i)
            qml.RX(x[i], wires=i)
            qml.RY(x[i], wires=i)
        qml.RealAmplitudes(weights, wires=range(self.n_qubits), reps=self.reps)
        return qml.expval(qml.PauliZ(wires=list(range(self.n_qubits))))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent vector from the quantum circuit."""
        weights = torch.randn(self.n_qubits * self.reps, requires_grad=True)
        return self.qnode(x, weights)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from latent vector."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        return self.decode(latent)

def QuantumAutoencoder(
    n_qubits: int,
    latent_dim: int,
    *,
    reps: int = 2,
    device: str = "default.qubit",
) -> QuantumAutoencoder:
    """Factory returning a configured quantum auto‑encoder."""
    return QuantumAutoencoder(n_qubits, latent_dim, reps=reps, device=device)

def train_quantum_autoencoder(
    model: QuantumAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> List[float]:
    """Training loop for the quantum auto‑encoder using autograd."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x in data:
            x = x.to(device)
            optimizer.zero_grad()
            recon = model(x)
            loss = loss_fn(recon, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(data)
        history.append(epoch_loss)

    return history

__all__ = ["QuantumAutoencoder", "train_quantum_autoencoder", "QuantumAutoencoder"]
