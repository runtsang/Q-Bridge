"""Quantum decoder for the VAE latent space using Pennylane."""

from __future__ import annotations

import pennylane as qml
import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple, List

__all__ = ["AutoencoderGen", "AutoencoderConfig", "train_quantum_decoder"]

@dataclass
class AutoencoderConfig:
    """Configuration shared with the classical VAE."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 64
    weight_decay: float = 0.0

class AutoencoderGen(nn.Module):
    """Quantum decoder that maps latent vector to reconstructed data."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.n_qubits = config.input_dim
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.latent_dim = config.latent_dim
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, latent: torch.Tensor) -> List[torch.Tensor]:
        for i, angle in enumerate(latent):
            qml.RX(angle, wires=i % self.n_qubits)
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.qnode(latent)

def train_quantum_decoder(
    model: AutoencoderGen,
    latent_samples: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device | None = None,
) -> List[float]:
    """Train the quantum decoder to reconstruct targets from latent samples."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.lr, weight_decay=model.config.weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    dataset = torch.utils.data.TensorDataset(latent_samples, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=model.config.batch_size, shuffle=True)

    for epoch in range(model.config.epochs):
        epoch_loss = 0.0
        for latent_batch, target_batch in loader:
            latent_batch = latent_batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad()
            recon = model(latent_batch)
            loss = loss_fn(recon, target_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(loader)
        history.append(epoch_loss)
    return history
