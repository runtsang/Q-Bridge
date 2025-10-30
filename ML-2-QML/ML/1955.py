python
import torch
import torch.nn as nn
import pennylane as qml
import pennylane.numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class AutoencoderConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 8
    hidden_dims: Tuple[int,...] = (64, 32)
    dropout: float = 0.1
    n_qubits: int = 8
    latent_layers: int = 2  # number of variational layers

class Autoencoder(nn.Module):
    """Hybrid autoencoder: classical encoder → quantum decoder."""
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config

        # Classical encoder
        self.encoder = nn.Sequential(
            *self._make_layers(config.input_dim, config.hidden_dims, config.latent_dim)
        )

        # Quantum device and decoder QNode
        self.dev = qml.device("default.qubit", wires=config.n_qubits)
        self.qnode = qml.QNode(self._quantum_decoder, self.dev, interface="torch")

        # Post‑decoder linear layer mapping qubit expectations to original dimension
        self.post_decoder = nn.Linear(config.n_qubits, config.input_dim)

    def _make_layers(self, in_dim: int, hidden_dims: Tuple[int,...], out_dim: int):
        layers: list[nn.Module] = []
        last_dim = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(last_dim, h), nn.ReLU(), nn.Dropout(self.config.dropout)])
            last_dim = h
        layers.append(nn.Linear(last_dim, out_dim))
        layers.append(nn.ReLU())
        return layers

    def _quantum_decoder(self, latent: torch.Tensor, *weights: torch.Tensor):
        """
        Variational quantum decoder.
        Parameters are split into qubit rotation angles (latent) and
        trainable variational parameters (weights).
        """
        # Encode latent into first `n_qubits` qubits via RX rotations
        for i, v in enumerate(weights[:self.config.n_qubits]):
            qml.RX(latent[i], wires=i)

        # Apply a stack of strongly entangling layers
        qml.templates.StronglyEntanglingLayers(
            weights[self.config.n_qubits:],
            wires=range(self.config.n_qubits),
            reps=self.config.latent_layers,
        )

        # Return expectation values of PauliZ on each qubit
        return torch.stack([qml.expval(qml.PauliZ(i)) for i in range(self.config.n_qubits)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        # Truncate or pad latent to match qubit count
        latent = latent[:, :self.config.n_qubits]
        q_out = self.qnode(latent)
        out = self.post_decoder(q_out)
        return out

def train_autoencoder(
    model: Autoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    """Simple reconstruction training loop returning the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = ["Autoencoder", "AutoencoderConfig", "train_autoencoder"]
