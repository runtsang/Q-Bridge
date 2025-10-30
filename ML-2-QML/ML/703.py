import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple

# The quantum encoder is defined in the separate qml module
from qml_code import QuantumLatentEncoder

@dataclass
class AutoencoderConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    num_qbits: int = 4
    num_layers: int = 2  # number of variational layers in the quantum circuit


class HybridAutoencoder(nn.Module):
    """A hybrid autoencoder that uses a classical encoder, a quantum latent
    representation, and a classical decoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        # Classical encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[1], config.latent_dim),
        )
        # Quantum encoder
        self.quantum_encoder = QuantumLatentEncoder(
            num_qubits=config.num_qbits,
            latent_dim=config.latent_dim,
            num_layers=config.num_layers
        )
        # Classical decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.num_qbits, config.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[1], config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[0], config.input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent_classical = self.encoder(x)
        latent_quantum = self.quantum_encoder(latent_classical)
        return self.decoder(latent_quantum)


def train_hybrid_autoencoder(
    model: nn.Module,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> list[float]:
    """Train the hybrid autoencoder and return the loss history."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
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


__all__ = ["AutoencoderConfig", "HybridAutoencoder", "train_hybrid_autoencoder"]
