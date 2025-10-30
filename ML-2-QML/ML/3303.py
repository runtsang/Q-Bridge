"""quantum_hybrid_autoencoder.py"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from quantum_encoder_circuit import get_quantum_encoder

@dataclass
class QuantumHybridAutoencoderConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    quantum_latent_dim: int = 8  # number of qubits used in the quantum block

class QuantumHybridAutoencoder(nn.Module):
    """Hybrid autoencoder that uses a classical encoder/decoder and a quantum encoder."""
    def __init__(self, config: QuantumHybridAutoencoderConfig) -> None:
        super().__init__()
        self.encoder = self._build_encoder(config)
        # quantum encoder maps classical latent vector to quantum latent vector
        self.quantum_encoder = get_quantum_encoder(
            num_qubits=config.quantum_latent_dim,
            input_dim=config.latent_dim,
        )
        self.decoder = self._build_decoder(config)

    @staticmethod
    def _build_encoder(config: QuantumHybridAutoencoderConfig) -> nn.Sequential:
        layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, config.latent_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _build_decoder(config: QuantumHybridAutoencoderConfig) -> nn.Sequential:
        layers = []
        in_dim = config.quantum_latent_dim
        for hidden in reversed(config.hidden_dims):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, config.input_dim))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical encode
        latent = self.encoder(x)
        # Quantum encode (detached to avoid backprop through quantum circuit)
        q_latent_np = self.quantum_encoder(latent.detach().cpu().numpy())
        q_latent = torch.tensor(q_latent_np, dtype=x.dtype, device=x.device)
        # Classical decode
        return self.decoder(q_latent)

def train_hybrid_autoencoder(
    model: QuantumHybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    """Simple training loop for the hybrid autoencoder."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

def _as_tensor(data: torch.Tensor | Iterable[float]) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

__all__ = [
    "QuantumHybridAutoencoder",
    "QuantumHybridAutoencoderConfig",
    "train_hybrid_autoencoder",
]
