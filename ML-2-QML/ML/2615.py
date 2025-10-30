"""HybridAutoencoder – classical component of a hybrid autoencoder architecture.

The module exposes a `HybridAutoencoder` class that contains:
- a PyTorch `AutoencoderNet` (fully‑connected encoder/decoder)
- a quantum‑variational `AutoencoderQNN` that refines the latent code
  via a swap‑test circuit.
The two halves are trained jointly: the classical network first
produces a latent vector, the quantum circuit operates on that vector
and returns a probability distribution; the loss is a weighted
combination of MSE and a fidelity‑based term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Import the quantum helper
from qml_autoencoder import build_autoencoder_qnn

# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

# --------------------------------------------------------------------------- #
# 1. Classical autoencoder backbone
# --------------------------------------------------------------------------- #
@dataclass
class HybridAutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    qnn_dropout: float = 0.0  # unused in this simplified version

class AutoencoderNet(nn.Module):
    """A lightweight multilayer perceptron autoencoder."""
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.encoder = self._make_layers(config.input_dim, config.hidden_dims, config.latent_dim, config.dropout)
        self.decoder = self._make_layers(config.latent_dim, list(reversed(config.hidden_dims)), config.input_dim, config.dropout)

    def _make_layers(self, in_dim: int, hidden: Tuple[int,...], out_dim: int, dropout: float) -> nn.Sequential:
        layers: List[nn.Module] = []
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

# --------------------------------------------------------------------------- #
# 2. Hybrid model – combines classical and quantum parts
# --------------------------------------------------------------------------- #
class HybridAutoencoder(nn.Module):
    """Hybrid autoencoder that couples a classical MLP with a quantum QNN."""
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.classical = AutoencoderNet(config)
        self.quantum = build_autoencoder_qnn(config.latent_dim)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (reconstruction, quantum_probability)."""
        latent = self.classical.encode(inputs)
        probs = self.quantum(latent).squeeze(-1)  # shape (batch,)
        recon = self.classical.decode(latent)
        return recon, probs

# --------------------------------------------------------------------------- #
# 3. Training helper
# --------------------------------------------------------------------------- #
def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    lambda_fidelity: float = 0.1,
    device: torch.device | None = None,
) -> List[float]:
    """Train the hybrid autoencoder with a combined MSE + fidelity loss."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    eps = 1e-8
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, probs = model(batch)
            loss_mse = mse_loss(recon, batch)
            loss_fid = -torch.log(probs + eps).mean()  # encourage high probability
            loss = loss_mse + lambda_fidelity * loss_fid
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderConfig",
    "AutoencoderNet",
    "train_hybrid_autoencoder",
]
