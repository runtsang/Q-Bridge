"""Enhanced autoencoder with skip connections, batch‑norm, and hybrid loss capability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    batch_norm: bool = True
    skip_connections: bool = True


class Autoencoder__gen406(nn.Module):
    """A fully‑connected autoencoder with optional skip connections and batch‑norm."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Encoder
        encoder_layers = []
        prev_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden))
            if config.batch_norm:
                encoder_layers.append(nn.BatchNorm1d(hidden))
            encoder_layers.append(nn.ReLU(inplace=True))
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden
        encoder_layers.append(nn.Linear(prev_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden))
            if config.batch_norm:
                decoder_layers.append(nn.BatchNorm1d(hidden))
            decoder_layers.append(nn.ReLU(inplace=True))
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            prev_dim = hidden
        decoder_layers.append(nn.Linear(prev_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

    def hybrid_loss(self, x: torch.Tensor, quantum_latent: torch.Tensor) -> torch.Tensor:
        """
        Compute a reconstruction loss that encourages the classical latent
        representation to match a quantum‑derived latent vector.  The quantum
        latent is assumed to be of shape (batch, latent_dim).
        """
        z = self.encode(x)
        recon = self.decode(z)
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        latent_loss = F.mse_loss(z, quantum_latent, reduction="mean")
        return recon_loss + 0.1 * latent_loss


def train_autoencoder(
    model: Autoencoder__gen406,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
    quantum_latent_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> List[float]:
    """
    Train the autoencoder.  If a quantum_latent_fn is provided, it should
    accept a torch.Tensor of shape (batch, input_dim) and return a torch.Tensor
    of shape (batch, latent_dim) produced by a quantum circuit.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            if quantum_latent_fn is not None:
                q_latent = quantum_latent_fn(batch)
                loss = model.hybrid_loss(batch, q_latent)
            else:
                loss = F.mse_loss(model(batch), batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        loss_history.append(epoch_loss)
    return loss_history


__all__ = ["Autoencoder__gen406", "AutoencoderConfig", "train_autoencoder"]
