"""Classical auto‑encoder with denoising, skip‑connections and an early‑stopping training loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
    """Configuration for the classical auto‑encoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    noise_factor: float = 0.0
    skip_connections: bool = False

class AutoencoderNet(nn.Module):
    """Fully‑connected auto‑encoder with optional denoising and skip connections."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _build_encoder(self) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_dim = self.config.input_dim
        for h in self.config.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if self.config.dropout > 0.0:
                layers.append(nn.Dropout(self.config.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, self.config.latent_dim))
        return nn.Sequential(*layers)

    def _build_decoder(self) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_dim = self.config.latent_dim
        for h in reversed(self.config.hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if self.config.dropout > 0.0:
                layers.append(nn.Dropout(self.config.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, self.config.input_dim))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        if self.config.skip_connections:
            # simple skip: concatenate latent with input
            z = torch.cat([z, x], dim=-1)
        return self.decode(z)

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    noise_factor: float = 0.0,
    skip_connections: bool = False,
) -> AutoencoderNet:
    """Factory returning a configured auto‑encoder."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        noise_factor=noise_factor,
        skip_connections=skip_connections,
    )
    return AutoencoderNet(config)

def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    early_stopping: bool = False,
    patience: int = 10,
    val_data: Optional[torch.Tensor] = None,
) -> List[float]:
    """Train the auto‑encoder with optional early stopping."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Add Gaussian noise if configured
    if model.config.noise_factor > 0.0:
        noise = torch.randn_like(data) * model.config.noise_factor
        noisy_data = data + noise
    else:
        noisy_data = data

    dataset = TensorDataset(_as_tensor(noisy_data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    best_val_loss = float("inf")
    counter = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

        # Early stopping
        if early_stopping and val_data is not None:
            model.eval()
            with torch.no_grad():
                val_recon = model(_as_tensor(val_data).to(device))
                val_loss = loss_fn(val_recon, _as_tensor(val_data).to(device)).item()
            model.train()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    return history

__all__ = ["Autoencoder", "AutoencoderNet", "AutoencoderConfig", "train_autoencoder"]
