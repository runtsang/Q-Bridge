"""Enhanced autoencoder with early‑stopping and latent regularization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# --------------------------------------------------------------------------- #
# Helper utilities
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
# Configuration dataclass
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Configuration values for :class:`Autoencoder__gen612`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    # New: latent regularization strength
    reg_strength: float = 0.0
    # New: early‑stop patience
    patience: int = 10


# --------------------------------------------------------------------------- #
# Core model
# --------------------------------------------------------------------------- #
class Autoencoder__gen612(nn.Module):
    """A lightweight multilayer perceptron autoencoder with optional regularization."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        # Encoder
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def Autoencoder__gen612_factory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    reg_strength: float = 0.0,
    patience: int = 10,
) -> Autoencoder__gen612:
    """Factory that creates a configured autoencoder."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        reg_strength=reg_strength,
        patience=patience,
    )
    return Autoencoder__gen612(cfg)


def train_autoencoder__gen612(
    model: Autoencoder__gen612,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    val_split: float = 0.1,
    patience: int = 10,
) -> list[float]:
    """Train the autoencoder with early‑stopping and latent regularization."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    # split
    n = len(dataset)
    val_n = int(n * val_split)
    train_n = n - val_n
    train_ds, val_ds = random_split(dataset, [train_n, val_n])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []
    best_val = float("inf")
    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            # latent regularization
            if model.config.reg_strength > 0.0:
                latent = model.encode(batch)
                loss += model.config.reg_strength * torch.mean(latent.pow(2))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= train_n
        history.append(epoch_loss)
        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss = loss_fn(recon, batch)
                if model.config.reg_strength > 0.0:
                    latent = model.encode(batch)
                    loss += model.config.reg_strength * torch.mean(latent.pow(2))
                val_loss += loss.item() * batch.size(0)
        val_loss /= val_n
        # early stopping
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    return history


__all__ = [
    "Autoencoder__gen612",
    "AutoencoderConfig",
    "Autoencoder__gen612_factory",
    "train_autoencoder__gen612",
]
