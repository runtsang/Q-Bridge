"""Enhanced classical autoencoder with residual connections and configurable features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Callable, List, Optional

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
class AutoencoderHybridConfig:
    """Configuration for :class:`AutoencoderHybridNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU
    residual: bool = True
    layer_norm: bool = False


class AutoencoderHybridNet(nn.Module):
    """An autoencoder with optional residual blocks, layer‑norm and configurable activation."""
    def __init__(self, config: AutoencoderHybridConfig) -> None:
        super().__init__()
        self.config = config
        self.activation = config.activation

        # Encoder
        enc_layers = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            if config.layer_norm:
                enc_layers.append(nn.LayerNorm(h))
            enc_layers.append(self.activation())
            if config.dropout > 0.0:
                enc_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            block = nn.Sequential(
                nn.Linear(in_dim, h),
                nn.LayerNorm(h) if config.layer_norm else nn.Identity(),
                self.activation(),
                nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity(),
            )
            if config.residual:
                block = nn.Sequential(
                    block,
                    nn.Linear(h, h),  # skip connection
                    nn.Identity()
                )
            dec_layers.append(block)
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def AutoencoderHybrid(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU,
    residual: bool = True,
    layer_norm: bool = False,
) -> AutoencoderHybridNet:
    """Convenience factory mirroring the quantum interface."""
    cfg = AutoencoderHybridConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
        residual=residual,
        layer_norm=layer_norm,
    )
    return AutoencoderHybridNet(cfg)


def train_autoencoder_hybrid(
    model: AutoencoderHybridNet,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    early_stopping: int = 20,
    verbose: bool = False,
) -> List[float]:
    """Train the hybrid autoencoder with early‑stopping and loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []
    best_loss = float("inf")
    patience = early_stopping

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

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience = early_stopping
        else:
            patience -= 1
            if patience <= 0:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss:.6f}")

    return history


__all__ = [
    "AutoencoderHybrid",
    "AutoencoderHybridNet",
    "AutoencoderHybridConfig",
    "train_autoencoder_hybrid",
]
