"""Hybrid classical autoencoder with optional quantum regularization."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable, Callable, Optional

def _to_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)

@dataclass
class AutoencoderConfig:
    """Configuration for the hybrid-classical autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    skip_connections: bool = False
    layer_norm: bool = False

class Autoencoder(nn.Module):
    """A flexible MLP autoencoder with optional skip connections and layer norm."""

    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Encoder
        self.enc_layers = nn.ModuleList()
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            self.enc_layers.append(nn.Linear(in_dim, hidden))
            if cfg.layer_norm:
                self.enc_layers.append(nn.LayerNorm(hidden))
            self.enc_layers.append(nn.ReLU())
            if cfg.dropout > 0:
                self.enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        self.enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        # Decoder
        self.dec_layers = nn.ModuleList()
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            self.dec_layers.append(nn.Linear(in_dim, hidden))
            if cfg.layer_norm:
                self.dec_layers.append(nn.LayerNorm(hidden))
            self.dec_layers.append(nn.ReLU())
            if cfg.dropout > 0:
                self.dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        self.dec_layers.append(nn.Linear(in_dim, cfg.input_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.enc_layers:
            out = layer(out)
        return out

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out = z
        for layer in self.dec_layers:
            out = layer(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        recon = self.decode(z)
        return recon

def train_autoencoder(
    model: Autoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    quantum_regularizer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    reg_weight: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Train the autoencoder with optional quantum regularization."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_to_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            if quantum_regularizer is not None and reg_weight > 0:
                z = model.encode(batch)
                reg = quantum_regularizer(z)
                loss = loss + reg_weight * reg
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = ["Autoencoder", "AutoencoderConfig", "train_autoencoder"]
