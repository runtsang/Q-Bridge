import torch
from torch import nn
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Iterable, List

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
    """Configuration values for :class:`AutoencoderGen470`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    sparse_reg: float = 0.0
    spectral_norm: bool = False

class AutoencoderGen470(nn.Module):
    """Hybrid neural‑net autoencoder with classical encoder, decoder, and optional spectral‑norm regularisation."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = nn.Sequential()
        in_dim = config.input_dim
        for idx, h in enumerate(config.hidden_dims):
            layer = nn.Linear(in_dim, h)
            if config.spectral_norm:
                layer = spectral_norm(layer)
            self.encoder.add_module(f"enc_{idx}_linear", layer)
            self.encoder.add_module(f"enc_{idx}_relu", nn.ReLU())
            if config.dropout > 0.0:
                self.encoder.add_module(f"enc_{idx}_dropout", nn.Dropout(config.dropout))
            in_dim = h
        self.latent_layer = nn.Linear(in_dim, config.latent_dim)
        self.decoder = nn.Sequential()
        in_dim = config.latent_dim
        for idx, h in enumerate(reversed(config.hidden_dims)):
            layer = nn.Linear(in_dim, h)
            self.decoder.add_module(f"dec_{idx}_linear", layer)
            self.decoder.add_module(f"dec_{idx}_relu", nn.ReLU())
            if config.dropout > 0.0:
                self.decoder.add_module(f"dec_{idx}_dropout", nn.Dropout(config.dropout))
            in_dim = h
        self.decoder.add_module("dec_out", nn.Linear(in_dim, config.input_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

def autoencoder_gen470_factory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    sparse_reg: float = 0.0,
    spectral_norm: bool = False,
) -> AutoencoderGen470:
    """Return a configured :class:`AutoencoderGen470` instance."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        sparse_reg=sparse_reg,
        spectral_norm=spectral_norm,
    )
    return AutoencoderGen470(cfg)

def train_autoencoder_gen470(
    model: AutoencoderGen470,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    sparse_weight: float = 0.0,
) -> List[float]:
    """Training loop that adds an L1 sparsity penalty on latent codes."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss_recon = mse_loss(recon, batch)
            if sparse_weight > 0.0:
                z = model.encode(batch)
                loss_sparse = sparse_weight * torch.mean(torch.abs(z))
            else:
                loss_sparse = 0.0
            loss = loss_recon + loss_sparse
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "AutoencoderGen470",
    "AutoencoderConfig",
    "autoencoder_gen470_factory",
    "train_autoencoder_gen470",
    "_as_tensor",
]
