"""AutoencoderGen345: classical autoencoder with extended training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Callable, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

@dataclass
class AutoencoderGen345Config:
    """Configuration for the classical autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    sparsity_lambda: float = 0.0  # L1 penalty on latent representation

class AutoencoderGen345(nn.Module):
    """A fully‑connected autoencoder with optional sparsity regularisation."""

    def __init__(self, config: AutoencoderGen345Config) -> None:
        super().__init__()
        self.config = config
        # Encoder
        encoder_layers: List[nn.Module] = []
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
        decoder_layers: List[nn.Module] = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def train_autoencoder(
    model: AutoencoderGen345,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    callback: Optional[Callable[[int, float, torch.Tensor], None]] = None,
) -> List[float]:
    """Train the autoencoder, optionally calling a callback after each epoch.

    Parameters
    ----------
    model : AutoencoderGen345
        The model to train.
    data : torch.Tensor
        Training data of shape (N, input_dim).
    epochs : int
        Number of epochs.
    batch_size : int
        Batch size.
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay for Adam.
    device : Optional[torch.device]
        Target device.
    callback : Optional[Callable[[int, float, torch.Tensor], None]]
        Called with (epoch, loss, latent) after each epoch.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            # Optional sparsity penalty
            if model.config.sparsity_lambda > 0.0:
                latent = model.encode(batch)
                loss += model.config.sparsity_lambda * torch.mean(torch.abs(latent))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
        if callback is not None:
            with torch.no_grad():
                latent = model.encode(dataset.tensors[0].to(device))
            callback(epoch, epoch_loss, latent)
    return history

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Helper that guarantees a float32 tensor on the current device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor

__all__ = ["AutoencoderGen345", "AutoencoderGen345Config", "train_autoencoder"]
