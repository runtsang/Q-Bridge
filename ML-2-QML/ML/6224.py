"""
Hybrid classical module that fuses a convolutional pre‑processor with a dense autoencoder.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable

# --------------------------------------------------------------------------- #
#  Classical convolutional filter (inspired by Conv.py)
# --------------------------------------------------------------------------- #
class ConvPreprocessor(nn.Module):
    """
    Drop‑in replacement for the original Conv filter. Accepts a 2‑D tensor
    and returns a scalar activation that can be used as a feature for the
    quantum autoencoder.  The kernel size and threshold are configurable.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        """
        patch: tensor of shape (B, H, W) where H=W=kernel_size.
        Returns a scalar per batch element.
        """
        x = patch.view(-1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(x)
        act = torch.sigmoid(logits - self.threshold)
        return act.mean(dim=(1, 2, 3))

# --------------------------------------------------------------------------- #
#  Dense autoencoder (inspired by Autoencoder.py)
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        layers = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*layers)

        rev_layers = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            rev_layers.append(nn.Linear(in_dim, h))
            rev_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                rev_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        rev_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*rev_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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

__all__ = [
    "ConvPreprocessor",
    "AutoencoderNet",
    "AutoencoderConfig",
    "Autoencoder",
    "train_autoencoder",
]
