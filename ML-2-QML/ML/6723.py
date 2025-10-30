"""Hybrid classical estimator that fuses an autoencoder with a regression head.

The network consists of:
  * An AutoencoderNet that learns a latent representation of the input.
  * A linear regressor that maps the latent vector to a scalar output.
The architecture is designed to be lightweight while still
capturing non‑linear features via the encoder and providing a
straightforward training loop for regression tasks.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

# --------------------------------------------------------------------------- #
#  Autoencoder components (adapted from the seed Autoencoder.py)
# --------------------------------------------------------------------------- #

@dataclass
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """A lightweight multilayer perceptron autoencoder."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        # Encoder
        encoder_layers = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

# --------------------------------------------------------------------------- #
#  Classical estimator head (adapted from EstimatorQNN.py)
# --------------------------------------------------------------------------- #

class EstimatorNN(nn.Module):
    """Simple fully‑connected regression network."""
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)

# --------------------------------------------------------------------------- #
#  Hybrid estimator
# --------------------------------------------------------------------------- #

class HybridEstimatorNet(nn.Module):
    """
    Combines an autoencoder and a regression head.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input data.
    latent_dim : int, default 32
        Size of the latent space.
    hidden_dims : tuple[int, int], default (128, 64)
        Hidden layer sizes for the autoencoder.
    dropout : float, default 0.1
        Dropout probability for the autoencoder.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        cfg = AutoencoderConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.autoencoder = AutoencoderNet(cfg)
        self.regressor = EstimatorNN(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the regression output."""
        latent = self.autoencoder.encode(x)
        return self.regressor(latent)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Expose the encoder for downstream use."""
        return self.autoencoder.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Expose the decoder for reconstruction."""
        return self.autoencoder.decode(z)

# --------------------------------------------------------------------------- #
#  Convenience training utilities
# --------------------------------------------------------------------------- #

def train_hybrid(
    model: HybridEstimatorNet,
    data: torch.Tensor,
    targets: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """
    Train the hybrid estimator on regression data.

    Parameters
    ----------
    model : HybridEstimatorNet
        The model to train.
    data : torch.Tensor
        Input features.
    targets : torch.Tensor
        Ground‑truth scalar targets.
    epochs : int
        Number of epochs.
    batch_size : int
        Batch size.
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay for Adam.
    device : torch.device | None
        Device to run on; defaults to CUDA if available.

    Returns
    -------
    history : list[float]
        Training loss per epoch.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(
        _as_tensor(data).to(device), _as_tensor(targets).to(device)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for X, y in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

def _as_tensor(data: torch.Tensor | Iterable[float]) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

__all__ = [
    "AutoencoderConfig",
    "AutoencoderNet",
    "EstimatorNN",
    "HybridEstimatorNet",
    "train_hybrid",
]
