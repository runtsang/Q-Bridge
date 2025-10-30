"""Hybrid classical regressor with auto‑encoder feature extraction.

This module defines a `HybridEstimator` that combines a lightweight
auto‑encoder (from the original Autoencoder.py seed) with a linear
regression head.  The encoder compresses the input into a latent
representation; the regressor then maps this representation to a
scalar target.  The design mirrors EstimatorQNN’s regression
behaviour while inheriting the dimensionality‑reduction benefits of
the auto‑encoder.

The model is fully PyTorch‑based and can be trained on any
device that PyTorch supports.
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

# --- Auto‑encoder definition (adapted from reference 2) ---

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
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Lightweight multilayer perceptron auto‑encoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
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

# --- Hybrid regressor definition ---

class HybridEstimator(nn.Module):
    """
    Combines an auto‑encoder encoder with a linear regression head.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    latent_dim : int, default 32
        Size of the latent representation learned by the encoder.
    hidden_dims : Tuple[int, int], default (128, 64)
        Hidden layer sizes for the auto‑encoder.
    dropout : float, default 0.1
        Dropout probability within the auto‑encoder.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.autoencoder = Autoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.regressor = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a scalar prediction for each sample."""
        latent = self.autoencoder.encode(x)
        return self.regressor(latent)

    # Training helper
    def fit(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        *,
        epochs: int = 200,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: torch.device | None = None,
    ) -> List[float]:
        """Train the hybrid model and return the loss history."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        dataset = TensorDataset(_as_tensor(data), _as_tensor(targets))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        history: List[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                preds = self(batch_x).squeeze(-1)
                loss = loss_fn(preds, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history

    def predict(self, data: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        """Return predictions for the supplied data."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        with torch.no_grad():
            return self(_as_tensor(data).to(device)).cpu().squeeze(-1)

__all__ = ["HybridEstimator"]
