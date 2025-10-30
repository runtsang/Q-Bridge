"""Combined classical estimator with auto‑encoder inspired architecture.

The module exposes a lightweight neural network that first compresses the
input via an encoder (mirroring the AutoencoderNet) and then predicts a
scalar target.  A convenience factory and a training routine are also
provided, making the model ready for rapid prototyping and benchmarking.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple, Iterable

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class EstimatorQNNConfig:
    """Hyper‑parameters for the hybrid regressor."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1

# --------------------------------------------------------------------------- #
# Utility
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
# Model
# --------------------------------------------------------------------------- #
class EstimatorQNN(nn.Module):
    """Hybrid regressor that first encodes the input and then predicts a scalar."""
    def __init__(self, config: EstimatorQNNConfig) -> None:
        super().__init__()
        # Encoder – identical to the AutoencoderNet encoder
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

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(config.latent_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.regressor(latent)

# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def EstimatorQNNFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
) -> EstimatorQNN:
    """Convenience constructor mirroring the original EstimatorQNN."""
    cfg = EstimatorQNNConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return EstimatorQNN(cfg)

# --------------------------------------------------------------------------- #
# Training routine
# --------------------------------------------------------------------------- #
def train_estimator(
    model: EstimatorQNN,
    data: torch.Tensor,
    targets: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Simple training loop that returns a history of training losses."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data), _as_tensor(targets))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

    return history

__all__ = [
    "EstimatorQNNFactory",
    "EstimatorQNN",
    "EstimatorQNNConfig",
    "train_estimator",
    "_as_tensor",
]
