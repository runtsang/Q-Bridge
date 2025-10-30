"""Hybrid classical‑quantum autoencoder with regression head.

The module defines a PyTorch model that mirrors the structure of the original
Autoencoder but injects a quantum latent layer (to be supplied by the QML
module).  It also exposes a small Estimator‑style regressor that can be used
to predict continuous targets from the latent representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# Helper: convert arbitrary data to a float32 tensor
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
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
class HybridAutoEncoderConfig:
    """Configuration for :class:`HybridAutoEncoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    # number of qubits used in the quantum latent circuit
    qnn_qubits: int = 4

# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #
class HybridAutoEncoderNet(nn.Module):
    """Classical encoder → (quantum latent) → classical decoder.

    The quantum part is represented by a separate circuit that is supplied by
    the QML module.  The forward pass currently skips the quantum step and
    operates purely classically; users can insert a QNN wrapper around the
    `encode` output when training with a hybrid backend.
    """
    def __init__(self, config: HybridAutoEncoderConfig) -> None:
        super().__init__()
        self.config = config

        # Encoder
        enc_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                enc_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers: List[nn.Module] = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                dec_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # Estimator head (mirrors EstimatorQNN)
        self.regressor = nn.Sequential(
            nn.Linear(config.latent_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical reconstruction path
        z = self.encode(x)
        # Quantum processing would occur here in a hybrid training loop
        return self.decode(z)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return regression prediction from latent representation."""
        z = self.encode(x)
        return self.regressor(z)

# --------------------------------------------------------------------------- #
# Training utilities
# --------------------------------------------------------------------------- #
def train_autoencoder(
    model: HybridAutoEncoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Simple reconstruction training loop."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
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
    return history

def train_regressor(
    model: HybridAutoEncoderNet,
    data: torch.Tensor,
    targets: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Train only the regression head."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data), _as_tensor(targets))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.regressor.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch, target) in loader:
            batch, target = batch.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model.predict(batch)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "HybridAutoEncoderConfig",
    "HybridAutoEncoderNet",
    "train_autoencoder",
    "train_regressor",
]
