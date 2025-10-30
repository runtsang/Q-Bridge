"""Hybrid classical/quantum estimator – classical side.

The module defines a lightweight auto‑encoder, a linear regressor, and a
training routine that jointly optimises both components.  The architecture
mirrors the two seed projects: it inherits the AutoencoderNet from the
second seed and the simple feed‑forward EstimatorQNN from the first.

The final network is a single :class:`HybridEstimatorNet` that can be
exported as a PyTorch ``nn.Module`` and trained with standard optimisers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# 1. Auto‑encoder (from Autoencoder.py seed)
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Multilayer perceptron auto‑encoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = self._build_mlp(
            config.input_dim,
            config.hidden_dims,
            config.latent_dim,
            config.dropout,
            encode=True,
        )
        self.decoder = self._build_mlp(
            config.latent_dim,
            tuple(reversed(config.hidden_dims)),
            config.input_dim,
            config.dropout,
            encode=False,
        )

    def _build_mlp(self, in_dim: int, hidden: Tuple[int,...], out_dim: int,
                   dropout: float, encode: bool) -> nn.Sequential:
        layers = []
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    return AutoencoderNet(
        AutoencoderConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    )

# --------------------------------------------------------------------------- #
# 2. Classical estimator (from EstimatorQNN.py seed)
# --------------------------------------------------------------------------- #
class ClassicalEstimator(nn.Module):
    """A shallow linear regressor that consumes the auto‑encoder latent."""
    def __init__(self, latent_dim: int, output_dim: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

# --------------------------------------------------------------------------- #
# 3. Hybrid network
# --------------------------------------------------------------------------- #
class HybridEstimatorNet(nn.Module):
    """Combines an auto‑encoder with a linear regressor."""
    def __init__(self, input_dim: int, latent_dim: int = 32) -> None:
        super().__init__()
        self.autoencoder = Autoencoder(input_dim, latent_dim=latent_dim)
        self.estimator = ClassicalEstimator(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.autoencoder.encode(x)
        return self.estimator(z)

def HybridEstimator(input_dim: int, latent_dim: int = 32) -> HybridEstimatorNet:
    """Factory that returns a fully‑configured hybrid network."""
    return HybridEstimatorNet(input_dim, latent_dim)

# --------------------------------------------------------------------------- #
# 4. Training routine (joint optimisation)
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        t = data
    else:
        t = torch.as_tensor(data, dtype=torch.float32)
    return t.to(dtype=torch.float32)

def train_hybrid(
    model: HybridEstimatorNet,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    """
    Joint training loop for the hybrid network.

    Parameters
    ----------
    model : HybridEstimatorNet
        The network to optimise.
    inputs : torch.Tensor
        Input features, shape ``(N, D_in)``.
    targets : torch.Tensor
        Ground‑truth regression values, shape ``(N, 1)``.
    epochs : int
        Number of epochs.
    batch_size : int
        Size of mini‑batches.
    lr : float
        Learning rate.
    device : torch.device | None
        CPU or GPU.  Defaults to CUDA if available.

    Returns
    -------
    list[float]
        Training loss history.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(inputs), _as_tensor(targets))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            optimizer.zero_grad()
            pred = model(inp)
            loss = loss_fn(pred, tgt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inp.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

    return history

__all__ = [
    "Autoencoder",
    "AutoencoderNet",
    "AutoencoderConfig",
    "ClassicalEstimator",
    "HybridEstimatorNet",
    "HybridEstimator",
    "train_hybrid",
]
