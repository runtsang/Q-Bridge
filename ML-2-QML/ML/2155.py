"""Hybrid classical‑quantum autoencoder.

The model consists of a standard multilayer perceptron encoder/decoder and an
optional variational quantum circuit that refines the latent representation.
The circuit is built using Pennylane and is differentiable via the
parameter‑shift rule, allowing end‑to‑end training with PyTorch optimisers.

The public API mirrors the original seed:
    - Autoencoder(...)
    - AutoencoderConfig
    - AutoencoderNet
    - train_autoencoder(...)
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Iterable, Tuple, List, Optional

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
# Configuration
# --------------------------------------------------------------------------- #
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderNet`."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        qlayer: bool = False,
        qnum: int = 0,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.qlayer = qlayer
        self.qnum = qnum

# --------------------------------------------------------------------------- #
# Quantum helper
# --------------------------------------------------------------------------- #
# The quantum circuit is defined in the separate QML module to keep
# the classical and quantum codebases independent.
try:
    from qml_code import quantum_refiner_module  # type: ignore
except Exception:
    def quantum_refiner_module(*_, **__):
        raise RuntimeError(
            "Quantum refiner module is missing. Install the QML dependency "
            "or provide a custom implementation."
        )

# --------------------------------------------------------------------------- #
# Encoder / Decoder
# --------------------------------------------------------------------------- #
def _build_encoder(config: AutoencoderConfig) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_dim = config.input_dim
    for hidden in config.hidden_dims:
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.ReLU())
        if config.dropout > 0.0:
            layers.append(nn.Dropout(config.dropout))
        in_dim = hidden
    layers.append(nn.Linear(in_dim, config.latent_dim))
    return nn.Sequential(*layers)

def _build_decoder(config: AutoencoderConfig) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_dim = config.latent_dim
    for hidden in reversed(config.hidden_dims):
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.ReLU())
        if config.dropout > 0.0:
            layers.append(nn.Dropout(config.dropout))
        in_dim = hidden
    layers.append(nn.Linear(in_dim, config.input_dim))
    return nn.Sequential(*layers)

# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #
class AutoencoderNet(nn.Module):
    """Hybrid classical‑quantum autoencoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = _build_encoder(config)
        self.decoder = _build_decoder(config)
        # Optional quantum refinement layer
        if config.qlayer and config.qnum > 0:
            self.quantum_refiner = quantum_refiner_module(
                latent_dim=config.latent_dim,
                qnum=config.qnum,
            )
        else:
            self.quantum_refiner = None

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def refine(self, latents: torch.Tensor) -> torch.Tensor:
        if self.quantum_refiner is None:
            return latents
        return self.quantum_refiner(latents)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.encode(inputs)
        refined = self.refine(latent)
        return self.decode(refined)

# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    qlayer: bool = False,
    qnum: int = 0,
) -> AutoencoderNet:
    """Return a configured :class:`AutoencoderNet`."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        qlayer=qlayer,
        qnum=qnum,
    )
    return AutoencoderNet(cfg)

# --------------------------------------------------------------------------- #
# Training helper
# --------------------------------------------------------------------------- #
def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Simple reconstruction training loop returning the loss history."""
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
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "train_autoencoder",
]
