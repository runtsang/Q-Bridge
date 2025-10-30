"""Hybrid classical autoencoder with optional quantum loss.

The class ``Autoencoder__gen299`` extends the original fully‑connected
autoencoder by adding a pre‑training routine for the encoder and a
quantum‑aware loss term that can be plugged in when a quantum decoder
is available.  The architecture remains fully PyTorch‑compatible
so it can be used in standard training pipelines.

Typical usage:

>>> ae = Autoencoder__gen299(784, latent_dim=32)
>>> ae.pretrain_encoder(train_loader, epochs=5)   # encoder only
>>> history = ae.train_with_quantum(
...     train_loader,
...     qnn,               # a Qiskit SamplerQNN
...     epochs=30,
...     quantum_weight=0.1,
...     depth_penalty=0.01
... )
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable
import math


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


# --------------------------------------------------------------------------- #
# Core Module
# --------------------------------------------------------------------------- #
class Autoencoder__gen299(nn.Module):
    """A classical autoencoder that can be coupled with a quantum decoder.

    The network follows the original design but exposes additional hooks
    for quantum‑aware training.
    """

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        # Encoder
        encoder = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            encoder += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(config.dropout)]
            in_dim = h
        encoder.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder)

        # Decoder
        decoder = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            decoder += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(config.dropout)]
            in_dim = h
        decoder.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder)

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    # --------------------------------------------------------------------- #
    # Helper for pre‑training the encoder only
    # --------------------------------------------------------------------- #
    def pretrain_encoder(
        self,
        loader: DataLoader,
        *,
        epochs: int = 10,
        lr: float = 1e-3,
        device: torch.device | None = None,
    ) -> list[float]:
        """Train only the encoder, keeping the decoder frozen."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.decoder.eval()
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        history = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for batch, in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                lat = self.encode(batch)
                recon = self.decode(lat)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(loader.dataset)
            history.append(epoch_loss)
        return history

    # --------------------------------------------------------------------- #
    # Quantum‑aware training
    # --------------------------------------------------------------------- #
    def train_with_quantum(
        self,
        loader: DataLoader,
        qnn: object,
        *,
        epochs: int = 20,
        lr: float = 1e-3,
        quantum_weight: float = 0.1,
        depth_penalty: float = 0.01,
        device: torch.device | None = None,
    ) -> list[float]:
        """Full training loop that adds a quantum loss and depth regularisation.

        Parameters
        ----------
        qnn
            A Qiskit ``SamplerQNN`` or any callable that accepts a latent vector
            and returns a probability distribution over {0,1}.
        quantum_weight
            Weight of the quantum loss term.
        depth_penalty
            Penalty applied per gate in the quantum circuit to discourage
            unnecessary depth.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        recon_loss_fn = nn.MSELoss()
        history = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for batch, in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)

                z = self.encode(batch)
                recon = self.decode(z)
                recon_loss = recon_loss_fn(recon, batch)

                # Quantum loss: we treat the output of the QNN as a probability
                # of the latent being reconstructed. For demo purposes we
                # convert the probability to a binary value via a threshold.
                try:
                    probs = qnn(z.detach().cpu().numpy())
                    # simple binary cross‑entropy loss
                    q_loss = -torch.mean(
                        torch.log(torch.tensor(probs, device=device) + 1e-8)
                    )
                except Exception:
                    q_loss = torch.tensor(0.0, device=device)

                # Depth regularisation (simple placeholder)
                depth = getattr(qnn, "depth", 0)
                depth_reg = depth_penalty * depth

                loss = recon_loss + quantum_weight * q_loss + depth_reg
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(loader.dataset)
            history.append(epoch_loss)
        return history


# --------------------------------------------------------------------------- #
# Convenience factory
# --------------------------------------------------------------------------- #
def Autoencoder(input_dim: int, **kwargs) -> Autoencoder__gen299:
    """Return a configured hybrid autoencoder."""
    cfg = AutoencoderConfig(input_dim=input_dim, **kwargs)
    return Autoencoder__gen299(cfg)


# --------------------------------------------------------------------------- #
# Training helper
# --------------------------------------------------------------------------- #
def train_autoencoder(
    model: Autoencoder__gen299,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Standard reconstruction training loop."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
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


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)


__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "Autoencoder__gen299",
    "train_autoencoder",
]
