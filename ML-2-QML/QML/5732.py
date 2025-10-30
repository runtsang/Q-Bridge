"""AutoencoderGen117: hybrid quantum‑classical autoencoder using Pennylane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import pennylane as qml
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

__all__ = [
    "AutoencoderGen117",
    "AutoencoderConfig",
    "train_autoencoder",
]


@dataclass
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderGen117`."""

    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    kl_weight: float = 0.0
    """Weight for the KL‑divergence term when training a VAE."""
    # Quantum decoder settings
    quantum_decoder: bool = True
    """Whether to use a quantum circuit for the decoder."""
    quantum_reps: int = 2
    """Number of variational repetitions in the quantum decoder."""


class QuantumDecoder(nn.Module):
    """A Pennylane variational circuit that maps latent vectors to reconstructions."""

    def __init__(self, latent_dim: int, output_dim: int, reps: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.reps = reps
        # Device with as many wires as output dimension
        self.dev = qml.device("default.qubit", wires=output_dim)
        # Weight matrix for the variational parameters
        self.weight = nn.Parameter(
            torch.randn(reps * 2 * output_dim)
        )  # 2 rotations per qubit per layer
        # QNode that returns expectation values of Pauli‑Z
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, latent: torch.Tensor, weights: torch.Tensor) -> list[torch.Tensor]:
        """Encode the latent vector and apply a variational ansatz."""
        # Angle‑encoding of the latent vector
        for i in range(self.latent_dim):
            qml.RY(latent[i], wires=i)
        # Variational layers
        idx = 0
        for _ in range(self.reps):
            for i in range(self.output_dim):
                qml.Rot(weights[idx], weights[idx + 1], weights[idx + 2], wires=i)
                idx += 3
            for i in range(self.output_dim - 1):
                qml.CNOT(wires=[i, i + 1])
        # Measure expectation values of Pauli‑Z
        return [qml.expval(qml.PauliZ(i)) for i in range(self.output_dim)]

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Return the reconstructed vector from the quantum decoder."""
        return self.qnode(latent, self.weight)


class AutoencoderGen117(nn.Module):
    """Hybrid VAE: classical encoder + quantum decoder."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = self._build_encoder(config)
        if config.quantum_decoder:
            self.decoder = QuantumDecoder(
                latent_dim=config.latent_dim,
                output_dim=config.input_dim,
                reps=config.quantum_reps,
            )
        else:
            self.decoder = self._build_decoder(config)

    # ------------------------------------------------------------------ #
    # Encoder (identical to classical version)
    # ------------------------------------------------------------------ #
    def _build_encoder(self, cfg: AutoencoderConfig) -> nn.Module:
        layers: list[nn.Module] = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, cfg.latent_dim * 2))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ------------------------------------------------------------------ #
    # Decoder (classical or quantum)
    # ------------------------------------------------------------------ #
    def _build_decoder(self, cfg: AutoencoderConfig) -> nn.Module:
        layers: list[nn.Module] = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, cfg.input_dim))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    # ------------------------------------------------------------------ #
    # Forward and loss
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss(self, recon: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        recon_loss = nn.functional.mse_loss(recon, x, reduction="sum")
        if self.config.kl_weight > 0.0:
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + self.config.kl_weight * kl
        return recon_loss


def train_autoencoder(
    model: AutoencoderGen117,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop that works for both classical and quantum decoders."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(batch)
            loss = model.loss(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor
