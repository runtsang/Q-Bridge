"""Hybrid classical autoencoder combining QCNN blocks and a dense latent bottleneck."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class HybridAutoencoderConfig:
    """Parameters for :class:`HybridAutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)   # first and second hidden layers
    dropout: float = 0.1
    qcnn_hidden: Tuple[int, int, int, int] = (16, 12, 8, 4)  # internal QCNN widths


# --------------------------------------------------------------------------- #
# QCNN‑style block
# --------------------------------------------------------------------------- #
class QCNNBlock(nn.Module):
    """Imitates a convolution‑pooling block with fixed internal widths."""
    def __init__(self, in_dim: int, out_dim: int, hidden: Tuple[int, int, int, int]) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, hidden[0]),
            nn.Tanh(),
            nn.Linear(hidden[0], hidden[0]),
            nn.Tanh(),
            nn.Linear(hidden[0], hidden[1]),
            nn.Tanh(),
            nn.Linear(hidden[1], hidden[2]),
            nn.Tanh(),
            nn.Linear(hidden[2], hidden[3]),
            nn.Tanh(),
            nn.Linear(hidden[3], out_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# --------------------------------------------------------------------------- #
# Hybrid autoencoder
# --------------------------------------------------------------------------- #
class HybridAutoencoderNet(nn.Module):
    """A dense autoencoder whose encoder/decoder are QCNN‑style blocks."""
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.encoder_qcnn = QCNNBlock(
            in_dim=config.input_dim,
            out_dim=config.hidden_dims[0],
            hidden=config.qcnn_hidden,
        )
        self.encoder_linear = nn.Linear(config.hidden_dims[0], config.latent_dim)
        self.decoder_linear = nn.Linear(config.latent_dim, config.hidden_dims[0])
        self.decoder_qcnn = QCNNBlock(
            in_dim=config.hidden_dims[0],
            out_dim=config.input_dim,
            hidden=config.qcnn_hidden,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_qcnn(x)
        return self.encoder_linear(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_linear(z)
        return self.decoder_qcnn(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


# --------------------------------------------------------------------------- #
# Factory and training helper
# --------------------------------------------------------------------------- #
def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    qcnn_hidden: Tuple[int, int, int, int] = (16, 12, 8, 4),
) -> HybridAutoencoderNet:
    """Instantiate a :class:`HybridAutoencoderNet` with the supplied configuration."""
    cfg = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        qcnn_hidden=qcnn_hidden,
    )
    return HybridAutoencoderNet(cfg)


def train_hybrid_autoencoder(
    model: HybridAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Train the hybrid autoencoder and return the epoch‑wise MSE loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    loss_fn = nn.MSELoss()
    history: list[float] = []

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


__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderNet",
    "HybridAutoencoderConfig",
    "train_hybrid_autoencoder",
]
