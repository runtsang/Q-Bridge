"""Hybrid classical model combining an autoencoder and a QCNN-like architecture.

The class `HybridQCNN` is a torch.nn.Module that first compresses the
input with a lightweight MLP autoencoder and then applies a
convolution‑style fully‑connected network mirroring the quantum
QCNN.  The design is inspired by the seed `QCNN.py` and
`Autoencoder.py`, but uses a shared configuration object to allow
co‑scaling of both parts.  The model can be trained with a standard
reconstruction loss or a supervised loss on the QCNN output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class HybridConfig:
    """Configuration for the hybrid model.

    Attributes
    ----------
    input_dim : int
        Dimensionality of the raw input.
    latent_dim : int
        Size of the autoencoder bottleneck.
    encoder_hidden : Tuple[int,...]
        Hidden layer sizes for the encoder.
    decoder_hidden : Tuple[int,...]
        Hidden layer sizes for the decoder (mirror of encoder).
    dropout : float
        Dropout probability for both encoder and decoder.
    qcnn_hidden : Tuple[int,...]
        Hidden layer sizes for the QCNN‑style network.
    """

    input_dim: int
    latent_dim: int = 32
    encoder_hidden: Tuple[int,...] = (128, 64)
    decoder_hidden: Tuple[int,...] = (64, 128)
    dropout: float = 0.1
    qcnn_hidden: Tuple[int,...] = (16, 12, 8, 4, 4)


class AutoencoderNet(nn.Module):
    """Lightweight fully‑connected autoencoder."""

    def __init__(self, cfg: HybridConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = cfg.input_dim
        for h in cfg.encoder_hidden:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = cfg.latent_dim
        for h in cfg.decoder_hidden:
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


class QCNNModel(nn.Module):
    """Convolution‑style fully‑connected network mirroring QCNN layers."""

    def __init__(self, cfg: HybridConfig) -> None:
        super().__init__()
        layers = []
        in_dim = cfg.input_dim
        for h in cfg.qcnn_hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.sigmoid(self.network(x))


class HybridQCNN(nn.Module):
    """
    Hybrid classical model that first compresses inputs via an
    autoencoder and then processes the latent representation with a
    QCNN‑style network.  The design is a direct classical analogue of
    the quantum QCNN seed, enabling comparative studies.
    """

    def __init__(self, cfg: HybridConfig) -> None:
        super().__init__()
        self.autoencoder = AutoencoderNet(cfg)
        self.qcnn = QCNNModel(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.autoencoder.encode(x)
        return self.qcnn(z)


def HybridQCNNFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    encoder_hidden: Tuple[int,...] = (128, 64),
    decoder_hidden: Tuple[int,...] = (64, 128),
    dropout: float = 0.1,
    qcnn_hidden: Tuple[int,...] = (16, 12, 8, 4, 4),
) -> HybridQCNN:
    """Convenience factory mirroring the seed design."""
    cfg = HybridConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        encoder_hidden=encoder_hidden,
        decoder_hidden=decoder_hidden,
        dropout=dropout,
        qcnn_hidden=qcnn_hidden,
    )
    return HybridQCNN(cfg)


def train_hybrid(
    model: HybridQCNN,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Train the hybrid model using a reconstruction loss."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(data)
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


__all__ = [
    "HybridQCNN",
    "HybridQCNNFactory",
    "HybridConfig",
    "AutoencoderNet",
    "QCNNModel",
    "train_hybrid",
]
