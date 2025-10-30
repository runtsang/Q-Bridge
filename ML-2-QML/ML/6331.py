"""Hybrid classical autoencoder with QCNN‑style feature extractor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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
class HybridAutoencoderConfig:
    """Configuration for :class:`HybridAutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    qcnn_hidden: Tuple[int, int, int] = (16, 12, 8)


class QCNNFeatureExtractor(nn.Module):
    """QCNN‑style feature extractor that mimics a shallow convolution + pooling stack."""
    def __init__(self, input_dim: int, qcnn_hidden: Tuple[int, int, int]) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, qcnn_hidden[0]), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(qcnn_hidden[0], qcnn_hidden[0]), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(qcnn_hidden[0], qcnn_hidden[1]), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(qcnn_hidden[1], qcnn_hidden[2]), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(qcnn_hidden[2], qcnn_hidden[2]), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(qcnn_hidden[2], qcnn_hidden[2]), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return x


class HybridAutoencoderNet(nn.Module):
    """Classical autoencoder that uses a QCNN feature extractor before encoding."""
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.feature_extractor = QCNNFeatureExtractor(config.input_dim, config.qcnn_hidden)

        encoder_layers = []
        in_dim = config.qcnn_hidden[2]
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
        """Apply QCNN feature extractor then encode."""
        x = self.feature_extractor(inputs)
        return self.encoder(x)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    qcnn_hidden: Tuple[int, int, int] = (16, 12, 8),
) -> HybridAutoencoderNet:
    """Factory that returns a configured :class:`HybridAutoencoderNet`."""
    config = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        qcnn_hidden=qcnn_hidden,
    )
    return HybridAutoencoderNet(config)


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
    """Training loop that returns a loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

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
    "HybridAutoencoder",
    "HybridAutoencoderConfig",
    "HybridAutoencoderNet",
    "train_hybrid_autoencoder",
]
