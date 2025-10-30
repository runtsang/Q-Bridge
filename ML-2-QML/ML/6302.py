"""Hybrid Autoencoder that combines QCNN feature extraction with a dense bottleneck.

The module defines a PyTorch model that first processes the input through a
QCNN-inspired feature extractor (stack of linear layers with Tanh activations)
and then feeds the representation into an encoder–decoder pair.
It is compatible with the original Autoencoder seed but adds
convolutional style layers that mirror the QCNNModel.
"""

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
    conv_hidden_dims: Tuple[int, int, int] = (16, 8, 4)
    dropout: float = 0.1


class QCNNFeatureExtractor(nn.Module):
    """QCNN-inspired feature extractor using linear layers."""
    def __init__(self, input_dim: int, conv_hidden_dims: Tuple[int, int, int]) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for out_dim in conv_hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Tanh())
            in_dim = out_dim
        self.feature_map = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_map(x)


class HybridAutoencoderNet(nn.Module):
    """Hybrid autoencoder that chains a QCNN feature extractor and a dense autoencoder."""
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.feature_extractor = QCNNFeatureExtractor(
            config.input_dim, config.conv_hidden_dims
        )

        encoder_layers = []
        in_dim = config.conv_hidden_dims[-1]
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
        decoder_layers.append(nn.Linear(in_dim, config.conv_hidden_dims[-1]))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode the input through feature extractor and encoder."""
        return self.encoder(self.feature_extractor(inputs))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode the latent vector back to the feature extractor space."""
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        encoded = self.encode(inputs)
        decoded = self.decode(encoded)
        # Reconstruct to original input dimension
        return torch.nn.functional.pad(decoded, (0, inputs.shape[-1] - decoded.shape[-1]))


def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    conv_hidden_dims: Tuple[int, int, int] = (16, 8, 4),
    dropout: float = 0.1,
) -> HybridAutoencoderNet:
    """Factory that returns a configured :class:`HybridAutoencoderNet`."""
    config = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        conv_hidden_dims=conv_hidden_dims,
        dropout=dropout,
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
    """Training loop for the hybrid autoencoder."""
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


__all__ = ["HybridAutoencoder", "HybridAutoencoderConfig", "HybridAutoencoderNet", "train_hybrid_autoencoder"]
