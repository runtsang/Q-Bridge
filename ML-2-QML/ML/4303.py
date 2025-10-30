"""Hybrid classical autoencoder combining convolutional feature extraction,
QCNN-inspired latent processing, and a dense decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Import the auxiliary modules from the seed codebase
from.Conv import Conv
from.QCNN import QCNN

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
    """Configuration for :class:`HybridAutoencoder`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    kernel_size: int = 2  # used by the ConvFilter


class HybridAutoencoder(nn.Module):
    """Hybrid autoencoder with a quantum-inspired latent layer and classical
    convolutional feature extractor."""
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Feature extractor: 2‑D convolution filter
        self.feature_extractor = Conv(kernel_size=config.kernel_size, threshold=0.5)

        # QCNN encoder (classical, quantum‑inspired)
        self.qcnn_encoder = QCNN()

        # Dense mapping from QCNN output to the latent space
        self.latent_mapper = nn.Sequential(
            nn.Linear(self.qcnn_encoder.head.out_features, config.latent_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()
        )

        # Dense decoder back to the input dimension
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity(),
            nn.Linear(config.hidden_dims[0], config.input_dim),
            nn.Sigmoid()
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode the input through convolution, QCNN, and dense layers."""
        # Run the 2‑D filter on each sample in the batch
        batch_size = inputs.shape[0]
        conv_features = []
        for i in range(batch_size):
            sample = inputs[i].view(self.config.kernel_size, self.config.kernel_size)
            conv_features.append(self.feature_extractor.run(sample))
        conv_tensor = torch.as_tensor(conv_features, dtype=torch.float32, device=inputs.device)

        # Pass through the QCNN encoder
        qcnn_out = self.qcnn_encoder(conv_tensor)

        # Map to latent space
        latent = self.latent_mapper(qcnn_out)
        return latent

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode the latent vector back to the original dimension."""
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    kernel_size: int = 2,
) -> HybridAutoencoder:
    """Factory returning a configured :class:`HybridAutoencoder`."""
    config = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        kernel_size=kernel_size,
    )
    return HybridAutoencoder(config)


def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Simple reconstruction training loop returning the loss history."""
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
    "HybridAutoencoder",
    "train_hybrid_autoencoder",
]
