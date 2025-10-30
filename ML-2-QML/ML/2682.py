"""Hybrid classical autoencoder with integrated self‑attention."""

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


class ClassicalSelfAttention:
    """Simple self‑attention block matching the Qiskit interface."""

    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        # rotation_params: (embed_dim, embed_dim)
        # entangle_params: (embed_dim, embed_dim)
        query = inputs @ rotation_params
        key = inputs @ entangle_params
        scores = torch.softmax(query @ key.T / (self.embed_dim ** 0.5), dim=-1)
        return scores @ inputs


@dataclass
class Gen250AutoencoderConfig:
    """Configuration for :class:`Gen250AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    attention_dim: int = 4  # used by SelfAttention


class Gen250AutoencoderNet(nn.Module):
    """Autoencoder that inserts a self‑attention block before encoding and after decoding."""

    def __init__(self, config: Gen250AutoencoderConfig) -> None:
        super().__init__()
        self.attention = ClassicalSelfAttention(config.attention_dim)

        # Build encoder
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder
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
        # Attention acts on raw inputs
        attn_out = self.attention.run(
            rotation_params=torch.eye(self.attention.embed_dim),
            entangle_params=torch.eye(self.attention.embed_dim),
            inputs=inputs,
        )
        return self.encoder(attn_out)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        recon = self.decoder(latents)
        # Apply attention again to the reconstruction
        return self.attention.run(
            rotation_params=torch.eye(self.attention.embed_dim),
            entangle_params=torch.eye(self.attention.embed_dim),
            inputs=recon,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def Gen250Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    attention_dim: int = 4,
) -> Gen250AutoencoderNet:
    """Factory mirroring the quantum helper."""
    config = Gen250AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        attention_dim=attention_dim,
    )
    return Gen250AutoencoderNet(config)


def train_gen250_autoencoder(
    model: Gen250AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop that returns the loss history."""
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
    "Gen250Autoencoder",
    "Gen250AutoencoderConfig",
    "Gen250AutoencoderNet",
    "train_gen250_autoencoder",
]
