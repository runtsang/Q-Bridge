"""Hybrid classical autoencoder with optional self‑attention and a latent‑space estimator.

The implementation mirrors the original simple MLP autoencoder but extends it with
a configurable self‑attention block (classical) and a lightweight regression head
used as a classifier on the latent representation.  The class exposes the same
factory signature as the original `Autoencoder` helper so that existing pipelines
continue to work unchanged, while the additional components provide richer
representation learning.
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
    """Configuration for :class:`HybridAutoencoder`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    use_attention: bool = False
    attention_dim: int = 4  # only used if use_attention


class ClassicalSelfAttention:
    """Simple dot‑product self‑attention block that mirrors the quantum interface."""
    def __init__(self, embed_dim: int) -> None:
        self.embed_dim = embed_dim

    def run(self, rotation_params: torch.Tensor, entangle_params: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        query = inputs @ rotation_params.reshape(self.embed_dim, -1)
        key = inputs @ entangle_params.reshape(self.embed_dim, -1)
        scores = torch.softmax(query @ key.T / (self.embed_dim ** 0.5), dim=-1)
        return scores @ inputs


class EstimatorNN(nn.Module):
    """Small regression head used as a classifier on the latent space."""
    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridAutoencoderNet(nn.Module):
    """Core network combining encoder, decoder, optional attention, and estimator."""
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Encoder
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

        # Self‑attention (optional)
        self.attention = ClassicalSelfAttention(config.attention_dim) if config.use_attention else None

        # Decoder
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

        # Estimator head
        self.estimator = EstimatorNN(config.latent_dim)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        z = self.encoder(inputs)
        if self.attention is not None:
            # Dummy rotation/entangle params to satisfy interface
            rotation = torch.randn(self.config.attention_dim, self.config.attention_dim)
            entangle = torch.randn(self.config.attention_dim, self.config.attention_dim)
            z = self.attention.run(rotation, entangle, z)
        return z

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = self.encode(inputs)
        return self.decode(z)

    def classify(self, inputs: torch.Tensor) -> torch.Tensor:
        z = self.encode(inputs)
        return self.estimator(z)


def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    use_attention: bool = False,
    attention_dim: int = 4,
) -> HybridAutoencoderNet:
    """Factory mirroring the original Autoencoder helper."""
    config = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_attention=use_attention,
        attention_dim=attention_dim,
    )
    return HybridAutoencoderNet(config)


def train_hybrid(
    model: HybridAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """End‑to‑end reconstruction training loop with optional classification loss."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rec_loss_fn = nn.MSELoss()
    class_loss_fn = nn.CrossEntropyLoss()

    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)

            recon = model(batch)
            rec_loss = rec_loss_fn(recon, batch)

            # Classification loss uses dummy labels (e.g., zeros) for illustration
            logits = model.estimator(model.encode(batch))
            dummy_labels = torch.zeros(batch.size(0), dtype=torch.long, device=device)
            class_loss = class_loss_fn(logits, dummy_labels)

            loss = rec_loss + 0.1 * class_loss
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = ["HybridAutoencoder", "HybridAutoencoderNet", "HybridAutoencoderConfig", "train_hybrid"]
