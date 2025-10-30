"""Hybrid classical autoencoder with quantum‑inspired sampler.

This module extends a standard fully‑connected autoencoder by inserting a
parameterised SamplerQNN (see reference pair 2) between encoder and decoder.
The sampler is a small neural network that mimics the behaviour of a quantum
sampler and serves as a regulariser that encourages low‑dimensional latent
representations.  The architecture is fully compatible with the original
Autoencoder API and can be trained with the provided ``train_autoencoder``
routine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    activation: nn.Module = nn.ReLU()
    weight_init: str = "xavier_uniform"

class SamplerQNN(nn.Module):
    """Quantum‑inspired sampler that maps 2‑dimensional inputs to 2‑dimensional
    logits using a shallow feed‑forward network.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.softmax(self.net(inputs), dim=-1)

class HybridAutoencoder(nn.Module):
    """Classical autoencoder that injects a SamplerQNN between encoder and decoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        # Encoder
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(config.activation)
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Sampler
        self.sampler = SamplerQNN()

        # Decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(config.activation)
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if config.weight_init == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                elif config.weight_init == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        # Map latent to 2‑dim space via sampler before decoding
        # The sampler expects 2‑dim inputs; we project latents accordingly.
        projected = latents[:, :2] if latents.shape[-1] >= 2 else latents
        sampled = self.sampler(projected)
        # Concatenate back to original latent dimension
        if latents.shape[-1] > 2:
            sampled = torch.cat([sampled, latents[:, 2:]], dim=-1)
        return self.decoder(sampled)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

def HybridAutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    activation: nn.Module = nn.ReLU(),
    weight_init: str = "xavier_uniform",
) -> HybridAutoencoder:
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
        weight_init=weight_init,
    )
    return HybridAutoencoder(config)

def train_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    patience: int | None = None,
) -> list[float]:
    """Training loop with optional early stopping."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []
    best_loss = float("inf")
    counter = 0

    for epoch in range(epochs):
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

        if patience is not None:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

    return history

__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderFactory",
    "train_autoencoder",
    "AutoencoderConfig",
]
