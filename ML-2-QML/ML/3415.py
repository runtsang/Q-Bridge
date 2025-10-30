"""
HybridAutoencoder – classical component
========================================

This module implements the classical encoder/decoder pair that mirrors the original
`Autoencoder.py` seed but adds a few extensions:

* Support for optional bias terms and layer‑wise dropout.
* A small regression network (`EstimatorNN`) that can be attached to the latent
  representation for quick downstream experiments.
* A convenient training loop that returns a loss history.

The API is intentionally lightweight: only the classes and helper functions
needed by the hybrid wrapper are exported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """
    Convert a sequence or tensor to a float32 :class:`torch.Tensor` on the default device.
    """
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


@dataclass
class AutoencoderConfig:
    """Configuration for :class:`ClassicalAutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    bias: bool = True


class ClassicalAutoencoderNet(nn.Module):
    """Multi‑layer perceptron autoencoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = nn.Sequential()
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            self.encoder.add_module(f"enc_lin_{hidden}",
                                    nn.Linear(in_dim, hidden, bias=config.bias))
            self.encoder.add_module(f"enc_relu_{hidden}", nn.ReLU())
            if config.dropout > 0.0:
                self.encoder.add_module(f"enc_drop_{hidden}",
                                        nn.Dropout(config.dropout))
            in_dim = hidden
        self.encoder.add_module("enc_latent",
                                nn.Linear(in_dim, config.latent_dim, bias=config.bias))

        self.decoder = nn.Sequential()
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            self.decoder.add_module(f"dec_lin_{hidden}",
                                    nn.Linear(in_dim, hidden, bias=config.bias))
            self.decoder.add_module(f"dec_relu_{hidden}", nn.ReLU())
            if config.dropout > 0.0:
                self.decoder.add_module(f"dec_drop_{hidden}",
                                        nn.Dropout(config.dropout))
            in_dim = hidden
        self.decoder.add_module("dec_out",
                                nn.Linear(in_dim, config.input_dim, bias=config.bias))

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def ClassicalAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    bias: bool = True,
) -> ClassicalAutoencoderNet:
    """Return a fully‑configured classical autoencoder."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        bias=bias,
    )
    return ClassicalAutoencoderNet(cfg)


def train_classical_autoencoder(
    model: ClassicalAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """
    Simple reconstruction training loop that returns a list of epoch‑wise MSE losses.
    """
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


class EstimatorNN(nn.Module):
    """A lightweight regression head that can be attached to the latent space."""
    def __init__(self, input_dim: int = 2, hidden_dims: Tuple[int, int] = (8, 4)) -> None:
        super().__init__()
        layers = []
        dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(dim, h))
            layers.append(nn.Tanh())
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)


def ClassicalEstimatorQNN(input_dim: int = 2, hidden_dims: Tuple[int, int] = (8, 4)) -> EstimatorNN:
    """Convenience factory for the classical regression head."""
    return EstimatorNN(input_dim, hidden_dims)


__all__ = [
    "AutoencoderConfig",
    "ClassicalAutoencoderNet",
    "ClassicalAutoencoder",
    "train_classical_autoencoder",
    "EstimatorNN",
    "ClassicalEstimatorQNN",
]
