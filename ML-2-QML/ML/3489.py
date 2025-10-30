"""Hybrid classical autoencoder with VAE regularisation and fraud‑style layer blocks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Configuration and helper data structures
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Encapsulates network hyper‑parameters."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    kl_weight: float = 1e-3  # weight of KL divergence term


@dataclass
class FraudLayerParameters:
    """Parameters used by an optional custom linear block."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Utility used by the fraud‑style block."""
    return max(-bound, min(bound, value))


def fraud_style_block(params: FraudLayerParameters, *, clip: bool = True) -> nn.Module:
    """
    Builds a linear layer with custom bias and scaling, mimicking the photonic
    fraud detection block but suitable for a dense autoencoder.
    """
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]],
                          dtype=torch.float32)
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)

    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)

    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class FraudBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            x = self.activation(self.linear(inputs))
            return x * self.scale + self.shift

    return FraudBlock()


# --------------------------------------------------------------------------- #
# Core autoencoder implementation
# --------------------------------------------------------------------------- #
class AutoencoderNet(nn.Module):
    """
    VAE‑style autoencoder with optional fraud‑style blocks.
    The encoder and decoder are built as sequential containers, each layer
    optionally replaced by a fraud_style_block if a callable is supplied.
    """
    def __init__(
        self,
        config: AutoencoderConfig,
        block_factory: Optional[Callable[[int, int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.config = config

        def _make_layer(in_dim: int, out_dim: int) -> nn.Module:
            if block_factory is None:
                layer = nn.Linear(in_dim, out_dim)
            else:
                layer = block_factory(in_dim, out_dim)
            layers = [layer, nn.ReLU()]
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            return nn.Sequential(*layers)

        # Encoder
        encoder_layers = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            encoder_layers.append(_make_layer(in_dim, h))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            decoder_layers.append(_make_layer(in_dim, h))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes VAE loss: reconstruction + weighted KL divergence.
        """
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.config.kl_weight * kl


# --------------------------------------------------------------------------- #
# Training helper
# --------------------------------------------------------------------------- #
def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
) -> list[float]:
    """
    Trains the VAE‑style autoencoder. The latent distribution is modelled
    with a simple Gaussian prior; mean and log‑variance are learned in the
    encoder head.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            # Encoder outputs mean and log‑variance
            z_mean = model.encode(batch)
            # For simplicity we use a linear layer to predict logvar
            z_logvar = nn.Linear(z_mean.shape[1], z_mean.shape[1]).to(device)(z_mean)

            # Reparameterise
            eps = torch.randn_like(z_mean)
            z = z_mean + torch.exp(0.5 * z_logvar) * eps

            recon = model.decode(z)
            loss = model.loss_function(recon, batch, z_mean, z_logvar)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = [
    "AutoencoderConfig",
    "FraudLayerParameters",
    "fraud_style_block",
    "AutoencoderNet",
    "train_autoencoder",
]
