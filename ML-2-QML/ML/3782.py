"""Hybrid auto‑encoder with fraud‑detection sub‑network.

This module builds on the classical autoencoder of the original seed and
introduces an additional fraud‑detection head inspired by the photonic
implementation.  The `HybridAutoFraudEncoder` class can be trained jointly
for reconstruction and fraud classification using a combined loss.
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
class AutoencoderConfig:
    """Configuration for the MLP auto‑encoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """Simple multilayer perceptron auto‑encoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
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
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)


# ------------------------------------------------------------------ #
# Fraud‑detection sub‑network (inspired by the photonic seed)
# ------------------------------------------------------------------ #
@dataclass
class FraudLayerParameters:
    """Parameters for a single fraud‑detection layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(
    params: FraudLayerParameters, *, clip: bool
) -> nn.Module:
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
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

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            return outputs * self.scale + self.shift

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential fraud‑detection net mirroring the photonic structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# ------------------------------------------------------------------ #
# Hybrid model
# ------------------------------------------------------------------ #
class HybridAutoFraudEncoder(nn.Module):
    """
    Combines a classical auto‑encoder and a fraud‑detection head.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the raw input.
    latent_dim : int, default 32
        Size of the latent representation.
    hidden_dims : Tuple[int, int], default (128, 64)
        Hidden layer sizes for the auto‑encoder.
    dropout : float, default 0.1
        Dropout probability in the auto‑encoder.
    fraud_params : Iterable[FraudLayerParameters]
        Parameters that define the fraud‑detection sub‑network.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        fraud_params: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        super().__init__()
        self.autoencoder = Autoencoder(
            input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims, dropout=dropout
        )
        if fraud_params is None:
            fraud_params = []
        self.fraud_head = build_fraud_detection_program(
            next(iter(fraud_params), FraudLayerParameters(0, 0, (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0))),
            fraud_params,
        )

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        reconstruction : torch.Tensor
            Reconstructed input.
        fraud_score : torch.Tensor
            Fraud probability (raw logits).
        """
        latent = self.autoencoder.encode(inputs)
        reconstruction = self.autoencoder.decode(latent)
        fraud_score = self.fraud_head(latent)
        return reconstruction, fraud_score


def train_hybrid(
    model: HybridAutoFraudEncoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    fraud_weight: float = 1.0,
) -> list[float]:
    """
    Joint training loop for reconstruction and fraud classification.

    The loss is a weighted sum of MSE (reconstruction) and BCE (fraud
    classification).  The fraud head operates on the latent vector.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    recon_loss_fn = nn.MSELoss()
    fraud_loss_fn = nn.BCEWithLogitsLoss()

    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, fraud_logits = model(batch)
            recon_loss = recon_loss_fn(reconstruction, batch)
            fraud_labels = torch.zeros_like(fraud_logits)  # placeholder; replace with real labels
            fraud_loss = fraud_loss_fn(fraud_logits, fraud_labels)
            loss = recon_loss + fraud_weight * fraud_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = [
    "Autoencoder",
    "AutoencoderNet",
    "AutoencoderConfig",
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "HybridAutoFraudEncoder",
    "train_hybrid",
]
