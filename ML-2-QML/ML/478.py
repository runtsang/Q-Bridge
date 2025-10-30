"""Enhanced PyTorch autoencoder with configurable layers, batch normalization, early stopping, and
cross‑entropy loss support for classification tasks.

The module retains the original factory API but adds richer training utilities.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Iterable, Tuple, Callable

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
    """Configuration for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU
    batch_norm: bool = False
    use_mse: bool = True  # if False, use BCEWithLogits for binary data


class AutoencoderNet(nn.Module):
    """A fully‑connected autoencoder with optional batch‑norm and configurable activation."""

    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        def make_layer(in_dim: int, out_dim: int) -> Sequence[nn.Module]:
            layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
            if cfg.batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(cfg.activation())
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(cfg.dropout))
            return layers

        # Encoder
        encoder_layers: list[nn.Module] = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            encoder_layers.extend(make_layer(in_dim, h))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers: list[nn.Module] = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            decoder_layers.extend(make_layer(in_dim, h))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
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
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU,
    batch_norm: bool = False,
    use_mse: bool = True,
) -> AutoencoderNet:
    """Factory that mirrors the quantum helper returning a configured network."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
        batch_norm=batch_norm,
        use_mse=use_mse,
    )
    return AutoencoderNet(cfg)


def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    early_stop_patience: int | None = None,
    lr_scheduler: bool = False,
) -> list[float]:
    """Extended training loop supporting early stopping and optional learning‑rate decay."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)
    loss_fn = nn.MSELoss() if model.cfg.use_mse else nn.BCEWithLogitsLoss()
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

        if lr_scheduler:
            scheduler.step(epoch_loss)

        if early_stop_patience is not None:
            if epoch_loss < best_loss - 1e-4:  # significant improvement
                best_loss = epoch_loss
                counter = 0
            else:
                counter += 1
                if counter >= early_stop_patience:
                    break

    return history


__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "train_autoencoder",
]
