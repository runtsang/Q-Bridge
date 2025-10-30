"""Hybrid autoencoder combining a classical MLP encoder‑decoder with QCNN‑inspired decoder layers.

The class `HybridAutoencoder` mirrors the original `Autoencoder` interface but offers an optional
QCNN‑style decoder that replaces the fully‑connected stack with a shallow convolution‑like
sequence of linear + tanh layers.  This design lets users experiment with a purely classical
model, a hybrid classical‑QCNN model, or a model that later accepts a quantum latent
representation produced by the counterpart in the QML module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Convert input data to a float32 tensor on the default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)


@dataclass
class HybridAutoencoderConfig:
    """Configuration values for :class:`HybridAutoencoder`."""

    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    use_qcnn_decoder: bool = False
    qcnn_layers: Tuple[int,...] = (16, 8, 4)


class HybridAutoencoder(nn.Module):
    """Hybrid autoencoder that optionally replaces the classical decoder with a QCNN‑style stack."""

    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        # Encoder
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.extend(
                [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(config.dropout)]
            )
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        if config.use_qcnn_decoder:
            decoder_layers = []
            in_dim = config.latent_dim
            for out_dim in config.qcnn_layers:
                decoder_layers.extend([nn.Linear(in_dim, out_dim), nn.Tanh()])
                in_dim = out_dim
            decoder_layers.append(nn.Linear(in_dim, config.input_dim))
            self.decoder = nn.Sequential(*decoder_layers)
        else:
            decoder_layers = []
            in_dim = config.latent_dim
            for hidden in reversed(config.hidden_dims):
                decoder_layers.extend(
                    [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(config.dropout)]
                )
                in_dim = hidden
            decoder_layers.append(nn.Linear(in_dim, config.input_dim))
            self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def HybridAutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    use_qcnn_decoder: bool = False,
    qcnn_layers: Tuple[int,...] = (16, 8, 4),
) -> HybridAutoencoder:
    """Return a :class:`HybridAutoencoder` configured with the provided parameters."""
    config = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_qcnn_decoder=use_qcnn_decoder,
        qcnn_layers=qcnn_layers,
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
    "HybridAutoencoderFactory",
    "HybridAutoencoderConfig",
    "train_hybrid_autoencoder",
]
