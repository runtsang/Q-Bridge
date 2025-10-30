"""AutoEncoderHybrid – classical implementation with optional quantum layer.

The class mirrors the original Autoencoder design but accepts a callable
``quantum_layer`` that transforms the latent representation.  This allows
plug‑in of a quantum module (e.g. a TorchQuantum or Qiskit sampler) while
keeping the training loop fully classical.

Typical usage::

    config = AutoencoderConfig(input_dim=784, latent_dim=32, hidden_dims=(128,64))
    model = AutoEncoderHybrid(config, quantum_layer=quantum_layer)
    history = train_autoencoder(model, data, epochs=200)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple

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
    """Configuration values for :class:`AutoEncoderHybrid`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoEncoderHybrid(nn.Module):
    """Classical auto‑encoder with an optional quantum transformation layer.

    Parameters
    ----------
    config: AutoencoderConfig
        Hyper‑parameters for the encoder/decoder.
    quantum_layer: Callable[[torch.Tensor], torch.Tensor] | None
        A function that receives the latent tensor and returns a transformed
        tensor of the same shape.  If ``None`` the latent vector is passed
        through unchanged.
    """

    def __init__(
        self,
        config: AutoencoderConfig,
        quantum_layer: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quantum_layer = quantum_layer

        # Encoder
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers: List[nn.Module] = []
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
        """Map inputs to latent space."""
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Map latent vector back to input space."""
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.encode(inputs)
        if self.quantum_layer is not None:
            latent = self.quantum_layer(latent)
        return self.decode(latent)


def AutoEncoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    quantum_layer: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> AutoEncoderHybrid:
    """Convenience factory mirroring the original API."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoEncoderHybrid(config, quantum_layer=quantum_layer)


def train_autoencoder(
    model: AutoEncoderHybrid,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Simple reconstruction training loop returning the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

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


__all__ = ["AutoEncoder", "AutoEncoderHybrid", "AutoencoderConfig", "train_autoencoder"]
