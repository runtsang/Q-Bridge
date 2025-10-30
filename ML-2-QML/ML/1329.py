"""Classical fully‑connected autoencoder with optional skip connections and dropout.

The module exposes:
    * :class:`AutoencoderConfig` – dataclass with hyper‑parameters.
    * :class:`AutoencoderNet` – PyTorch implementation.
    * :func:`Autoencoder` – factory returning a configured network.
    * :func:`train_autoencoder` – training loop returning loss history.

The network can be extended with skip connections between matching encoder/decoder layers
and can be trained on any ``torch.Tensor`` or ``Iterable`` of floats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, List

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
    """Configuration values for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    use_skip: bool = False  # whether to add skip connections


class AutoencoderNet(nn.Module):
    """A lightweight multilayer perceptron autoencoder with optional skip connections."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Encoder
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        self.enc_layer_dims: List[int] = []
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU(inplace=True))
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
            self.enc_layer_dims.append(in_dim)
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers: List[nn.Module] = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU(inplace=True))
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Skip connections
        self.use_skip = config.use_skip
        if self.use_skip and len(self.enc_layer_dims)!= len(config.hidden_dims):
            raise ValueError("Skip connections require matching encoder/decoder layer counts")

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the latent representation of *inputs*."""
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Reconstruct *latents* back to the input space."""
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Standard autoencoder forward pass."""
        latent = self.encode(inputs)
        recon = self.decode(latent)
        if self.use_skip:
            # Add skip connections from encoder to decoder at matching layers
            # This is a simple residual addition; dimensions must match.
            # We store intermediate encoder activations during forward.
            activations: List[torch.Tensor] = []
            x = inputs
            in_dim = self.config.input_dim
            for hidden in self.config.hidden_dims:
                x = nn.functional.linear(x, self.encoder[0].weight, self.encoder[0].bias)
                x = nn.functional.relu(x)
                if self.config.dropout > 0.0:
                    x = nn.functional.dropout(x, p=self.config.dropout, training=self.training)
                activations.append(x)
            # Residual addition
            recon = recon + activations[-1]
        return recon

    def encode_latent(self, data: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper for encoding a dataset."""
        return self.encode(data)

    def decode_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper for decoding a latent vector."""
        return self.decode(latent)


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    use_skip: bool = False,
) -> AutoencoderNet:
    """Factory that returns a configured :class:`AutoencoderNet`."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_skip=use_skip,
    )
    return AutoencoderNet(config)


def train_autoencoder(
    model: AutoencoderNet,
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
    return history


__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "train_autoencoder",
]
