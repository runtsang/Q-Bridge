"""PyTorch implementation of an extended fully‑connected autoencoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)


@dataclass
class AutoencoderConfig:
    """Configuration values for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    use_batchnorm: bool = False


class AutoencoderNet(nn.Module):
    """A lightweight multilayer perceptron autoencoder with optional batch‑norm."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers: Sequence[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            if config.use_batchnorm:
                encoder_layers.append(nn.BatchNorm1d(hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: Sequence[nn.Module] = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            if config.use_batchnorm:
                decoder_layers.append(nn.BatchNorm1d(hidden))
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
    use_batchnorm: bool = False,
) -> AutoencoderNet:
    """Factory that mirrors the quantum helper returning a configured network."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_batchnorm=use_batchnorm,
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
    patience: int | None = None,
) -> list[float]:
    """
    Simple reconstruction training loop returning the loss history.

    Parameters
    ----------
    model : AutoencoderNet
        The autoencoder to train.
    data : torch.Tensor
        Training data, shape (N, input_dim).
    epochs : int, default 100
        Number of epochs.
    batch_size : int, default 64
        Batch size.
    lr : float, default 1e-3
        Learning rate.
    weight_decay : float, default 0.0
        L2 regularisation.
    device : torch.device | None, default None
        Device to train on; defaults to GPU if available.
    patience : int | None, default None
        If set, stop training early when validation loss stops improving.

    Returns
    -------
    history : list[float]
        Training loss per epoch.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []
    best_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

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
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    break
    return history


__all__ = ["Autoencoder", "AutoencoderConfig", "AutoencoderNet", "train_autoencoder"]
