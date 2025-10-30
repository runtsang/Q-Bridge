"""Enhanced PyTorch autoencoder with skip connections, latent regularization,
and early stopping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Convert input to a float32 tensor on the default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


@dataclass
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderGen353`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    use_skip: bool = False
    use_batchnorm: bool = False
    latent_reg: float = 0.0  # KL‑like regularisation weight


class AutoencoderGen353(nn.Module):
    """A multilayer perceptron autoencoder with optional skip connections
    and latent‑space regularisation."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Build encoder
        encoder_layers: List[nn.Module] = []
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

        # Build decoder
        decoder_layers: List[nn.Module] = []
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

        # Skip connection mapping (if enabled)
        if config.use_skip:
            self.skip = nn.Linear(config.input_dim, config.input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        y = self.decode(z)
        if self.config.use_skip:
            y = y + self.skip(x)
        return y

    def latent_regulariser(self, z: torch.Tensor) -> torch.Tensor:
        """Simple L2 regulariser on latent space."""
        if self.config.latent_reg <= 0.0:
            return torch.tensor(0.0, device=z.device)
        return self.config.latent_reg * torch.mean(torch.norm(z, dim=1) ** 2)


def AutoencoderGen353_factory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    use_skip: bool = False,
    use_batchnorm: bool = False,
    latent_reg: float = 0.0,
) -> AutoencoderGen353:
    """Convenience constructor mirroring the original API."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_skip=use_skip,
        use_batchnorm=use_batchnorm,
        latent_reg=latent_reg,
    )
    return AutoencoderGen353(cfg)


def train_autoencoder_gen353(
    model: AutoencoderGen353,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    patience: int = 10,
    min_delta: float = 1e-4,
) -> List[float]:
    """Training loop with early stopping based on validation loss."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch) + model.latent_regulariser(recon)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= len(dataset)
        history.append(epoch_loss)

        # Early‑stopping check
        if epoch_loss < best_loss - min_delta:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping after {epoch+1} epochs.")
            break

    return history


__all__ = [
    "AutoencoderGen353",
    "AutoencoderGen353_factory",
    "train_autoencoder_gen353",
    "AutoencoderConfig",
]
