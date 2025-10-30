"""Hybrid‑enhanced autoencoder with configurable latent schedules and dropout annealing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence, Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

def _as_tensor(data: Iterable[float] | torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert *data* to a 32‑bit float tensor on the current device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=dtype)
    if tensor.dtype!= dtype:
        tensor = tensor.to(dtype=dtype)
    return tensor


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderNet`."""

    input_dim: int
    latent_dims: Sequence[int] = (32,)  # support multiple latent stages
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    dropout_schedule: Optional[Callable[[int], float]] = None  # epoch → dropout

    def __post_init__(self) -> None:
        if self.dropout_schedule is None:
            # Default: linear decay from *dropout* to 0 over 100 epochs
            self.dropout_schedule = lambda epoch: max(0.0, self.dropout * (1 - epoch / 100))


# --------------------------------------------------------------------------- #
# Network definition
# --------------------------------------------------------------------------- #

class AutoencoderNet(nn.Module):
    """Multi‑latent MLP autoencoder with adaptive dropout."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoders: nn.ModuleList = nn.ModuleList()
        self.decoders: nn.ModuleList = nn.ModuleList()

        # Build encoder chain
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            self.encoders.append(nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
            ))
            in_dim = hidden

        # Build each latent block
        for latent_dim in config.latent_dims:
            self.encoders.append(nn.Linear(in_dim, latent_dim))

        # Build decoder chain (reverse of encoder)
        hidden_rev = list(reversed(config.hidden_dims))
        for hidden in hidden_rev:
            self.decoders.append(nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
            ))
            in_dim = hidden

        self.decoders.append(nn.Linear(in_dim, config.input_dim))

    # --------------------------------------------------------------------- #
    # Forward hooks
    # --------------------------------------------------------------------- #

    def encode(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        """Return a list of latent embeddings for each configured latent_dim."""
        embeddings: list[torch.Tensor] = []
        out = x
        for layer in self.encoders:
            out = layer(out)
            if isinstance(layer, nn.Linear) and layer.out_features in self.config.latent_dims:
                embeddings.append(out)
        return embeddings

    def decode(self, latents: Sequence[torch.Tensor]) -> torch.Tensor:
        """Reconstruct from the deepest latent representation."""
        # Use the last embedding for reconstruction
        out = latents[-1]
        for layer in self.decoders:
            out = layer(out)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full reconstruction."""
        latents = self.encode(x)
        return self.decode(latents)


# --------------------------------------------------------------------------- #
# Factory & training utilities
# --------------------------------------------------------------------------- #

def Autoencoder(
    input_dim: int,
    *,
    latent_dims: Sequence[int] = (32,),
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Instantiate a configured autoencoder."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dims=latent_dims,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(cfg)


def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> list[float]:
    """Train *model* on *data* and return the loss history."""
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
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= len(dataset)
        history.append(epoch_loss)

        if verbose:
            print(f"[Epoch {epoch+1:3d}] loss={epoch_loss:.6f}")

    return history


__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "train_autoencoder",
]
