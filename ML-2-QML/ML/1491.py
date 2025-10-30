"""Enhanced autoencoder with early stopping, batch normalization and serialization."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


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
class AutoencoderGen049Config:
    """Configuration for :class:`AutoencoderGen049`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    batch_norm: bool = True
    early_stopping_patience: int = 10
    lr_schedule: Optional[str] = None  # e.g. 'cosine','step'


class AutoencoderGen049(nn.Module):
    """A fully‑connected autoencoder with optional batch‑norm and dropout."""

    def __init__(self, config: AutoencoderGen049Config) -> None:
        super().__init__()
        self.config = config
        self.encoder = self._build_mlp(
            in_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            out_dim=config.latent_dim,
            prefix="enc"
        )
        self.decoder = self._build_mlp(
            in_dim=config.latent_dim,
            hidden_dims=list(reversed(config.hidden_dims)),
            out_dim=config.input_dim,
            prefix="dec"
        )

    def _build_mlp(self, in_dim: int, hidden_dims: Tuple[int,...],
                   out_dim: int, prefix: str) -> nn.Sequential:
        layers: List[nn.Module] = []
        cur_dim = in_dim
        for i, hidden in enumerate(hidden_dims):
            layers.append(nn.Linear(cur_dim, hidden, bias=not self.config.batch_norm))
            if self.config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU(inplace=True))
            if self.config.dropout > 0.0:
                layers.append(nn.Dropout(self.config.dropout))
            cur_dim = hidden
        layers.append(nn.Linear(cur_dim, out_dim, bias=not self.config.batch_norm))
        return nn.Sequential(*layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the latent representation."""
        return self.encode(inputs)

    def evaluate_reconstruction_error(self, data: torch.Tensor) -> float:
        """Compute MSE over the entire dataset."""
        self.eval()
        with torch.no_grad():
            recon = self.forward(data)
            return F.mse_loss(recon, data, reduction="mean").item()

    def save(self, path: str) -> None:
        """Persist model weights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, config: AutoencoderGen049Config) -> "AutoencoderGen049":
        """Load model weights."""
        model = cls(config)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model


def train_autoencoder(
    model: AutoencoderGen049,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    verbose: bool = True,
) -> Tuple[List[float], AutoencoderGen049]:
    """Hybrid training loop with early stopping and optional learning‑rate schedule."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Optional schedule
    if model.config.lr_schedule:
        if model.config.lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif model.config.lr_schedule == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        else:
            scheduler = None
    else:
        scheduler = None

    loss_fn = nn.MSELoss()
    history: List[float] = []
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= len(dataset)
        history.append(epoch_loss)
        if verbose:
            print(f"Epoch {epoch+1:03d}/{epochs} – loss: {epoch_loss:.6f}")

        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= model.config.early_stopping_patience:
                if verbose:
                    print(f"Early stopping after {epoch+1} epochs.")
                break

        if scheduler:
            scheduler.step()

    # Restore best model
    model.load_state_dict(best_state)
    return history, model


__all__ = ["AutoencoderGen049", "AutoencoderGen049Config", "train_autoencoder"]
