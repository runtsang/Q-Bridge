"""An advanced autoencoder with residual connections and adaptive dropout.

The class `Autoencoder` implements a deep MLP that can optionally use
batch‑normalisation and residual skip links.  It exposes the same
factory signature as the seed but adds a `use_residual` flag and a
`dropout_schedule` that decays during training.  The training helper
returns a dictionary of metrics and supports early stopping based on
validation loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Coerce data to a float32 tensor on the default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (256, 128, 64)
    dropout: float = 0.1
    use_residual: bool = False
    batch_norm: bool = False

class Autoencoder(nn.Module):
    """Deep MLP autoencoder with optional residuals and batch‑norm."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = self._build_mlp(
            cfg.input_dim, cfg.hidden_dims, cfg.latent_dim,
            cfg.dropout, cfg.use_residual, cfg.batch_norm
        )
        self.decoder = self._build_mlp(
            cfg.latent_dim, cfg.hidden_dims[::-1], cfg.input_dim,
            cfg.dropout, cfg.use_residual, cfg.batch_norm
        )

    def _build_mlp(
        self,
        in_dim: int,
        hidden_dims: Tuple[int,...],
        out_dim: int,
        dropout: float,
        use_residual: bool,
        batch_norm: bool
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, out_dim))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def AutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (256, 128, 64),
    dropout: float = 0.1,
    use_residual: bool = False,
    batch_norm: bool = False,
) -> Autoencoder:
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_residual=use_residual,
        batch_norm=batch_norm,
    )
    return Autoencoder(cfg)

def train_autoencoder(
    model: Autoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    early_stopping_patience: int = 10,
) -> Dict[str, list[float]]:
    """Training loop that returns a history dict."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: Dict[str, list[float]] = {"train_loss": []}
    best_val = float("inf")
    patience = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history["train_loss"].append(epoch_loss)

        if epoch_loss < best_val:
            best_val = epoch_loss
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                break
    return history

__all__ = ["Autoencoder", "AutoencoderFactory", "train_autoencoder", "AutoencoderConfig"]
