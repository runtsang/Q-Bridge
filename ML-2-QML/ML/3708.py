from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable, List

# ------------------------------------------------------------------
# Classical autoencoder skeleton – inspired by ReferencePair[2]
# ------------------------------------------------------------------
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Lightweight MLP autoencoder."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = self._build_mlp(
            cfg.input_dim, cfg.hidden_dims, cfg.latent_dim, cfg.dropout
        )
        self.decoder = self._build_mlp(
            cfg.latent_dim, cfg.hidden_dims[::-1], cfg.input_dim, cfg.dropout
        )

    @staticmethod
    def _build_mlp(in_dim: int, hidden: Tuple[int,...], out_dim: int, dropout: float) -> nn.Sequential:
        layers = []
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

# ------------------------------------------------------------------
# HybridAutoSampler – classical wrapper
# ------------------------------------------------------------------
class HybridAutoSampler(nn.Module):
    """
    Classical wrapper that exposes an autoencoder interface.
    The class name mirrors the quantum implementation for seamless
    cross‑module substitution.  The quantum counterpart can be swapped
    in for the latent sampling step.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        cfg = AutoencoderConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.autoencoder = AutoencoderNet(cfg)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.autoencoder(x)

    def train_autoencoder(
        self,
        data: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: torch.device | None = None,
    ) -> List[float]:
        """Train loop returning loss history."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        dataset = TensorDataset(_as_tensor(data))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        history: List[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                recon = self(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history

# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

__all__ = ["HybridAutoSampler"]
