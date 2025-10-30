from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, Sequence

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    activation: str = "ReLU"
    dropout: float = 0.1
    batch_norm: bool = False
    residual: bool = False
    noise_std: float = 0.0
    weight_init: str = "xavier_uniform"

class Autoencoder(nn.Module):
    """Classical fully‑connected autoencoder with optional residuals, batch‑norm, denoising and early‑stopping."""

    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        act = getattr(nn, cfg.activation)()
        self.encoder = self._build_mlp(cfg.input_dim, cfg.hidden_dims, cfg.latent_dim, act, cfg)
        self.decoder = self._build_mlp(cfg.latent_dim, cfg.hidden_dims[::-1], cfg.input_dim, act, cfg)
        self._init_weights()

    def _build_mlp(self, in_dim: int, hidden_dims: Sequence[int], out_dim: int,
                   act: nn.Module, cfg: AutoencoderConfig) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if cfg.batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act)
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(cfg.dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        init_fn = getattr(nn.init, self.cfg.weight_init)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_fn(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.noise_std > 0.0:
            noise = torch.randn_like(x) * self.cfg.noise_std
            x = x + noise
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def train_autoencoder(
    model: Autoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    patience: int = 10,
    val_split: float = 0.1,
    verbose: bool = False,
) -> list[float]:
    """Train with early‑stopping on a random train/val split."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    patience_left = patience
    history: list[float] = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch, in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch, in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                val_loss += loss_fn(recon, batch).item() * batch.size(0)
        val_loss /= n_val

        history.append(epoch_loss)
        if verbose:
            print(f"Epoch {epoch+1:03d} | Train {epoch_loss:.4f} | Val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_left = patience
            torch.save(model.state_dict(), "best_autoencoder.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose:
                    print("Early stopping")
                break
    model.load_state_dict(torch.load("best_autoencoder.pt"))
    return history

__all__ = ["Autoencoder", "AutoencoderConfig", "train_autoencoder"]
