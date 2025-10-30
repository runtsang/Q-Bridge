"""Hybrid classical autoencoder with quantum‑kernel regularization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


@dataclass
class HybridAutoencoderConfig:
    """Configuration for :class:`HybridAutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    kernel_gamma: float = 1.0
    kernel_reg_weight: float = 0.0


class HybridAutoencoderNet(nn.Module):
    """Classical autoencoder equipped with a kernel‑based regularizer."""
    def __init__(self, cfg: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Encoder
        enc_layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(cfg.dropout)])
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(cfg.dropout)])
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

    def kernel_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """Compute a Gaussian RBF kernel matrix on the latent vectors."""
        diff = z.unsqueeze(1) - z.unsqueeze(0)  # shape (N,N,D)
        sq_norm = (diff**2).sum(-1)
        return torch.exp(-self.cfg.kernel_gamma * sq_norm)

    def kernel_regularizer(self, z: torch.Tensor) -> torch.Tensor:
        """Encourage diversity in latent space via kernel similarity."""
        k = self.kernel_matrix(z)
        # Penalize off‑diagonal similarity (i.e., high overlap)
        mask = torch.eye(k.shape[0], device=k.device).bool()
        return (k[~mask] ** 2).mean()


def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    kernel_gamma: float = 1.0,
    kernel_reg_weight: float = 0.0,
) -> HybridAutoencoderNet:
    cfg = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        kernel_gamma=kernel_gamma,
        kernel_reg_weight=kernel_reg_weight,
    )
    return HybridAutoencoderNet(cfg)


def train_hybrid_autoencoder(
    model: HybridAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Train with reconstruction + optional kernel regularization."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            recon = model(batch)
            rec_loss = loss_fn(recon, batch)
            if model.cfg.kernel_reg_weight > 0.0:
                z = model.encode(batch)
                kern_reg = model.kernel_regularizer(z)
                loss = rec_loss + model.cfg.kernel_reg_weight * kern_reg
            else:
                loss = rec_loss
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = ["HybridAutoencoder", "HybridAutoencoderConfig", "HybridAutoencoderNet", "train_hybrid_autoencoder"]
