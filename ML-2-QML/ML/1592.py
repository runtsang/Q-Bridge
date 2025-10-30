"""PyTorch implementation of an extended autoencoder with optional
batch‑norm, residual connections, and Gaussian noise injection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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
class AutoencoderConfig:
    """Configuration for :class:`MultiModalAutoencoder`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    use_batchnorm: bool = False
    use_residual: bool = False
    noise_std: float = 0.0


class MultiModalAutoencoder(nn.Module):
    """A deep MLP autoencoder with optional structural variants."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = self._make_mlp(
            cfg.input_dim, cfg.hidden_dims, cfg.latent_dim,
            cfg.use_batchnorm, cfg.use_residual, cfg.noise_std)
        self.decoder = self._make_mlp(
            cfg.latent_dim, cfg.hidden_dims[::-1], cfg.input_dim,
            cfg.use_batchnorm, cfg.use_residual, cfg.noise_std)

    def _make_mlp(
        self,
        in_dim: int,
        hidden_dims: Tuple[int,...],
        out_dim: int,
        batchnorm: bool,
        residual: bool,
        noise: float,
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if residual:
                layers.append(nn.Dropout(0.0))  # placeholder for residual skip
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        if noise > 0.0:
            layers.append(nn.GaussianNoise(noise))  # custom GaussianNoise layer
        return nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


def MultiModalAutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    use_batchnorm: bool = False,
    use_residual: bool = False,
    noise_std: float = 0.0,
) -> MultiModalAutoencoder:
    """Convenience constructor mirroring the legacy helper."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_batchnorm=use_batchnorm,
        use_residual=use_residual,
        noise_std=noise_std,
    )
    return MultiModalAutoencoder(cfg)


def train_autoencoder(
    model: MultiModalAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Training loop that records a mean‑squared‑error loss history."""
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


__all__ = [
    "MultiModalAutoencoder",
    "AutoencoderConfig",
    "MultiModalAutoencoderFactory",
    "train_autoencoder",
]
