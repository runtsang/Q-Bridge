"""Hybrid‑style autoencoder with optional residual and KL‑regularisation.

The classical implementation extends the original fully‑connected design by
adding batch‑normalisation, residual skip‑connections and an optional KL‑loss
term for a VAE‑style objective.  It returns a single ``nn.Module`` that can
be used in any PyTorch training pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional, Any

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
class AutoencoderGen240Config:
    """Configuration for :class:`AutoencoderGen240`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    use_batchnorm: bool = True
    use_residual: bool = True
    vae: bool = False  # if True add KL‑divergence term


class AutoencoderGen240(nn.Module):
    """A fully‑connected autoencoder with optional residuals, batch‑norm and KL regularisation."""
    def __init__(self, cfg: AutoencoderGen240Config) -> None:
        super().__init__()
        self.cfg = cfg

        # Encoder construction
        enc_layers = []
        in_dim = cfg.input_dim
        for i, hidden in enumerate(cfg.hidden_dims):
            enc_layers.append(nn.Linear(in_dim, hidden))
            if cfg.use_batchnorm:
                enc_layers.append(nn.BatchNorm1d(hidden))
            enc_layers.append(nn.ReLU(inplace=True))
            if cfg.dropout > 0.0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim * (2 if cfg.vae else 1)))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder construction
        dec_layers = []
        in_dim = cfg.latent_dim * (2 if cfg.vae else 1)
        hidden_dims_rev = list(reversed(cfg.hidden_dims))
        for hidden in hidden_dims_rev:
            dec_layers.append(nn.Linear(in_dim, hidden))
            if cfg.use_batchnorm:
                dec_layers.append(nn.BatchNorm1d(hidden))
            dec_layers.append(nn.ReLU(inplace=True))
            if cfg.dropout > 0.0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # Optional residual connection
        if cfg.use_residual and cfg.input_dim == cfg.latent_dim:
            self.residual = nn.Identity()
        else:
            self.residual = None

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return latent representation (or mean&logvar for VAE)."""
        z = self.encoder(inputs)
        if self.cfg.vae:
            mean, logvar = torch.chunk(z, 2, dim=-1)
            return mean, logvar
        return z

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent code."""
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Standard autoencoder forward pass."""
        z = self.encode(inputs)
        if self.cfg.vae:
            mean, logvar = z
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + eps * std
        recon = self.decode(z)
        if self.residual is not None:
            recon = recon + self.residual(inputs)
        return recon

    def reconstruction_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """MSE reconstruction loss."""
        return nn.functional.mse_loss(recon, target, reduction="mean")

    def kl_divergence(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL divergence for VAE."""
        return -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())


def train_autoencoder_gen240(
    model: AutoencoderGen240,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    beta: float = 1.0,  # KL weight for VAE
) -> List[float]:
    """Train the autoencoder and return loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            recon = model(batch)
            loss = loss_fn(recon, batch)

            if model.cfg.vae:
                mean, logvar = model.encode(batch)
                loss += beta * model.kl_divergence(mean, logvar)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)

        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = [
    "AutoencoderGen240",
    "AutoencoderGen240Config",
    "train_autoencoder_gen240",
]
