# autoencoder_hybrid_ml.py

"""
AutoencoderHybrid: Classical autoencoder with optional VAE capabilities.

Features:
- Configurable architecture (hidden layers, dropout, batchnorm).
- Supports deterministic autoencoder and variational autoencoder modes.
- Provides training loop with reconstruction + KL divergence.
- Supports sampling from prior and reconstruction of new inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Callable, Optional
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def _as_tensor(data: torch.Tensor | list | tuple) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

@dataclass
class AutoencoderHybridConfig:
    """Configuration values for :class:`AutoencoderHybrid`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    batchnorm: bool = False
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU
    vae: bool = False

class AutoencoderHybrid(nn.Module):
    """A flexible autoencoder that can act as a deterministic or variational model."""
    def __init__(self, config: AutoencoderHybridConfig):
        super().__init__()
        self.config = config
        self.vae = config.vae

        # Encoder
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            if config.batchnorm:
                encoder_layers.append(nn.BatchNorm1d(hidden))
            encoder_layers.append(config.activation())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        self.encoder = nn.Sequential(*encoder_layers)

        if self.vae:
            self.mu_layer = nn.Linear(in_dim, config.latent_dim)
            self.logvar_layer = nn.Linear(in_dim, config.latent_dim)
        else:
            self.latent_layer = nn.Linear(in_dim, config.latent_dim)

        # Decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            if config.batchnorm:
                decoder_layers.append(nn.BatchNorm1d(hidden))
            decoder_layers.append(config.activation())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        h = self.encoder(x)
        if self.vae:
            mu = self.mu_layer(h)
            logvar = self.logvar_layer(h)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, mu, logvar
        else:
            return self.latent_layer(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass."""
        if self.vae:
            z, mu, logvar = self.encode(x)
            return self.decode(z), mu, logvar
        else:
            z = self.encode(x)
            return self.decode(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from its latent representation."""
        if self.vae:
            z, _, _ = self.encode(x)
        else:
            z = self.encode(x)
        return self.decode(z)

    def sample_prior(self, num_samples: int) -> torch.Tensor:
        """Sample from the prior (standard normal) and decode."""
        z = torch.randn(num_samples, self.config.latent_dim, device=self.device)
        return self.decode(z)

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return next(self.parameters()).device

def train_autoencoder_hybrid(
    model: AutoencoderHybrid,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
) -> Tuple[list[float], list[float]]:
    """
    Train the AutoencoderHybrid model.

    Returns:
        Tuple of reconstruction loss history and KL divergence history (empty list if not VAE).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss = nn.MSELoss(reduction="sum")
    recon_hist: list[float] = []
    kl_hist: list[float] = []

    for _ in range(epochs):
        epoch_recon = 0.0
        epoch_kl = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            if model.vae:
                recon, mu, logvar = model(batch)
                recon_loss = mse_loss(recon, batch)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl_loss
                epoch_kl += kl_loss.item()
            else:
                recon = model(batch)
                recon_loss = mse_loss(recon, batch)
                loss = recon_loss
            loss.backward()
            optimizer.step()
            epoch_recon += recon_loss.item()
        epoch_recon /= len(dataset)
        recon_hist.append(epoch_recon)
        if model.vae:
            epoch_kl /= len(dataset)
            kl_hist.append(epoch_kl)
    return recon_hist, kl_hist

__all__ = [
    "AutoencoderHybrid",
    "AutoencoderHybridConfig",
    "train_autoencoder_hybrid",
]
