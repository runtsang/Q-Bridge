"""Extended PyTorch implementation of a variational autoencoder with optional KL regularisation."""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class AutoencoderGen251Config:
    """Configuration for the AutoencoderGen251 network."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    kl_weight: float = 0.0  # weight of KL term; >0 activates VAE behaviour


class AutoencoderGen251(nn.Module):
    """A flexible autoencoder that can act as a vanilla MLP autoencoder or a VAE."""

    def __init__(self, config: AutoencoderGen251Config) -> None:
        super().__init__()
        self.use_vae = config.kl_weight > 0.0
        self.latent_dim = config.latent_dim

        # Encoder
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        self.encoder = nn.Sequential(*encoder_layers)

        if self.use_vae:
            self.mean_layer = nn.Linear(in_dim, config.latent_dim)
            self.logvar_layer = nn.Linear(in_dim, config.latent_dim)
        else:
            self.latent_layer = nn.Linear(in_dim, config.latent_dim)

        # Decoder
        decoder_layers: List[nn.Module] = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.kl_weight = config.kl_weight

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        h = self.encoder(x)
        if self.use_vae:
            return self.mean_layer(h), self.logvar_layer(h)
        return self.latent_layer(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterisation trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """Forward pass. Returns reconstruction and, if VAE, mu and logvar."""
        if self.use_vae:
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            recon = self.decode(z)
            return recon, mu, logvar
        z = self.encode(x)
        recon = self.decode(z)
        return recon

    def sample(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        """Sample from the prior to generate new data."""
        device = device or torch.device("cpu")
        z = torch.randn(batch_size, self.latent_dim, device=device)
        return self.decode(z)


def AutoencoderGen251(
    input_dim: int,
    *, latent_dim: int = 32, hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1, kl_weight: float = 0.0
) -> AutoencoderGen251:
    """Factory that returns a configured AutoencoderGen251 instance."""
    config = AutoencoderGen251Config(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        kl_weight=kl_weight,
    )
    return AutoencoderGen251(config)


def train_autoencoder_gen251(
    model: AutoencoderGen251,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> List[float]:
    """Train the model and return a history of loss values."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data = torch.as_tensor(data, dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    history: List[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            if model.use_vae:
                recon, mu, logvar = model(batch)
                recon_loss = mse_loss(recon, batch)
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + model.kl_weight * kl / batch.size(0)
            else:
                recon = model(batch)
                loss = mse_loss(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = ["AutoencoderGen251", "AutoencoderGen251Config", "train_autoencoder_gen251"]
