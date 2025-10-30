"""PyTorch implementation of a residual variational autoencoder with early‑stopping."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable, List

@dataclass
class AutoencoderConfig:
    """Configuration values for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (256, 128)
    dropout: float = 0.1
    batchnorm: bool = True
    vae: bool = False  # if True, add KL divergence term

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, batchnorm: bool = True):
        super().__init__()
        layers = [nn.Linear(dim, dim)]
        if batchnorm:
            layers.append(nn.BatchNorm1d(dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class AutoencoderNet(nn.Module):
    """Residual VAE / standard autoencoder."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        encoder_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            if cfg.batchnorm:
                encoder_layers.append(nn.BatchNorm1d(h))
            encoder_layers.append(nn.ReLU(inplace=True))
            if cfg.dropout > 0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            encoder_layers.append(ResidualBlock(h, cfg.dropout, cfg.batchnorm))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim * (2 if cfg.vae else 1)))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            if cfg.batchnorm:
                decoder_layers.append(nn.BatchNorm1d(h))
            decoder_layers.append(nn.ReLU(inplace=True))
            if cfg.dropout > 0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            decoder_layers.append(ResidualBlock(h, cfg.dropout, cfg.batchnorm))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)
        if self.cfg.vae:
            mu, logvar = torch.chunk(out, 2, dim=-1)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std, mu, logvar
        return out

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x) if not self.cfg.vae else self.encode(x)[0]
        return self.decode(z)

def Autoencoder(input_dim: int, **kwargs) -> AutoencoderNet:
    cfg = AutoencoderConfig(input_dim=input_dim, **kwargs)
    return AutoencoderNet(cfg)

def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    patience: int = 20,
) -> List[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(TensorDataset(_as_tensor(data)), batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss(reduction="sum")

    best_loss = float("inf")
    epochs_no_improve = 0
    history: List[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            if model.cfg.vae:
                z, mu, logvar = model.encode(batch)
                recon = model.decode(z)
                recon_loss = mse(recon, batch)
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl
            else:
                recon = model(batch)
                loss = mse(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(loader.dataset)
        history.append(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    return history

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

__all__ = ["Autoencoder", "AutoencoderConfig", "AutoencoderNet", "train_autoencoder"]
