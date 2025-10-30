import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable, List

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Convert input to a float32 tensor on the CPU."""
    if isinstance(data, torch.Tensor):
        return data.to(dtype=torch.float32)
    return torch.as_tensor(data, dtype=torch.float32)

@dataclass
class AutoencoderConfig:
    """Configuration for the classical VAE."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 64
    weight_decay: float = 0.0
    early_stop_patience: int = 10

class VAE(nn.Module):
    """Variational autoencoder with fully‑connected encoder/decoder."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Encoder layers
        encoder_layers = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        self.enc_mu = nn.Linear(in_dim, cfg.latent_dim)
        self.enc_logvar = nn.Linear(in_dim, cfg.latent_dim)
        self.encoder = nn.Sequential(*encoder_layers)
        # Decoder layers
        decoder_layers = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def VAE(cfg: AutoencoderConfig) -> VAE:
    """Factory that returns a configured VAE."""
    return VAE(cfg)

def train_vae(
    model: VAE,
    data: torch.Tensor,
    *,
    cfg: AutoencoderConfig | None = None,
    device: torch.device | None = None,
) -> List[float]:
    """Train the VAE with early‑stopping."""
    cfg = cfg or AutoencoderConfig(input_dim=data.shape[1])
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss(reduction="sum")
    history: List[float] = []

    best_loss = float("inf")
    patience = cfg.early_stop_patience
    counter = 0

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            recon_loss = mse(recon, batch)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    return history

__all__ = ["AutoencoderConfig", "VAE", "VAE", "train_vae"]
