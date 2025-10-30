"""Enhanced classical autoencoder with beta‑VAE style regularization and early stopping."""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, List, Optional

def _as_tensor(data: torch.Tensor | List[float]) -> torch.Tensor:
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
    dropout: float = 0.1
    beta: float = 1.0

class AutoencoderHybrid(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config
        # Encoder
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(config.dropout)])
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, 2 * config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        # Decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.extend([nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(config.dropout)])
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z)

    def loss(self, recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.config.beta * kl

def train_autoencoder(
    model: AutoencoderHybrid,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    early_stopping_patience: Optional[int] = None,
) -> List[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: List[float] = []
    best_loss = float("inf")
    patience = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            mu, logvar = model.encode(batch)
            loss = model.loss(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
        if early_stopping_patience is not None:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience = 0
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    break
    return history

__all__ = ["AutoencoderHybrid", "AutoencoderConfig", "train_autoencoder"]
