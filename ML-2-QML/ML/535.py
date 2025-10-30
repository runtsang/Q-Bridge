"""Enhanced classical autoencoder with residual connections, dropout, and early stopping."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

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
class AutoencoderConfig:
    """Configuration for :class:`HybridAutoencoder`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    residual: bool = True
    early_stop_patience: int = 10
    reg_weight: float = 0.0

class ResidualBlock(nn.Module):
    """A simple residual block with optional dimension matching."""
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        if in_dim!= out_dim:
            self.skip = nn.Linear(in_dim, out_dim)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.bn(out)
        out = self.relu(out)
        return self.relu(out + self.skip(x))

class HybridAutoencoder(nn.Module):
    """Residual autoencoder with configurable encoder/decoder depth."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            if config.residual:
                encoder_layers.append(ResidualBlock(in_dim, hidden))
            else:
                encoder_layers.append(nn.Linear(in_dim, hidden))
                encoder_layers.append(nn.ReLU())
                if config.dropout > 0.0:
                    encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            if config.residual:
                decoder_layers.append(ResidualBlock(in_dim, hidden))
            else:
                decoder_layers.append(nn.Linear(in_dim, hidden))
                decoder_layers.append(nn.ReLU())
                if config.dropout > 0.0:
                    decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

def create_hybrid_autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    residual: bool = True,
    early_stop_patience: int = 10,
    reg_weight: float = 0.0,
) -> HybridAutoencoder:
    """Factory returning a configured :class:`HybridAutoencoder`."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        residual=residual,
        early_stop_patience=early_stop_patience,
        reg_weight=reg_weight,
    )
    return HybridAutoencoder(config)

def train_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    early_stop_patience: int = 10,
) -> List[float]:
    """Train the autoencoder with optional early stopping."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
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

        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping after {epoch+1} epochs.")
                break
    return history

__all__ = [
    "HybridAutoencoder",
    "create_hybrid_autoencoder",
    "train_autoencoder",
    "AutoencoderConfig",
]
