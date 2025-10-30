from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, List, Callable

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
    """Configuration values for :class:`AutoencoderExtended`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()

class AutoencoderExtended(nn.Module):
    """A fully‑connected autoencoder with optional L1 regularisation and early‑stopping."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(config.activation)
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(config.activation)
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the latent representation."""
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Return the reconstruction from latent codes."""
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

    def loss(self, recon: torch.Tensor, target: torch.Tensor, l1_reg: float = 0.0) -> torch.Tensor:
        """Compute MSE loss optionally with L1 regularisation."""
        mse = nn.functional.mse_loss(recon, target, reduction="mean")
        if l1_reg > 0.0:
            l1 = sum(p.abs().sum() for p in self.parameters())
            return mse + l1_reg * l1
        return mse

    def evaluate(self, data_loader: DataLoader) -> float:
        """Return average MSE loss over a data loader."""
        self.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in data_loader:
                batch = batch[0]
                recon = self(batch)
                loss = nn.functional.mse_loss(recon, batch, reduction="sum")
                total += loss.item()
                count += batch.size(0)
        return total / count

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
) -> AutoencoderExtended:
    """Factory mirroring the quantum helper, returning a configured network."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
    )
    return AutoencoderExtended(config)

def train_autoencoder(
    model: AutoencoderExtended,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    l1_reg: float = 0.0,
    val_data: Optional[torch.Tensor] = None,
    patience: int = 10,
    device: torch.device | None = None,
) -> List[float]:
    """Train with optional early‑stopping."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    history: List[float] = []
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = model.loss(recon, batch, l1_reg=l1_reg)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

        if val_data is not None:
            val_loader = DataLoader(
                TensorDataset(_as_tensor(val_data)), batch_size=batch_size
            )
            val_loss = model.evaluate(val_loader)
            if val_loss < best_val:
                best_val = val_loss
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break
    return history

__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderExtended",
    "train_autoencoder",
]
