from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

@dataclass
class AutoencoderConfig:
    """Configuration for the hybrid classical autoencoder."""
    input_dim: int
    latent_dim: int = 32
    dropout: float = 0.1

class HybridAutoencoder(nn.Module):
    """A hybrid classical autoencoder that blends a QCNN‑inspired feature extractor
    with a fully‑connected encoder/decoder.  The encoder uses a sequence of
    linear + activation layers that mimic the QCNN convolution and pooling
    stages; the decoder mirrors the architecture in reverse."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Encoder – QCNN inspired
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(), nn.Dropout(config.dropout),
            nn.Linear(16, 12), nn.Tanh(), nn.Dropout(config.dropout),
            nn.Linear(12, 8), nn.Tanh(), nn.Dropout(config.dropout),
            nn.Linear(8, config.latent_dim)
        )

        # Decoder – reverse of encoder
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, 8), nn.Tanh(),
            nn.Linear(8, 12), nn.Tanh(),
            nn.Linear(12, 16), nn.Tanh(),
            nn.Linear(16, config.input_dim)
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    dropout: float = 0.1,
) -> HybridAutoencoder:
    """Factory that returns a :class:`HybridAutoencoder` configured with
    the supplied parameters."""
    config = AutoencoderConfig(input_dim=input_dim, latent_dim=latent_dim, dropout=dropout)
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
) -> list[float]:
    """Training loop that optimises reconstruction error."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

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

__all__ = ["Autoencoder", "AutoencoderConfig", "HybridAutoencoder", "train_autoencoder"]
