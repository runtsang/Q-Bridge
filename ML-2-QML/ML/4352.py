"""Hybrid classical autoencoder with optional transformer layers and regression utilities."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Tuple

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Convert input to a float32 tensor on the default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data with sinusoidal target."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset wrapping the synthetic superposition data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

@dataclass
class HybridAutoencoderConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    use_transformer: bool = False
    transformer_heads: int = 4
    transformer_ffn: int = 256
    transformer_layers: int = 2

class HybridAutoencoderNet(nn.Module):
    """Classical autoencoder that optionally uses transformer blocks."""
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        if config.use_transformer:
            # Encoder: linear stack → transformer encoder → latent
            encoder_layers = []
            in_dim = config.input_dim
            for hidden in config.hidden_dims:
                encoder_layers.append(nn.Linear(in_dim, hidden))
                encoder_layers.append(nn.ReLU())
                if config.dropout > 0.0:
                    encoder_layers.append(nn.Dropout(config.dropout))
                in_dim = hidden
            encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
            self.encoder = nn.Sequential(*encoder_layers)

            # Decoder: latent → transformer decoder → output
            decoder_layers = []
            in_dim = config.latent_dim
            for hidden in reversed(config.hidden_dims):
                decoder_layers.append(nn.Linear(in_dim, hidden))
                decoder_layers.append(nn.ReLU())
                if config.dropout > 0.0:
                    decoder_layers.append(nn.Dropout(config.dropout))
                in_dim = hidden
            decoder_layers.append(nn.Linear(in_dim, config.input_dim))
            self.decoder = nn.Sequential(*decoder_layers)
        else:
            # Pure MLP encoder/decoder
            encoder_layers = []
            in_dim = config.input_dim
            for hidden in config.hidden_dims:
                encoder_layers.append(nn.Linear(in_dim, hidden))
                encoder_layers.append(nn.ReLU())
                if config.dropout > 0.0:
                    encoder_layers.append(nn.Dropout(config.dropout))
                in_dim = hidden
            encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
            self.encoder = nn.Sequential(*encoder_layers)

            decoder_layers = []
            in_dim = config.latent_dim
            for hidden in reversed(config.hidden_dims):
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

def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    use_transformer: bool = False,
    transformer_heads: int = 4,
    transformer_ffn: int = 256,
    transformer_layers: int = 2,
) -> HybridAutoencoderNet:
    """Factory returning a configured hybrid autoencoder."""
    config = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_transformer=use_transformer,
        transformer_heads=transformer_heads,
        transformer_ffn=transformer_ffn,
        transformer_layers=transformer_layers,
    )
    return HybridAutoencoderNet(config)

def train_hybrid_autoencoder(
    model: HybridAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop for the hybrid autoencoder."""
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

__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderConfig",
    "HybridAutoencoderNet",
    "train_hybrid_autoencoder",
    "RegressionDataset",
    "generate_superposition_data",
]
