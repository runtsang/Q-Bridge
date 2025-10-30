"""Hybrid classical autoencoder with residual blocks and layer‑norm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

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
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    use_residual: bool = True
    layer_norm: bool = True

class ResidualBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0, layer_norm: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.layer_norm(out)
        return out + x  # residual connection

class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = self._build_mlp(
            config.input_dim,
            config.hidden_dims,
            config.latent_dim,
            config.dropout,
            config.use_residual,
            config.layer_norm,
        )
        self.decoder = self._build_mlp(
            config.latent_dim,
            tuple(reversed(config.hidden_dims)),
            config.input_dim,
            config.dropout,
            config.use_residual,
            config.layer_norm,
        )

    def _build_mlp(
        self,
        in_dim: int,
        hidden_dims: Tuple[int,...],
        out_dim: int,
        dropout: float,
        use_residual: bool,
        layer_norm: bool,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev_dim = in_dim
        for hidden in hidden_dims:
            if use_residual:
                layers.append(ResidualBlock(prev_dim, hidden, dropout, layer_norm))
            else:
                layers.append(nn.Linear(prev_dim, hidden))
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
                if layer_norm:
                    layers.append(nn.LayerNorm(hidden))
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, out_dim))
        return nn.Sequential(*layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    use_residual: bool = True,
    layer_norm: bool = True,
) -> AutoencoderNet:
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_residual=use_residual,
        layer_norm=layer_norm,
    )
    return AutoencoderNet(config)

def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
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

__all__ = ["Autoencoder", "AutoencoderConfig", "AutoencoderNet", "train_autoencoder", "ResidualBlock"]
