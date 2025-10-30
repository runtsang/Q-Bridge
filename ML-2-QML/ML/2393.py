from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Convert input to a float32 tensor on the default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)

@dataclass
class HybridAutoencoderConfig:
    """Configuration for the hybrid auto‑encoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    quantum_layer: Callable[[torch.Tensor], torch.Tensor] | None = None

class HybridAutoencoder(nn.Module):
    """Classical MLP encoder/decoder with a quantum transformation in latent space."""
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.encoder = self._make_mlp(
            in_dim=config.input_dim,
            out_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        )
        self.quantum_layer = config.quantum_layer or (lambda x: x)
        self.decoder = self._make_mlp(
            in_dim=config.latent_dim,
            out_dim=config.input_dim,
            hidden_dims=config.hidden_dims[::-1],
            dropout=config.dropout,
        )

    def _make_mlp(self, in_dim: int, out_dim: int, hidden_dims: Tuple[int,...], dropout: float) -> nn.Sequential:
        layers: list[nn.Module] = []
        current = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(current, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current = h
        layers.append(nn.Linear(current, out_dim))
        return nn.Sequential(*layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def quantum_transform(self, latents: torch.Tensor) -> torch.Tensor:
        return self.quantum_layer(latents)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z = self.encode(inputs)
        z_q = self.quantum_transform(z)
        return self.decode(z_q)

def HybridAutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    quantum_layer: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> HybridAutoencoder:
    """Create a configured hybrid auto‑encoder."""
    config = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        quantum_layer=quantum_layer,
    )
    return HybridAutoencoder(config)

def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Standard reconstruction training loop for the hybrid model."""
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
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = ["HybridAutoencoder", "HybridAutoencoderFactory", "train_hybrid_autoencoder"]
