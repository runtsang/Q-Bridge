"""Hybrid autoencoder: classical encoder + optional quantum decoder."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Tuple, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

@dataclass
class HybridAutoencoderConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1

class HybridAutoencoderNet(nn.Module):
    """Hybrid autoencoder with classical encoder and optional quantum decoder."""
    def __init__(
        self,
        config: HybridAutoencoderConfig,
        quantum_decoder: Optional[Callable[[Iterable[float]], Iterable[float]]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quantum_decoder = quantum_decoder

        # Build classical encoder
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Classical decoder mirrors the encoder if quantum decoder not supplied
        if quantum_decoder is None:
            decoder_layers: List[nn.Module] = []
            in_dim = config.latent_dim
            for h in reversed(config.hidden_dims):
                decoder_layers.append(nn.Linear(in_dim, h))
                decoder_layers.append(nn.ReLU())
                if config.dropout > 0.0:
                    decoder_layers.append(nn.Dropout(config.dropout))
                in_dim = h
            decoder_layers.append(nn.Linear(in_dim, config.input_dim))
            self.decoder = nn.Sequential(*decoder_layers)
        else:
            self.decoder = None  # quantum decoder will be used

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self.quantum_decoder is None:
            return self.decoder(latent)
        # Convert latent tensor to numpy, run quantum decoder, then back to torch
        latent_np = latent.detach().cpu().numpy()
        recon_np = self.quantum_decoder(latent_np)
        return torch.as_tensor(recon_np, dtype=torch.float32, device=latent.device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.encode(inputs)
        return self.decode(latent)

def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    quantum_decoder: Optional[Callable[[Iterable[float]], Iterable[float]]] = None,
) -> HybridAutoencoderNet:
    """Utility constructor mirroring the quantum helper."""
    cfg = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return HybridAutoencoderNet(cfg, quantum_decoder=quantum_decoder)

def train_hybrid_autoencoder(
    model: HybridAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Training loop for the hybrid autoencoder."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

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

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderNet",
    "HybridAutoencoderConfig",
    "train_hybrid_autoencoder",
]
