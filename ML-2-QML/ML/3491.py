from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
@dataclass
class HybridAutoencoderConfig:
    """Parameters for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    # Quantum interface – a callable that maps a latent tensor to a quantum‑encoded tensor
    quantum_encoder: Callable[[torch.Tensor], torch.Tensor] | None = None


# --------------------------------------------------------------------------- #
#  Classical encoder / decoder
# --------------------------------------------------------------------------- #
class ClassicalEncoder(nn.Module):
    """MLP encoder producing a latent vector."""
    def __init__(self, cfg: HybridAutoencoderConfig) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if cfg.dropout:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassicalDecoder(nn.Module):
    """Decoder that reconstructs the input from a latent vector."""
    def __init__(self, cfg: HybridAutoencoderConfig) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if cfg.dropout:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# --------------------------------------------------------------------------- #
#  Hybrid autoencoder
# --------------------------------------------------------------------------- #
class HybridAutoencoderNet(nn.Module):
    """
    Combines a classical encoder/decoder with an optional quantum encoder.
    The quantum encoder receives the classical latent vector and returns a refined
    latent representation.  If no quantum_encoder is supplied, the classical
    latent is used directly.
    """
    def __init__(self, cfg: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = ClassicalEncoder(cfg)
        self.decoder = ClassicalDecoder(cfg)
        # The quantum encoder is expected to be a callable taking a torch.Tensor
        # and returning a torch.Tensor of the same shape.
        self.q_encoder = cfg.quantum_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if self.q_encoder is not None:
            z = self.q_encoder(z)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the (possibly quantum‑encoded) latent vector."""
        z = self.encoder(x)
        if self.q_encoder is not None:
            z = self.q_encoder(z)
        return z


# --------------------------------------------------------------------------- #
#  Training helper
# --------------------------------------------------------------------------- #
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
    """Simple reconstruction training loop."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
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


__all__ = [
    "HybridAutoencoderConfig",
    "HybridAutoencoderNet",
    "train_hybrid_autoencoder",
]
