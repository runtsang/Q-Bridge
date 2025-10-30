from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

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
    use_quantum_kernel: bool = False

class HybridAutoencoder(nn.Module):
    """Classical autoencoder with optional quantum‑kernel utilities."""
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Encoder
        enc_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                enc_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers: List[nn.Module] = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                dec_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return an RBF kernel matrix or raise if quantum kernel requested."""
        if self.config.use_quantum_kernel:
            raise NotImplementedError("Quantum kernel integration is only available in the QML backend.")
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        return torch.exp(-self.config.latent_dim * diff.pow(2).sum(-1))

    def fidelity_adjacency(
        self,
        states: torch.Tensor,
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> torch.Tensor:
        """Build a weighted adjacency matrix from pairwise cosine similarities."""
        norm = states / (states.norm(dim=-1, keepdim=True) + 1e-12)
        sims = norm @ norm.t()
        adjacency = torch.zeros_like(sims)
        adjacency[sims >= threshold] = 1.0
        if secondary is not None:
            mask = (sims >= secondary) & (sims < threshold)
            adjacency[mask] = secondary_weight
        return adjacency

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    use_quantum_kernel: bool = False,
) -> HybridAutoencoder:
    """Factory mirroring the QML helper."""
    cfg = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_quantum_kernel=use_quantum_kernel,
    )
    return HybridAutoencoder(cfg)

def train_autoencoder(
    model: HybridAutoencoder,
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
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

__all__ = ["HybridAutoencoder", "HybridAutoencoderConfig", "Autoencoder", "train_autoencoder"]
