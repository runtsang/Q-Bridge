import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np

from.quantum_autoencoder import QuantumLatentEncoder

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    use_attention: bool = False

class ClassicalSelfAttention(nn.Module):
    """A lightweight self‑attention block mimicking the classical helper."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        query = inputs @ rotation_params.reshape(self.embed_dim, -1)
        key = inputs @ entangle_params.reshape(self.embed_dim, -1)
        scores = nn.functional.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs

class AutoencoderNet(nn.Module):
    """Classical fully‑connected auto‑encoder."""
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder
        enc_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Optional attention
        self.attn = ClassicalSelfAttention(cfg.latent_dim) if cfg.use_attention else None

        # Decoder
        dec_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if self.attn is not None:
            # dummy parameters for the sake of illustration
            rot = torch.randn(1, self.cfg.latent_dim, device=z.device)
            ent = torch.randn(1, self.cfg.latent_dim, device=z.device)
            z = self.attn(rot, ent, z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

class HybridAutoencoder(nn.Module):
    """Hybrid auto‑encoder that uses a quantum circuit for latent reconstruction."""
    def __init__(self, cfg: AutoencoderConfig, quantum_cfg: dict):
        super().__init__()
        self.auto = AutoencoderNet(cfg)
        self.quantum = QuantumLatentEncoder(**quantum_cfg)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.auto.encode(x)

    def quantum_reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        return self.quantum(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.auto.decode(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        qz = self.quantum_reconstruct(z)
        return self.decode(qz)

def _as_tensor(data: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Utility to ensure a float32 tensor."""
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    quantum_weight: float = 0.1,
    device: torch.device | None = None,
) -> List[float]:
    """Train the hybrid auto‑encoder with a combined MSE + quantum loss."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ds = TensorDataset(_as_tensor(data))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    history: List[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            opt.zero_grad()
            # Classical reconstruction
            rec = model(batch)
            loss_cls = mse(rec, batch)
            # Quantum reconstruction
            z = model.encode(batch)
            qrec = model.quantum_reconstruct(z)
            loss_q = mse(qrec, batch)
            loss = loss_cls + quantum_weight * loss_q
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(ds)
        history.append(epoch_loss)
    return history

__all__ = [
    "AutoencoderConfig",
    "AutoencoderNet",
    "HybridAutoencoder",
    "train_hybrid_autoencoder",
]
