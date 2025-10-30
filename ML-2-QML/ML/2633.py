from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

# Import the classical SamplerQNN from the same repository
from SamplerQNN import SamplerQNN

def _as_tensor(data: torch.Tensor | torch.Tensor) -> torch.Tensor:
    """Convert any array‑like input to a float32 torch tensor."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

@dataclass
class AutoencoderHybridConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1
    quantum_input_dim: int = 2  # dimensionality that the quantum sampler expects

class AutoencoderHybridNet(nn.Module):
    """Hybrid classical–quantum autoencoder.

    The encoder is a standard MLP that maps the input to a latent vector.
    The latent vector is projected to ``quantum_input_dim`` dimensions,
    fed into a quantum SamplerQNN, and the resulting 2‑dimensional output
    is transformed back to the full reconstruction through a decoder MLP.
    """
    def __init__(self, cfg: AutoencoderHybridConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Classical encoder
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

        # Projection to quantum input dimension
        self.latent_to_q = nn.Linear(cfg.latent_dim, cfg.quantum_input_dim)

        # Quantum sampler network
        self.qnn = SamplerQNN()

        # Decoder mapping from quantum output back to input space
        dec_layers = []
        in_dim = cfg.quantum_input_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def quantum_forward(self, z: torch.Tensor) -> torch.Tensor:
        q_in = self.latent_to_q(z)
        return self.qnn(q_in)

    def decode(self, q_out: torch.Tensor) -> torch.Tensor:
        return self.decoder(q_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        q_out = self.quantum_forward(z)
        return self.decode(q_out)

def AutoencoderHybrid(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int,...] = (128, 64),
    dropout: float = 0.1,
    quantum_input_dim: int = 2,
) -> AutoencoderHybridNet:
    cfg = AutoencoderHybridConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        quantum_input_dim=quantum_input_dim,
    )
    return AutoencoderHybridNet(cfg)

def train_autoencoder_hybrid(
    model: AutoencoderHybridNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Train the hybrid autoencoder and return loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

__all__ = [
    "AutoencoderHybrid",
    "AutoencoderHybridNet",
    "AutoencoderHybridConfig",
    "train_autoencoder_hybrid",
]
