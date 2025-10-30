"""Hybrid classical autoencoder that uses a Qiskit SamplerQNN as encoder.

The module keeps the same public API as the original Autoencoder.py
so downstream code can call Autoencoder(...).  Internally it
creates a decoder MLP and wraps a quantum encoder supplied via
``quantum_encoder``.  The quantum encoder must be a torch.nn.Module
(e.g. a Qiskit SamplerQNN) that maps an input vector to a latent
vector of shape ``(batch, latent_dim)``.  If no encoder is supplied
a simple random‑noise encoder is used.

The training loop uses PyTorch's autograd to jointly optimise the
quantum and classical parameters.  A helper ``latent_graph`` builds
a similarity graph of latent codes using state fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import networkx as nx
import numpy as np


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Convert input to a float32 tensor."""
    if isinstance(data, torch.Tensor):
        return data.float()
    return torch.as_tensor(data, dtype=torch.float32)


def _default_quantum_encoder(x: torch.Tensor) -> torch.Tensor:
    """Fallback encoder that adds Gaussian noise to the first feature."""
    noise = torch.randn_like(x[:, :1]) * 0.01
    latent = torch.cat([x[:, :1], noise], dim=-1)
    return latent


@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    quantum_encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


class AutoencoderNet(nn.Module):
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        self.cfg = cfg

        # Decoder MLP
        layers = []
        in_dim = cfg.latent_dim
        for h in cfg.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*layers)

        # Quantum encoder wrapper
        self.quantum_encoder = cfg.quantum_encoder or _default_quantum_encoder

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantum_encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    quantum_encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> AutoencoderNet:
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        quantum_encoder=quantum_encoder,
    )
    return AutoencoderNet(cfg)


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
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


def latent_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return squared overlap between two latent vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def latent_graph(
    model: AutoencoderNet,
    data: torch.Tensor,
    threshold: float = 0.8,
    secondary: Optional[float] = None,
) -> nx.Graph:
    """Build a similarity graph of latent codes for the given data."""
    model.eval()
    with torch.no_grad():
        latent_codes = model.encode(_as_tensor(data).to(next(model.parameters()).device))
    latent_codes = latent_codes.cpu()
    graph = nx.Graph()
    graph.add_nodes_from(range(len(latent_codes)))
    for i in range(len(latent_codes)):
        for j in range(i + 1, len(latent_codes)):
            fid = latent_fidelity(latent_codes[i], latent_codes[j])
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary)
    return graph


__all__ = [
    "Autoencoder",
    "AutoencoderNet",
    "AutoencoderConfig",
    "train_autoencoder",
    "latent_graph",
]
