"""Hybrid classical autoencoder with graph‑based latent space and optional classifier.

The module builds on the original Autoencoder.py while incorporating
GraphQNN utilities for fidelity‑based adjacency graphs and adding a
classifier head.  Training and inference follow the same patterns
as the original implementation but now expose the latent graph
directly.

The public API mirrors the anchor file:
    * `HybridAutoencoder` – factory returning a configured `HybridAutoencoderNet`.
    * `train_hybrid_autoencoder` – training loop returning loss history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import networkx as nx
import numpy as np

# Import graph utilities from the shared GraphQNN module
try:
    # Prefer relative import if this file is part of a package
    from.GraphQNN import fidelity_adjacency as _fidelity_adjacency
except Exception:
    # Fallback to absolute import – the seed module may be on sys.path
    from GraphQNN import fidelity_adjacency as _fidelity_adjacency


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


@dataclass
class AutoencoderConfig:
    """Configuration values for :class:`HybridAutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    use_classifier: bool = False
    classifier_hidden: int = 32


class HybridAutoencoderNet(nn.Module):
    """A dense autoencoder with an optional classifier head and a latent graph."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Encoder
        encoder_layers: List[nn.Module] = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers: List[nn.Module] = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Optional classifier
        if config.use_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(config.latent_dim, config.classifier_hidden),
                nn.ReLU(),
                nn.Linear(config.classifier_hidden, 2),
            )
        else:
            self.classifier = None

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the latent representation."""
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent space."""
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Standard autoencoder forward pass."""
        return self.decode(self.encode(inputs))

    def latent_graph(self, inputs: torch.Tensor, threshold: float = 0.9) -> nx.Graph:
        """
        Build a fidelity graph from the batch of latent vectors.

        Parameters
        ----------
        inputs : torch.Tensor
            Input batch.
        threshold : float
            Fidelity threshold for edge existence.

        Returns
        -------
        nx.Graph
            Weighted graph where nodes are latent samples and edges
            represent high‑fidelity similarity.
        """
        latents = self.encode(inputs).detach().cpu().numpy()
        # Use classical fidelity (cosine similarity) as proxy
        norms = np.linalg.norm(latents, axis=1, keepdims=True) + 1e-12
        normalized = latents / norms
        dot_matrix = normalized @ normalized.T
        graph = nx.Graph()
        graph.add_nodes_from(range(len(latents)))
        for i in range(len(latents)):
            for j in range(i + 1, len(latents)):
                fid = dot_matrix[i, j] ** 2
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
        return graph


def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    use_classifier: bool = False,
    classifier_hidden: int = 32,
) -> HybridAutoencoderNet:
    """Factory that returns a configured hybrid autoencoder."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_classifier=use_classifier,
        classifier_hidden=classifier_hidden,
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


__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderNet",
    "train_hybrid_autoencoder",
    "AutoencoderConfig",
]
