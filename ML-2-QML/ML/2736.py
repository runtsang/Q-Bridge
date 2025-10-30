"""Hybrid graph neural network with classical autoencoding.

This module merges the graph‑based propagation from the original
GraphQNN with a lightweight torch autoencoder.  Node features are
propagated through a sequence of linear layers that respect the graph
adjacency, then flattened and compressed by an MLP autoencoder.
The resulting latent vector can be used for downstream tasks or
re‑constructed back to node embeddings.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import List, Tuple

import networkx as nx
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

Tensor = torch.Tensor


def _as_tensor(data: Iterable[float] | Tensor) -> Tensor:
    """Utility to ensure a float32 tensor on the default device."""
    if isinstance(data, Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix for a linear layer."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic (input, target) pairs for a linear layer."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random linear network and synthetic training data."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Propagate each sample through the linear network."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap between two normalized tensors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


@dataclass
class AutoencoderConfig:
    """Configuration for the autoencoder part of the hybrid model."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """Simple fully‑connected autoencoder used inside the hybrid graph model."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
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

    def encode(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(latents)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.decode(self.encode(inputs))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Factory mirroring the quantum helper."""
    return AutoencoderNet(
        AutoencoderConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    )


class GraphQNNAutoencoder(nn.Module):
    """
    Hybrid graph‑neural‑network + autoencoder.

    * The graph propagation uses a stack of linear layers whose
      dimensions are given by ``qnn_arch``.
    * After the last linear layer the node embeddings are flattened
      and passed through a classical autoencoder.
    * The latent representation can be used for graph clustering,
      node classification, or as a compact graph descriptor.
    """
    def __init__(
        self,
        qnn_arch: Sequence[int],
        latent_dim: int = 32,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.linear_layers = nn.ModuleList()
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            self.linear_layers.append(nn.Linear(in_f, out_f))
        # Flatten all node embeddings into a single vector
        flat_dim = qnn_arch[-1] * qnn_arch[-1]
        self.autoencoder = Autoencoder(
            input_dim=flat_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def forward(self, node_features: Tensor, adjacency: Tensor) -> Tensor:
        """
        node_features: (num_nodes, feature_dim)
        adjacency: (num_nodes, num_nodes) adjacency matrix (0/1)
        """
        hidden = node_features
        for layer in self.linear_layers:
            hidden = torch.tanh(layer(hidden))
            hidden = torch.matmul(adjacency, hidden)
        flat = hidden.reshape(-1)
        return self.autoencoder(flat)

    def encode(self, node_features: Tensor, adjacency: Tensor) -> Tensor:
        hidden = node_features
        for layer in self.linear_layers:
            hidden = torch.tanh(layer(hidden))
            hidden = torch.matmul(adjacency, hidden)
        flat = hidden.reshape(-1)
        return self.autoencoder.encode(flat)

    def decode(self, latents: Tensor) -> Tensor:
        return self.autoencoder.decode(latents)


def train_autoencoder(
    model: nn.Module,
    data: Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Generic training loop for the hybrid model."""
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
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
    "train_autoencoder",
    "GraphQNNAutoencoder",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
