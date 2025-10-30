"""Hybrid classical estimator combining autoencoder, self‑attention and graph aggregation.

The model mirrors the structure of the reference EstimatorQNN but augments it with:
* an encoder/decoder pair that compresses the 2‑dimensional input into a latent
  representation (autoencoder style),
* a self‑attention block that learns pairwise interactions between latent
  dimensions,
* a graph‑convolution step that aggregates features according to a fidelity‑based
  adjacency (inspired by the graph QNN helpers),
* a regression head that outputs a single value.

The class is fully PyTorch‑compatible and can be trained with standard
optimisers.  The quantum equivalent is provided in the QML module.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class HybridConfig:
    """Configuration for :class:`HybridEstimatorNet`."""
    input_dim: int = 2
    latent_dim: int = 8
    hidden_dims: Tuple[int,...] = (32, 16)
    attention_dim: int = 4
    graph_threshold: float = 0.8
    dropout: float = 0.1


# --------------------------------------------------------------------------- #
# Self‑attention helper
# --------------------------------------------------------------------------- #

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, embed_dim: int = 4) -> None:
        super().__init__()
        self.proj_q = nn.Linear(dim, embed_dim, bias=False)
        self.proj_k = nn.Linear(dim, embed_dim, bias=False)
        self.proj_v = nn.Linear(dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)
        scores = F.softmax((q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5), dim=-1)
        return self.out_proj(scores @ v)


# --------------------------------------------------------------------------- #
# Autoencoder
# --------------------------------------------------------------------------- #

class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: Tuple[int,...], dropout: float) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: Tuple[int,...], dropout: float) -> None:
        super().__init__()
        layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


# --------------------------------------------------------------------------- #
# Graph aggregation
# --------------------------------------------------------------------------- #

def build_fidelity_graph(states: Tensor, threshold: float) -> nx.Graph:
    """Return a graph where nodes are batch elements and edges are added when
    the cosine similarity of their latent vectors exceeds ``threshold``."""
    n = states.size(0)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    norms = states.norm(p=2, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=1e-12)
    normalized = states / norms
    sims = normalized @ normalized.t()
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= threshold:
                graph.add_edge(i, j, weight=1.0)
    return graph


def graph_aggregate(states: Tensor, graph: nx.Graph) -> Tensor:
    """Simple mean‑pool aggregation over neighbours defined by ``graph``."""
    n = states.size(0)
    agg = torch.zeros_like(states)
    for i in range(n):
        neighbors = list(graph.neighbors(i))
        if neighbors:
            agg[i] = states[neighbors].mean(dim=0)
        else:
            agg[i] = states[i]
    return agg


# --------------------------------------------------------------------------- #
# Hybrid estimator
# --------------------------------------------------------------------------- #

class HybridEstimatorNet(nn.Module):
    """Classical hybrid estimator combining encoder, attention and graph pooling."""
    def __init__(self, config: HybridConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(config.input_dim, config.latent_dim, config.hidden_dims, config.dropout)
        self.attention = SelfAttentionBlock(config.latent_dim, config.attention_dim)
        self.graph_threshold = config.graph_threshold
        self.regressor = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[0], 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        z = self.attention(z)
        graph = build_fidelity_graph(z, self.graph_threshold)
        z = graph_aggregate(z, graph)
        out = self.regressor(z)
        return out.squeeze(-1)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)


def HybridEstimatorQNN(config: HybridConfig) -> HybridEstimatorNet:
    return HybridEstimatorNet(config)


# --------------------------------------------------------------------------- #
# Training helper
# --------------------------------------------------------------------------- #

def train_hybrid(
    model: HybridEstimatorNet,
    data: Tensor,
    targets: Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> List[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data), _as_tensor(targets))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


def _as_tensor(data: Iterable[float] | Tensor) -> Tensor:
    if isinstance(data, Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)


__all__ = [
    "HybridEstimatorNet",
    "HybridEstimatorQNN",
    "HybridConfig",
    "train_hybrid",
]
