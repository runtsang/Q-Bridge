"""
Graph‑Quantum Neural Network with classical autoencoder.
This module defines:
  * AutoencoderConfig, AutoencoderNet, Autoencoder factory and training routine,
    mirroring the PyTorch implementation from Autoencoder.py.
  * GraphQNNAutoencoder – a hybrid GNN that first encodes node features
    through the autoencoder and then propagates them via a sequence of
    classical weight matrices.
  * Utility functions for random network generation, fidelity‑based graph
    construction and state fidelity.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# 1. Classical autoencoder
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """Lightweight fully‑connected autoencoder."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = self._make_mlp(
            config.input_dim,
            config.hidden_dims,
            config.latent_dim,
            config.dropout,
        )
        self.decoder = self._make_mlp(
            config.latent_dim,
            tuple(reversed(config.hidden_dims)),
            config.input_dim,
            config.dropout,
        )

    @staticmethod
    def _make_mlp(
        in_dim: int,
        hidden_dims: Tuple[int,...],
        out_dim: int,
        dropout: float,
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Factory mirroring the original Autoencoder.py."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(cfg)


def train_autoencoder(
    model: AutoencoderNet,
    data: Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Train the autoencoder and return the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dataset = TensorDataset(data.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        history.append(epoch_loss / len(dataset))
    return history


# --------------------------------------------------------------------------- #
# 2. Classical Graph‑QNN core
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    return [
        (torch.randn(weight.size(1)), weight @ torch.randn(weight.size(1)))
        for _ in range(samples)
    ]


def random_network(qnn_arch: Sequence[int], samples: int):
    weights: List[Tensor] = [
        _random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])
    ]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    activations: List[List[Tensor]] = []
    for features, _ in samples:
        current = features
        layerwise = [current]
        for w in weights:
            current = torch.tanh(w @ current)
            layerwise.append(current)
        activations.append(layerwise)
    return activations


def state_fidelity(a: Tensor, b: Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# 3. Hybrid Graph‑QNN with autoencoder
# --------------------------------------------------------------------------- #
class GraphQNNAutoencoder:
    """
    Classical GNN that first maps node features through a trained
    autoencoder and then propagates them with a stack of linear layers.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        ae_cfg: AutoencoderConfig | None = None,
    ) -> None:
        self.qnn_arch = list(qnn_arch)
        self.weights = [
            _random_linear(in_f, out_f)
            for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])
        ]
        self.autoencoder = Autoencoder(
            ae_cfg.input_dim if ae_cfg else qnn_arch[0],
            latent_dim=ae_cfg.latent_dim if ae_cfg else 32,
            hidden_dims=ae_cfg.hidden_dims if ae_cfg else (128, 64),
            dropout=ae_cfg.dropout if ae_cfg else 0.1,
        )

    def encode_inputs(self, inputs: Tensor) -> Tensor:
        return self.autoencoder.encode(inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        encoded = self.encode_inputs(inputs)
        x = encoded
        for w in self.weights:
            x = torch.tanh(w @ x)
        return x

    def train_autoencoder(
        self,
        data: Tensor,
        *,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: torch.device | None = None,
    ) -> List[float]:
        return train_autoencoder(self.autoencoder, data, epochs=epochs,
                                 batch_size=batch_size, lr=lr,
                                 weight_decay=weight_decay, device=device)

    def feedforward(
        self,
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        activations: List[List[Tensor]] = []
        for features, _ in samples:
            current = self.encode_inputs(features)
            layerwise = [current]
            for w in self.weights:
                current = torch.tanh(w @ current)
                layerwise.append(current)
            activations.append(layerwise)
        return activations

    def fidelity_graph(self, states: Sequence[Tensor], threshold: float) -> nx.Graph:
        return fidelity_adjacency(states, threshold)


__all__ = [
    "AutoencoderConfig",
    "AutoencoderNet",
    "Autoencoder",
    "train_autoencoder",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "GraphQNNAutoencoder",
]
