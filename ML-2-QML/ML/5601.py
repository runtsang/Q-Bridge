"""
GraphQNNGen – Classical hybrid utilities.

This module implements:
* Random graph‑based neural networks (classical and quantum).
* Feed‑forward propagation for both classical tensors and quantum Qobj states.
* Fidelity‑based graph construction.
* Classical QCNN, Quantum‑NAT inspired fully‑connected model, and auto‑encoder.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional

import networkx as nx
import torch
import torch.nn as nn

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Random network generation and training data
# ---------------------------------------------------------------------------

def _random_linear(in_features: int, out_features: int, generator: Optional[torch.Generator] = None) -> Tensor:
    """Generate a random weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32, generator=generator)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Create synthetic (x, Wx) pairs for a given linear layer."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network_classical(arch: Sequence[int], samples: int, seed: Optional[int] = None):
    """Build a simple feed‑forward network with random weights."""
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    weights: List[Tensor] = [
        _random_linear(in_f, out_f, generator) for in_f, out_f in zip(arch[:-1], arch[1:])
    ]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(arch), weights, training_data, target_weight


# ---------------------------------------------------------------------------
# Classical feed‑forward and fidelity utilities
# ---------------------------------------------------------------------------

def feedforward_classical(arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
    """Return activations for each layer of a classical network."""
    activations: List[List[Tensor]] = []
    for features, _ in samples:
        current = features
        layerwise = [current]
        for weight in weights:
            current = torch.tanh(weight @ current)
            layerwise.append(current)
        activations.append(layerwise)
    return activations


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap of two normalized vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)


def fidelity_adjacency(states: Sequence[Tensor], threshold: float, *, secondary: Optional[float] = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Create a weighted graph from state fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G


# ---------------------------------------------------------------------------
# Classical QCNN implementation
# ---------------------------------------------------------------------------

class QCNNModel(nn.Module):
    """Stack of fully‑connected layers emulating the quantum convolution steps."""

    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


def QCNN() -> QCNNModel:
    """Factory returning the configured :class:`QCNNModel`."""
    return QCNNModel()


# ---------------------------------------------------------------------------
# Classical Quantum‑NAT inspired model
# ---------------------------------------------------------------------------

class QFCModel(nn.Module):
    """Simple CNN followed by a fully‑connected projection to four features."""

    def __init__(self, input_dim: int = 16, latent_dim: int = 4, hidden_dims: Tuple[int, int] = (64, 32)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, latent_dim))
        self.norm = nn.BatchNorm1d(latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)


# ---------------------------------------------------------------------------
# Classical auto‑encoder
# ---------------------------------------------------------------------------

@dataclass
class AutoencoderConfig:
    """Configuration values for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """A lightweight multilayer perceptron auto‑encoder."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
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

    def encode(self, inputs: Tensor) -> Tensor:
        return self.encoder(inputs)

    def decode(self, latents: Tensor) -> Tensor:
        return self.decoder(latents)

    def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    """Factory that mirrors the quantum helper returning a configured network."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderNet(config)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "feedforward_classical",
    "fidelity_adjacency",
    "random_network_classical",
    "random_training_data",
    "state_fidelity",
    "QCNN",
    "QCNNModel",
    "QFCModel",
    "Autoencoder",
    "AutoencoderNet",
    "AutoencoderConfig",
]
