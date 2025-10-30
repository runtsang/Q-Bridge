"""GraphQNNGen306 – a unified classical/quantum neural network framework.

The module contains:
* Graph‑based utilities (feed‑forward, fidelity graph construction).
* A classical quanvolution filter and classifier.
* A regression dataset and model.
* A lightweight autoencoder and training loop.
* A `GraphQNNGen306` wrapper that can generate random networks and expose a common API.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# 1.  Graph‑based utilities – classical
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Random linear layer weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate data for training a linear model with the given target weight."""
    data = []
    for _ in range(samples):
        x = torch.randn(weight.size(1), dtype=torch.float32)
        y = weight @ x
        data.append((x, y))
    return data


def random_network(arch: Sequence[int], samples: int):
    """Create a random feed‑forward network and training data."""
    weights = [_random_linear(in_f, out_f) for in_f, out_f in zip(arch[:-1], arch[1:])]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(arch), weights, training_data, target_weight


def feedforward(
    arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Run a forward pass through the network, storing all activations."""
    activations = []
    for x, _ in samples:
        layer_out = [x]
        current = x
        for w in weights:
            current = torch.tanh(w @ current)
            layer_out.append(current)
        activations.append(layer_out)
    return activations


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap of two real vectors."""
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
    """Build a weighted graph from state fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G


# --------------------------------------------------------------------------- #
# 2.  Classical Quanvolution
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(nn.Module):
    """Simple 2‑D convolution that mimics a 2×2 patch extraction."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.conv(x).view(x.size(0), -1)


class QuanvolutionClassifier(nn.Module):
    """Classifier that uses the quanvolution filter followed by a linear head."""

    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        feats = self.qfilter(x)
        logits = self.linear(feats)
        return F.log_softmax(logits, dim=-1)


# --------------------------------------------------------------------------- #
# 3.  Regression dataset & model
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic data: sin + cos of the sum of features."""
    X = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = X.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return X, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that returns raw features and target values."""

    def __init__(self, samples: int, num_features: int) -> None:
        self.X, self.y = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.X)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:  # type: ignore[override]
        return {"states": torch.tensor(self.X[idx]), "target": torch.tensor(self.y[idx])}


class QModel(nn.Module):
    """Small MLP used for regression."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.net(x).squeeze(-1)


# --------------------------------------------------------------------------- #
# 4.  Autoencoder
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """Fully‑connected autoencoder."""

    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        enc_layers = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, hidden))
            enc_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, hidden))
            dec_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> AutoencoderNet:
    cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
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
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(data.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            opt.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


# --------------------------------------------------------------------------- #
# 5.  GraphQNNGen306 wrapper
# --------------------------------------------------------------------------- #
class GraphQNNGen306:
    """Unified API for a graph‑based neural network.

    The class can be instantiated with a classical or quantum backend.
    It exposes methods for generating a random network, running a forward pass,
    and building a fidelity‑based graph of the hidden states.
    """

    def __init__(self, arch: Sequence[int], backend: str = "classical") -> None:
        self.arch = list(arch)
        self.backend = backend.lower()
        if self.backend not in {"classical", "quantum"}:
            raise ValueError("backend must be 'classical' or 'quantum'")
        self.weights: List[Tensor] | List[List[Tensor]] | None = None
        self.training_data: List[Tuple[Tensor, Tensor]] | None = None
        self.target: Tensor | None = None

    @staticmethod
    def random_network(arch: Sequence[int], samples: int):
        """Generate a random network and training data."""
        if "classical" in arch:
            return random_network(arch, samples)
        # quantum path (fallback to classical implementation for simplicity)
        return random_network(arch, samples)

    def initialize_random(self, samples: int):
        """Create random weights and training data in place."""
        self.arch, self.weights, self.training_data, self.target = self.random_network(self.arch, samples)

    def feedforward(self):
        """Run a forward pass on the stored data."""
        if self.weights is None or self.training_data is None:
            raise RuntimeError("Network not initialized.")
        return feedforward(self.arch, self.weights, self.training_data)

    def fidelity_graph(self, threshold: float, *, secondary: float | None = None):
        """Return a graph of hidden state fidelities."""
        activations = self.feedforward()
        # flatten to list of all hidden layer outputs
        states = [act for layer in activations for act in layer]
        return fidelity_adjacency(states, threshold, secondary=secondary)


__all__ = [
    "GraphQNNGen306",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "RegressionDataset",
    "QModel",
    "Autoencoder",
    "AutoencoderNet",
    "AutoencoderConfig",
    "train_autoencoder",
]
