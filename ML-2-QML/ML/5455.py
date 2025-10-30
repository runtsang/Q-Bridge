"""Combined classical estimator with kernel, graph, and autoencoder utilities.

The module defines a single public class `CombinedEstimatorQNN` that
encapsulates four sub‑modules:

* `RegressionModel` – a small feed‑forward regressor (mirroring EstimatorQNN).
* `KernelModel` – an RBF kernel compatible with TorchQuantum.
* `GraphModel` – utilities for state fidelity graphs and random network generation.
* `AutoencoderModel` – a lightweight MLP autoencoder.

Each sub‑module is fully testable on its own, but the public class
offers a coherent interface for hybrid experiments.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------------------------------- #
# 1. Regression component (from EstimatorQNN)
# --------------------------------------------------------------------------- #
class RegressionModel(nn.Module):
    """Simple fully‑connected regressor."""

    def __init__(self, input_dim: int = 2, hidden_dims: Tuple[int,...] = (8, 4)) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# 2. Kernel component (from QuantumKernelMethod)
# --------------------------------------------------------------------------- #
class RBFKernel(nn.Module):
    """Radial basis function kernel."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix for two batches using the RBF kernel."""
    kernel = RBFKernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


# --------------------------------------------------------------------------- #
# 3. Graph utilities (from GraphQNN)
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate synthetic regression data from a target weight matrix."""
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random dense network and training data."""
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    """Forward pass through a classical feed‑forward network."""
    stored: List[List[torch.Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap between two unit‑norm vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, ai), (j, bj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(ai, bj)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# 4. Autoencoder component (from Autoencoder)
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


class AutoencoderNet(nn.Module):
    """Multilayer perceptron autoencoder."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Training loop returning the loss history."""
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


# --------------------------------------------------------------------------- #
# 5. Public hybrid interface
# --------------------------------------------------------------------------- #
class CombinedEstimatorQNN:
    """Unified estimator that exposes classical and quantum components.

    Attributes
    ----------
    regression : RegressionModel
        Feed‑forward regressor.
    kernel : RBFKernel
        Classical RBF kernel (also used by the TorchQuantum implementation).
    graph : dict
        Holds graph utilities and random network generation.
    autoencoder : AutoencoderNet
        Autoencoder used for dimensionality reduction.
    """

    def __init__(self, input_dim: int = 2, latent_dim: int = 32) -> None:
        self.regression = RegressionModel(input_dim)
        self.kernel = RBFKernel()
        self.autoencoder = AutoencoderNet(AutoencoderConfig(input_dim, latent_dim))
        self.graph = {
            "random_network": random_network,
            "feedforward": feedforward,
            "fidelity_adjacency": fidelity_adjacency,
        }

    # --------------------------------------------------------------------- #
    # 5.1 Regression helpers
    # --------------------------------------------------------------------- #
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return regression predictions."""
        return self.regression(X)

    def train_regression(
        self,
        data: torch.Tensor,
        *,
        epochs: int = 200,
        lr: float = 1e-3,
        device: torch.device | None = None,
    ) -> None:
        """Train the regression network on a dataset."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.regression.to(device)
        dataset = TensorDataset(_as_tensor(data))
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(self.regression.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for _ in range(epochs):
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                pred = self.regression(batch)
                loss = loss_fn(pred, batch)
                loss.backward()
                optimizer.step()

    # --------------------------------------------------------------------- #
    # 5.2 Kernel utilities
    # --------------------------------------------------------------------- #
    def gram_matrix(self, X: Sequence[torch.Tensor], Y: Sequence[torch.Tensor]) -> np.ndarray:
        """Convenience wrapper around :func:`kernel_matrix`."""
        return kernel_matrix(X, Y, gamma=self.kernel.gamma)

    # --------------------------------------------------------------------- #
    # 5.3 Graph utilities
    # --------------------------------------------------------------------- #
    def build_graph(self, states: Sequence[torch.Tensor], threshold: float) -> nx.Graph:
        """Return a fidelity‑based graph."""
        return fidelity_adjacency(states, threshold)

    # --------------------------------------------------------------------- #
    # 5.4 Autoencoder helpers
    # --------------------------------------------------------------------- #
    def encode(self, X: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(X)

    def decode(self, Z: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(Z)

    def train_autoencoder(
        self,
        data: torch.Tensor,
        *,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: torch.device | None = None,
    ) -> List[float]:
        return train_autoencoder(
            self.autoencoder, data,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )

__all__ = ["CombinedEstimatorQNN"]
