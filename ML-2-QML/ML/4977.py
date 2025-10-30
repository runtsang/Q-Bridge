"""Hybrid classical implementation of QuantumNATGen222.

This module packages together a convolutional feature extractor, a
fully‑connected classifier, and a graph‑based neural network helper
while preserving the interface of the original reference modules.
"""

from __future__ import annotations

import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

# --------------------------------------------------------------------------- #
#  Core CNN + FC + Classifier
# --------------------------------------------------------------------------- #

class _CNNEncoder(nn.Module):
    """Simple CNN feature extractor used in the original QuantumNAT."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.norm(x)


class _Classifier(nn.Module):
    """Stacked linear layers with ReLU activations and a 2‑class head."""

    def __init__(self, num_features: int, depth: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --------------------------------------------------------------------------- #
#  Fast Estimator (classical)
# --------------------------------------------------------------------------- #

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastEstimator:
    """Deterministic estimator with optional Gaussian shot noise."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        self.model.eval()
        with torch.no_grad():
            raw = [
                [
                    float(
                        obs(self.model(torch.as_tensor(params, dtype=torch.float32)))
                       .mean()
                       .cpu()
                    )
                    for obs in observables
                ]
                for params in parameter_sets
            ]
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        return [
            [float(rng.normal(m, max(1e-6, 1 / shots))) for m in row] for row in raw
        ]


# --------------------------------------------------------------------------- #
#  Graph‑based helpers
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
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
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> "nx.Graph":
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
#  Hybrid class combining everything
# --------------------------------------------------------------------------- #

class QuantumNATGen222(nn.Module):
    """
    Hybrid classical model that mirrors the functionality of the original
    QuantumNAT examples while adding graph‑based utilities and an estimator.

    Parameters
    ----------
    num_features : int
        Dimensionality of the linear layers in the classifier part.
    depth : int
        Depth of the classifier stack.
    graph_arch : Sequence[int] | None
        Architecture of the graph‑based neural network.  If ``None`` the
        graph helpers are still available but not pre‑instantiated.
    """

    def __init__(
        self,
        num_features: int = 8,
        depth: int = 2,
        graph_arch: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = _CNNEncoder()
        self.classifier = _Classifier(num_features, depth)
        self._graph_arch = list(graph_arch) if graph_arch is not None else None
        if self._graph_arch:
            self.graph_weights, self.graph_training, self._target, _ = random_network(
                self._graph_arch, samples=100
            )
        self.estimator = FastEstimator(self)

    # ----------------------------------------------------------------------- #
    #  Forward pass
    # ----------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the concatenated CNN + classifier output."""
        features = self.encoder(x)
        out = self.classifier(features)
        return out

    # ----------------------------------------------------------------------- #
    #  Estimation helpers
    # ----------------------------------------------------------------------- #
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Delegate to :class:`FastEstimator` with optional shot noise."""
        return self.estimator.evaluate(
            observables, parameter_sets, shots=shots, seed=seed
        )

    # ----------------------------------------------------------------------- #
    #  Graph utilities
    # ----------------------------------------------------------------------- #
    @property
    def graph_arch(self) -> Sequence[int] | None:
        return self._graph_arch

    @property
    def graph_weights(self) -> List[torch.Tensor] | None:
        return self.graph_weights if hasattr(self, "graph_weights") else None

    def graph_feedforward(
        self, samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]
    ) -> List[List[torch.Tensor]]:
        """Run the stored graph network on given samples."""
        if not self.graph_weights:
            raise RuntimeError("Graph network not initialized.")
        return feedforward(self._graph_arch, self.graph_weights, samples)

    def graph_fidelity_adjacency(
        self, threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5
    ) -> "nx.Graph":
        """Build a fidelity‑based adjacency graph from the stored training data."""
        if not self.graph_weights:
            raise RuntimeError("Graph network not initialized.")
        states = [s[0] for s in self.graph_training]
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

__all__ = ["QuantumNATGen222"]
