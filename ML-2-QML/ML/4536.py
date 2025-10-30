"""Hybrid Graph Neural Network implementation with classical PyTorch backbone and optional QCNN feature map.

The class combines:
- random network generation and feed‑forward propagation from the original GraphQNN.
- fidelity‑based graph construction.
- a lightweight estimator that can inject Gaussian shot noise (FastEstimator).
- an optional QCNN feature‑map block that mimics the quantum convolutional layers.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch import nn

import networkx as nx

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

# -------------------------------------------------------------------------

def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    """Return a random weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

# -------------------------------------------------------------------------

class QCNNFeatureMap(nn.Module):
    """A lightweight, fully‑connected replacement for the quantum convolutional layers
    used in the original QCNN example.  It is inserted as the first block of the
    classical network so that the overall architecture can be trained end‑to‑end.
    """
    def __init__(self, in_features: int = 8, hidden: int = 16) -> None:
        super().__init__()
        self.map = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.map(x)

# -------------------------------------------------------------------------

class FastEstimator:
    """Wraps a torch.nn.Module and adds optional Gaussian shot noise to the
    deterministic predictions.
    """
    def __init__(self, model: nn.Module, shots: int | None = None, seed: int | None = None) -> None:
        self.model = model
        self.shots = shots
        self.rng = np.random.default_rng(seed)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                if self.shots is not None:
                    row = [float(self.rng.normal(mean, max(1e-6, 1 / self.shots))) for mean in row]
                results.append(row)
        return results

# -------------------------------------------------------------------------

def random_network(qnn_arch: Sequence[int], samples: int = 10) -> Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Generate a random network consistent with the current architecture
    and return training data for the last layer.
    """
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        w = _random_linear(in_f, out_f)
        weights.append(w)
    target_weight = weights[-1]
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(target_weight.size(1), dtype=torch.float32)
        target = target_weight @ features
        dataset.append((features, target))
    return list(qnn_arch), weights, dataset, target_weight

# -------------------------------------------------------------------------

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    """Propagate the inputs through the network and return activations per layer."""
    activations: List[List[torch.Tensor]] = []
    for features, _ in samples:
        layerwise = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            layerwise.append(current)
        activations.append(layerwise)
    return activations

# -------------------------------------------------------------------------

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap of two classical state vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

# -------------------------------------------------------------------------

def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# -------------------------------------------------------------------------

class GraphQNNHybrid(nn.Module):
    """Hybrid graph‑neural‑network that can be instantiated either purely
    classically (using PyTorch) or with a quantum feature‑map block.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Layer sizes of the network.  The first element is the input dimension.
    use_qcnn : bool, default=True
        When True the network starts with a QCNNFeatureMap block; otherwise a
        simple linear layer is used.
    shots : int | None, optional
        If provided, Gaussian noise with variance 1/shots is added to the
        predictions.
    seed : int | None, optional
        Seed for the noise generator.
    """
    def __init__(
        self,
        qnn_arch: Sequence[int],
        use_qcnn: bool = True,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.arch = list(qnn_arch)
        self.use_qcnn = use_qcnn
        self.weights: List[torch.Tensor] = []

        layers: List[nn.Module] = []
        if use_qcnn:
            layers.append(QCNNFeatureMap(in_features=qnn_arch[0]))
            prev = qnn_arch[0]
        else:
            prev = qnn_arch[0]
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            layers.append(nn.Sequential(nn.Linear(in_f, out_f), nn.Tanh()))
            self.weights.append(_random_linear(in_f, out_f))
        self.backbone = nn.Sequential(*layers)

        self.estimator = FastEstimator(self.backbone, shots=shots, seed=seed)

    def random_network(self, samples: int = 10) -> Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        return random_network(self.arch, samples)

    def feedforward(self, samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
        return feedforward(self.arch, self.weights, samples)

    def state_fidelity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        return state_fidelity(a, b)

    def fidelity_adjacency(
        self,
        states: Sequence[torch.Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        return self.estimator.evaluate(observables, parameter_sets)

__all__ = [
    "GraphQNNHybrid",
    "QCNNFeatureMap",
    "FastEstimator",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
