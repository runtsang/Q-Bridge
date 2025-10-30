"""FastBaseEstimator (classical) – lightweight PyTorch evaluation with
graph and convolution support.

The module merges the original FastBaseEstimator, the Conv filter,
and the GraphQNN utilities.  It offers deterministic and noisy
inference, a parameter‑aware convolution filter, and functions to
generate random networks, datasets, and fidelity‑based graphs.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch import nn

import networkx as nx

# --------------------------------------------------------------------------- #
# Helper: batch conversion
# --------------------------------------------------------------------------- #
def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert an iterable of floats to a 2‑D tensor of shape (B, D)."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

# --------------------------------------------------------------------------- #
# FastBaseEstimator
# --------------------------------------------------------------------------- #
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

class FastBaseEstimator:
    """Evaluate a PyTorch `nn.Module` for batches of parameters.

    Parameters
    ----------
    model : nn.Module
        The network to evaluate.  The module must be in eval mode.
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Return a list of rows, each containing the observable values for a
        parameter set.  If no observables are supplied, the mean of the last
        layer is returned.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    val = observable(outputs)
                    scalar = float(val.mean().cpu()) if isinstance(val, torch.Tensor) else float(val)
                    row.append(scalar)
                results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Same as `FastBaseEstimator` but adds Gaussian shot noise."""
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

# --------------------------------------------------------------------------- #
# Conv filter (classical)
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """Simple 2‑D convolution filter that emulates the quantum quanvolution."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data) -> float:
        """Return the mean sigmoid activation after the convolution."""
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

def Conv() -> ConvFilter:
    """Convenience factory returning a default `ConvFilter`."""
    return ConvFilter()

# --------------------------------------------------------------------------- #
# GraphQNN utilities
# --------------------------------------------------------------------------- #
Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic (x, y) pairs for a linear target."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random feed‑forward network along with training data."""
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
    """Forward propagate all samples through the network."""
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
    """Squared overlap between two classical vectors."""
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
    """Build a weighted graph from pairwise fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

__all__ = [
    "FastBaseEstimator",
    "FastEstimator",
    "Conv",
    "ConvFilter",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
