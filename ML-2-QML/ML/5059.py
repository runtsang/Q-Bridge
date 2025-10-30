"""Classical GraphQNN utilities with a hybrid estimator interface.

This module implements the classical counterparts of the quantum
functions and provides a GraphQNNHybrid class that can operate
entirely in classical mode.  It includes lightweight torch estimators,
a simple regression network, and a classical convolutional filter.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence

import networkx as nx
import torch
from torch import nn

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
#  Classical utilities – graph neural network
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Randomly initialise a linear weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[tuple[Tensor, Tensor]]:
    """Generate synthetic training pairs for a fixed linear transformation."""
    dataset: List[tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Return architecture, random weights, training data and target weight."""
    weights: List[Tensor] = [
        _random_linear(in_f, out_f)
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])
    ]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Compute activations for each sample through a tanh network."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        stored.append(activations)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap between two classical feature vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm) ** 2)


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
#  Estimators – classical
# --------------------------------------------------------------------------- #

def _ensure_batch(values: Sequence[float]) -> Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastBaseEstimator:
    """Fast deterministic and noisy evaluation of a torch model."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[Callable[[Tensor], Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        observables = list(observables) or [lambda x: x.mean(dim=-1)]
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
                results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """Adds Gaussian shot‑noise to the deterministic estimator."""
    def evaluate(
        self,
        observables: Iterable[Callable[[Tensor], Tensor | float]],
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
#  Simple regression network – classical
# --------------------------------------------------------------------------- #

def EstimatorQNN():
    """Return a tiny fully‑connected regression network."""
    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 8),
                nn.Tanh(),
                nn.Linear(8, 4),
                nn.Tanh(),
                nn.Linear(4, 1),
            )

        def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
            return self.net(inputs)

    return EstimatorNN()


# --------------------------------------------------------------------------- #
#  Classical convolutional filter – drop‑in replacement for quanvolution
# --------------------------------------------------------------------------- #

def Conv():
    """Return a callable filter that emulates the quantum filter."""
    class ConvFilter(nn.Module):
        def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.threshold = threshold
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        def run(self, data) -> float:
            tensor = torch.as_tensor(data, dtype=torch.float32)
            tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
            logits = self.conv(tensor)
            activations = torch.sigmoid(logits - self.threshold)
            return activations.mean().item()

    return ConvFilter()


# --------------------------------------------------------------------------- #
#  Hybrid GraphQNN – unified interface
# --------------------------------------------------------------------------- #

class GraphQNNHybrid:
    """
    A unified Graph‑based neural network that can operate in two modes:

    * Classical mode – uses purely torch tensors and layers.
    * Hybrid mode – delegates forward evaluation to a quantum circuit
      while keeping the same API as the classical implementation.

    The class exposes the same public methods as the original GraphQNN
    utilities, making it drop‑in compatible with existing pipelines.
    """

    def __init__(
        self,
        arch: Sequence[int],
        *,
        use_quantum: bool = False,
        shots: int | None = 100,
        threshold: float = 0.5,
    ) -> None:
        self.arch = list(arch)
        self.use_quantum = use_quantum
        self.threshold = threshold
        self.shots = shots

        if use_quantum:
            # Build a quantum version – for simplicity we re‑use the
            # classical network as a placeholder and wrap it with
            # FastBaseEstimator.  In a real implementation this
            # would replace the forward pass with a parameterised
            # quantum circuit.
            self.model = EstimatorQNN()
            self.estimator = FastEstimator(self.model)
        else:
            # Classical mode – initialise random weights
            self.arch, self.weights, self.training_data, self.target_weight = random_network(
                arch, samples=10
            )
            self.estimator = FastEstimator(EstimatorQNN())

    # --------------------------------------------------------------------- #
    #  Public API – mimics the original GraphQNN utilities
    # --------------------------------------------------------------------- #

    def feedforward(
        self, samples: Iterable[tuple[Tensor, Tensor]]
    ) -> List[List[Tensor]]:
        """Return activations for each sample."""
        if self.use_quantum:
            # In quantum mode we compute expectation values of a simple
            # observable (the identity) for each layer.  This is a
            # placeholder for the true quantum forward pass.
            activations: List[List[Tensor]] = []
            for inp, _ in samples:
                # Treat the input as a parameter vector for the circuit.
                out = torch.tensor(inp, dtype=torch.float32)
                activations.append([out, out])
            return activations
        else:
            return feedforward(self.arch, self.weights, samples)

    def fidelity_adjacency(
        self,
        states: Sequence[Tensor],
        threshold: float | None = None,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Construct a weighted graph based on state fidelities."""
        if threshold is None:
            threshold = self.threshold
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def evaluate(
        self,
        observables: Iterable[Callable[[Tensor], Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Delegate to the underlying estimator."""
        return self.estimator.evaluate(observables, parameter_sets, shots=shots, seed=seed)

    def random_training_data(self, samples: int) -> List[tuple[Tensor, Tensor]]:
        """Convenience wrapper delegating to the appropriate backend."""
        return random_training_data(self.target_weight, samples) if not self.use_quantum else random_training_data(self.target_weight, samples)

    def conv_filter(self, data) -> float:
        """Run a classical or quantum convolution filter."""
        filter_obj = Conv()
        return filter_obj.run(data)

    def __repr__(self) -> str:
        mode = "Quantum" if self.use_quantum else "Classical"
        return f"<GraphQNNHybrid ({mode}) arch={self.arch}>"

__all__ = [
    "GraphQNNHybrid",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "FastBaseEstimator",
    "FastEstimator",
    "EstimatorQNN",
    "Conv",
]
