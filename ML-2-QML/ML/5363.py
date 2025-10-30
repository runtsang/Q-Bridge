"""Combined classical graph neural network utilities and simple layer wrappers."""

import itertools
from typing import Iterable, Sequence, List, Tuple

import torch
import networkx as nx

Tensor = torch.Tensor

# --- Basic utilities -------------------------------------------------------

def _random_linear(in_features: int, out_features: int) -> Tensor:
    # Weight matrix with shape (out_features, in_features)
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic training data for a linear target."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Build a random classical MLP and generate a training set for its last layer."""
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
    """Forward pass through the MLP, returning activations per layer."""
    activations: List[List[Tensor]] = []
    for features, _ in samples:
        layer_inputs: List[Tensor] = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            layer_inputs.append(current)
        activations.append(layer_inputs)
    return activations

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Cosine similarity squared between two state vectors."""
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
    """Build a weighted graph where edges represent high fidelity between states."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --- Simple layer wrappers --------------------------------------------------

class FCL(torch.nn.Module):
    """A minimal fully‑connected layer with a single scalar output."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> torch.Tensor:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach()

class SamplerQNN(torch.nn.Module):
    """Soft‑max sampler inspired by the quantum SamplerQNN example."""
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.Tanh(),
            torch.nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.nn.functional.softmax(self.net(inputs), dim=-1)

# --- Combined hybrid class -------------------------------------------------

class GraphQNNHybrid:
    """
    Classical graph‑based neural network that can optionally emulate a quantum
    architecture by delegating the forward pass to a quantum backend.
    The classical implementation mirrors the original GraphQNN utilities
    while providing a clean interface for future quantum extensions.
    """
    def __init__(self, arch: Sequence[int], use_quantum: bool = False):
        self.arch = list(arch)
        self.use_quantum = use_quantum
        if use_quantum:
            raise RuntimeError(
                "Quantum mode is only available in the qml_code module."
            )
        self.weights = [ _random_linear(in_f, out_f) for in_f, out_f in zip(arch[:-1], arch[1:]) ]

    def feedforward(self, inputs: Tensor) -> List[Tensor]:
        """Return activations from each layer."""
        activations = [inputs]
        current = inputs
        for weight in self.weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        return activations

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        return state_fidelity(a, b)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    @classmethod
    def random_network(cls, arch: Sequence[int], samples: int):
        """Convenience constructor that returns a fully‑fledged network."""
        arch, weights, training_data, target_weight = random_network(arch, samples)
        return cls(arch, use_quantum=False), weights, training_data, target_weight

__all__ = [
    "GraphQNNHybrid",
    "FCL",
    "SamplerQNN",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
