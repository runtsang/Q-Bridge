import torch
import torch.nn as nn
import networkx as nx
import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Callable

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

class FastBaseEstimator:
    """Lightweight estimator utilities implemented with PyTorch modules."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

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
                results.append(row)
        return results

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class GraphQNNGen(nn.Module):
    """Classical graph neural network with estimator interface."""
    def __init__(self, architecture: Sequence[int]) -> None:
        super().__init__()
        self.arch = list(architecture)
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            self.layers.append(nn.Linear(in_f, out_f))
        self.estimator = FastBaseEstimator(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x

    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
        return torch.randn(out_features, in_features, dtype=torch.float32)

    @staticmethod
    def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(arch: Sequence[int], samples: int):
        weights: List[torch.Tensor] = []
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            weights.append(GraphQNNGen._random_linear(in_f, out_f))
        target_weight = weights[-1]
        training_data = GraphQNNGen.random_training_data(target_weight, samples)
        return list(arch), weights, training_data, target_weight

    def feedforward(
        self,
        samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[List[torch.Tensor]]:
        stored: List[List[torch.Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for layer in self.layers:
                current = torch.tanh(layer(current))
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
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
            fid = GraphQNNGen.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # Estimator API -------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        return self.estimator.evaluate(observables, parameter_sets)

__all__ = [
    "GraphQNNGen",
    "FastBaseEstimator",
]
