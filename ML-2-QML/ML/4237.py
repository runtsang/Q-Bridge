import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import itertools
from typing import Iterable, List, Sequence, Callable
from.FastBaseEstimator import FastBaseEstimator, FastEstimator

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

class HybridEstimator(FastEstimator):
    """
    Classical hybrid estimator that extends FastEstimator with graph‑based
    analysis of hidden activations.  It can evaluate a PyTorch model,
    inject Gaussian noise, and build weighted adjacency graphs from
    state fidelities.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)

    def feedforward(self,
                    qnn_arch: Sequence[int],
                    weights: Sequence[torch.Tensor],
                    samples: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
        stored: List[List[torch.Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for weight in weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    def fidelity_adjacency(self,
                           states: Sequence[torch.Tensor],
                           threshold: float,
                           *,
                           secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = torch.dot(a / (torch.norm(a)+1e-12),
                            b / (torch.norm(b)+1e-12)).item()**2
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def random_network(self,
                       qnn_arch: Sequence[int],
                       samples: int) -> tuple[Sequence[int], List[torch.Tensor], List[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        weights: List[torch.Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(torch.randn(out_f, in_f, dtype=torch.float32))
        target_weight = weights[-1]
        training_data = [(torch.randn(target_weight.size(1), dtype=torch.float32),
                          target_weight @ torch.randn(target_weight.size(1), dtype=torch.float32))
                         for _ in range(samples)]
        return qnn_arch, weights, training_data, target_weight

    def random_training_data(self,
                             weight: torch.Tensor,
                             samples: int) -> List[tuple[torch.Tensor, torch.Tensor]]:
        dataset: List[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

__all__ = ["HybridEstimator"]
