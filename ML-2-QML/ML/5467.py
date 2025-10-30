import torch
from torch import nn
import networkx as nx
import itertools
import numpy as np
from typing import Sequence, Iterable, List, Callable
from FastBaseEstimator import FastBaseEstimator, FastEstimator

# ---------------- GraphQNN utilities ---------------------------------
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: torch.Tensor, samples: int) -> List[tuple[torch.Tensor, torch.Tensor]]:
    dataset = []
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
    return qnn_arch, weights, training_data, target_weight

def feedforward(qnn_arch: Sequence[int], weights: Sequence[torch.Tensor],
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

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[torch.Tensor], threshold: float,
                       *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# ---------------- QCNN model -----------------------------------------
class QCNNModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

def QCNN() -> QCNNModel:
    return QCNNModel()

# ---------------- Conv filter ----------------------------------------
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

# ---------------- Hybrid estimator -----------------------------------
class HybridEstimator(FastBaseEstimator):
    """
    Combines deterministic FastBaseEstimator evaluation with graph‑based
    state fidelity analysis and a classical QCNN predictor.
    """
    def __init__(self, model: nn.Module, graph_arch: Sequence[int]) -> None:
        super().__init__(model)
        self.graph_arch = graph_arch
        self.qcnn_model = QCNN()

    def graph_from_params(self, params: Sequence[float], threshold: float = 0.8) -> nx.Graph:
        """
        Build a fidelity graph from a single parameter vector.
        """
        arch, weights, _, _ = random_network(self.graph_arch, samples=1)
        sample = [(torch.tensor(params, dtype=torch.float32), None)]
        activations = feedforward(self.graph_arch, weights, sample)
        states = [act[-1] for act in activations]
        return fidelity_adjacency(states, threshold)

    def qcnn_predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward the inputs through the embedded QCNN model.
        """
        return self.qcnn_model(inputs)

    def evaluate_with_graph(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        threshold: float = 0.8,
    ) -> tuple[List[List[float]], List[nx.Graph]]:
        """
        Evaluate observables and return graph adjacency matrices
        for each parameter set.
        """
        results = super().evaluate(observables, parameter_sets)
        graphs = [self.graph_from_params(params, threshold) for params in parameter_sets]
        return results, graphs

__all__ = ["HybridEstimator", "ConvFilter"]
