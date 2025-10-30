import torch
import torch.nn as nn
import itertools
import networkx as nx
from typing import Iterable, Tuple, Sequence, List

Tensor = torch.Tensor

class QuantumClassifierModel(nn.Module):
    """Hybrid classical classifier with graph‑based fidelity regularisation."""
    def __init__(self, num_features: int, hidden_sizes: Sequence[int], depth: int):
        super().__init__()
        self.network, self.encoding, self.weight_sizes, self.observables = \
            self.build_classifier_circuit(num_features, hidden_sizes, depth)

    @staticmethod
    def _random_linear(in_features: int, out_features: int) -> Tensor:
        return torch.randn(out_features, in_features, dtype=torch.float32)

    @staticmethod
    def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.randn(weight.size(1), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        weights: List[Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(QuantumClassifierModel._random_linear(in_f, out_f))
        target_weight = weights[-1]
        training_data = QuantumClassifierModel.random_training_data(target_weight, samples)
        return list(qnn_arch), weights, training_data, target_weight

    @staticmethod
    def feedforward(qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations = [features]
            current = features
            for weight in weights:
                current = torch.tanh(weight @ current)
                activations.append(current)
            stored.append(activations)
        return stored

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = QuantumClassifierModel.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def build_classifier_circuit(num_features: int, hidden_sizes: Sequence[int], depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
        """Constructs an MLP with optional graph layer derived from hidden activations."""
        layers = []
        in_dim = num_features
        encoding = list(range(num_features))
        weight_sizes = []
        for size in hidden_sizes:
            linear = nn.Linear(in_dim, size)
            layers.append(linear)
            layers.append(nn.Tanh())
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = size
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())
        net = nn.Sequential(*layers)
        observables = list(range(2))
        return net, encoding, weight_sizes, observables

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)
