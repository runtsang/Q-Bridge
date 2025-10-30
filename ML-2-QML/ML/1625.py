import numpy as np
import torch
import networkx as nx
import itertools
from typing import List, Tuple, Sequence, Iterable

Tensor = torch.Tensor

class GraphQNNGen192:
    """Classical graph‑based neural network utilities with reproducible random generation and gradient training."""

    @staticmethod
    def _random_linear(in_features: int, out_features: int, rng: np.random.Generator) -> Tensor:
        return torch.tensor(rng.normal(size=(out_features, in_features)), dtype=torch.float32)

    @staticmethod
    def random_training_data(weight: Tensor, samples: int, rng: np.random.Generator) -> List[Tuple[Tensor, Tensor]]:
        dataset: List[Tuple[Tensor, Tensor]] = []
        for _ in range(samples):
            features = torch.tensor(rng.normal(size=(weight.shape[1],)), dtype=torch.float32)
            target = weight @ features
            dataset.append((features, target))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int, rng: np.random.Generator) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
        weights: List[Tensor] = []
        for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
            weights.append(GraphQNNGen192._random_linear(in_f, out_f, rng))
        target_weight = weights[-1]
        training_data = GraphQNNGen192.random_training_data(target_weight, samples, rng)
        return list(qnn_arch), weights, training_data, target_weight

    @staticmethod
    def feedforward(qnn_arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        stored: List[List[Tensor]] = []
        for features, _ in samples:
            activations: List[Tensor] = [features]
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
            fid = GraphQNNGen192.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def train(weights: List[Tensor], training_data: List[Tuple[Tensor, Tensor]], lr: float = 0.01, epochs: int = 100) -> List[Tensor]:
        opt = torch.optim.SGD([w for w in weights], lr=lr)
        loss_fn = torch.nn.MSELoss()
        for _ in range(epochs):
            opt.zero_grad()
            loss = 0.0
            for x, y in training_data:
                out = x
                for w in weights:
                    out = torch.tanh(w @ out)
                loss += loss_fn(out, y)
            loss /= len(training_data)
            loss.backward()
            opt.step()
        return weights

__all__ = ["GraphQNNGen192"]
