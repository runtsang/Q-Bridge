import torch
from torch import nn
import networkx as nx
import itertools
from typing import Iterable, Sequence, Tuple, List
import numpy as np

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix with shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic linear data: target = weight @ features."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a toy network architecture and a matching synthetic dataset."""
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
    """Return activations for every layer of a classical feed‑forward network."""
    activations: List[List[Tensor]] = []
    for features, _ in samples:
        layerwise = [features]
        current = features
        for w in weights:
            current = torch.tanh(w @ current)
            layerwise.append(current)
        activations.append(layerwise)
    return activations

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap between two classical feature vectors."""
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
    """Build a graph where nodes are samples and edges reflect fidelity above a threshold."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class UnifiedEstimatorQNN(nn.Module):
    """Classical feed‑forward regressor with shared weight matrices.

    The network uses the same weight matrices that are exposed to the quantum
    counterpart.  The forward method returns the classical prediction only.
    """

    def __init__(self, arch: Sequence[int] = (2, 8, 4, 1)):
        super().__init__()
        self.arch = arch
        self.weight_matrices = nn.ParameterList(
            [nn.Parameter(_random_linear(in_f, out_f)) for in_f, out_f in zip(arch[:-1], arch[1:])]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the classical network."""
        current = x
        for w in self.weight_matrices:
            current = torch.tanh(w @ current)
        return current

    def get_weights_numpy(self) -> List[np.ndarray]:
        """Return the weight matrices as NumPy arrays for use in Qiskit."""
        return [w.detach().cpu().numpy() for w in self.weight_matrices]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(arch={self.arch})"
