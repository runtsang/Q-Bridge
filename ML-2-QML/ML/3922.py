"""ml_code for UnifiedEstimatorQNN"""

import numpy as np
import torch
from torch import nn, Tensor
import networkx as nx
import itertools
from typing import List, Tuple, Sequence, Iterable

# Embedding network
class EmbeddingNet(nn.Module):
    """Tiny feed‑forward network that maps 2‑D inputs to a 1‑D embedding."""
    def __init__(self, in_features: int = 2, hidden: int = 8, out_features: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 8),
            nn.Tanh(),
            nn.Linear(8, out_features),
        )
        self.embedding = nn.Linear(out_features, 1, bias=False)  # 1‑D embedding

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        return self.embedding(x)

# Classical graph‑QNN utilities
def _random_linear(in_features: int, out_features: int) -> np.ndarray:
    """Random weight matrix for a linear layer."""
    return np.random.randn(out_features, in_features).astype(np.float32)

def random_training_data(weight: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate synthetic training pairs for a linear mapping."""
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(samples):
        features = np.random.randn(weight.shape[1]).astype(np.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random classical feed‑forward network and a synthetic training set."""
    weights: List[np.ndarray] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(qnn_arch: Sequence[int],
                weights: Sequence[np.ndarray],
                samples: Iterable[Tuple[np.ndarray, np.ndarray]]) -> List[List[np.ndarray]]:
    """Forward pass through a classical network."""
    stored: List[List[np.ndarray]] = []
    for features, _ in samples:
        activations: List[np.ndarray] = [features]
        current = features
        for weight in weights:
            current = np.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Squared-overlap fidelity for two real vectors."""
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_norm, b_norm) ** 2)

def fidelity_adjacency(states: Sequence[np.ndarray],
                       threshold: float,
                       *,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Construct an adjacency graph weighted by state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class ClassicalGraphQNN:
    """Classical analogue of the graph‑QNN."""
    def __init__(self,
                 architecture: Sequence[int],
                 weights: List[np.ndarray],
                 threshold: float = 0.8,
                 secondary: float | None = None):
        self.architecture = architecture
        self.weights = weights
        self.threshold = threshold
        self.secondary = secondary

    def forward(self, x: np.ndarray) -> float:
        """Forward propagation and weighted aggregation."""
        activations = [x]
        current = x
        for weight in self.weights:
            current = np.tanh(weight @ current)
            activations.append(current)
        final_acts = activations[-1]
        # Build adjacency graph from final activations
        graph = fidelity_adjacency([final_acts], self.threshold,
                                   secondary=self.secondary)
        # Return scalar prediction
        return float(final_acts[0]) if final_acts.size == 1 else float(np.mean(final_acts))

class UnifiedEstimatorQNN:
    """Hybrid estimator combining classical embedding and graph‑based network."""
    def __init__(self, architecture: Sequence[int] = (1, 2, 1)):
        self.embedding = EmbeddingNet()
        arch, weights, _, _ = random_network(architecture, samples=10)
        self.graph_qnn = ClassicalGraphQNN(architecture, weights)

    def predict(self, x: Tensor) -> Tensor:
        """Predict the target value for a batch of inputs."""
        embed = self.embedding(x)  # (batch, 1)
        outputs = []
        for vec in embed.detach().cpu().numpy():
            out = self.graph_qnn.forward(vec)
            outputs.append(out)
        return torch.tensor(outputs, dtype=torch.float32)

__all__ = ["EmbeddingNet", "ClassicalGraphQNN", "UnifiedEstimatorQNN"]
