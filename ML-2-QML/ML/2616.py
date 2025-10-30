import torch
import networkx as nx
import itertools
from typing import List, Tuple, Sequence, Iterable, Optional

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(target: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic (input, target) pairs for a linear mapping."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        inp = torch.randn(target.size(1), dtype=torch.float32)
        out = target @ inp
        dataset.append((inp, out))
    return dataset

def random_network(arch: Sequence[int], samples: int):
    """Build a random feed‑forward network and corresponding training set."""
    weights: List[Tensor] = [_random_linear(in_f, out_f) for in_f, out_f in zip(arch[:-1], arch[1:])]
    target_weight = weights[-1]
    training = random_training_data(target_weight, samples)
    return list(arch), weights, training, target_weight

def feedforward(arch: Sequence[int], weights: Sequence[Tensor], data: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
    """Collect activations for each sample through the network."""
    activations: List[List[Tensor]] = []
    for inp, _ in data:
        layer_out = inp
        layer_vals = [inp]
        for w in weights:
            layer_out = torch.tanh(w @ layer_out)
            layer_vals.append(layer_out)
        activations.append(layer_vals)
    return activations

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap of two normalized vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[Tensor], threshold: float,
                       *, secondary: Optional[float] = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph from pairwise fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

class QuanvolutionFilter(torch.nn.Module):
    """A 2×2 patch extractor followed by a small 2‑D convolution."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        feats = self.conv(x)
        return feats.view(x.size(0), -1)

class QuanvolutionClassifier(torch.nn.Module):
    """Classifier that chains a quanvolution filter with a linear head."""
    def __init__(self) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter()
        self.head = torch.nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        feats = self.filter(x)
        logits = self.head(feats)
        return torch.nn.functional.log_softmax(logits, dim=-1)

class GraphQNNHybridML:
    """Encapsulates a classical graph‑based neural net with optional quanvolution head."""
    def __init__(self, arch: Sequence[int]) -> None:
        self.arch = list(arch)
        self.weights = [_random_linear(in_f, out_f) for in_f, out_f in zip(arch[:-1], arch[1:])]

    def forward(self, x: Tensor) -> List[Tensor]:
        activations = [x]
        for w in self.weights:
            x = torch.tanh(w @ x)
            activations.append(x)
        return activations

    def fidelity_graph(self, states: Sequence[Tensor], threshold: float,
                       *, secondary: Optional[float] = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary,
                                   secondary_weight=secondary_weight)

    def train_random(self, samples: int) -> None:
        _, _, self.training, _ = random_network(self.arch, samples)

__all__ = [
    "GraphQNNHybridML",
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
