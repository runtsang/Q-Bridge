import torch
import torch.nn as nn
import networkx as nx
import itertools
from typing import List, Tuple, Sequence, Iterable

Tensor = torch.Tensor

class QFCModel(nn.Module):
    """Convolutional encoder producing a 4‑node feature vector."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 7 * 7, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: Tensor) -> Tensor:
        feat = self.encoder(x)
        flat = self.flatten(feat)
        return self.norm(self.fc(flat))

def _init_linear(in_f: int, out_f: int) -> Tensor:
    return torch.randn(out_f, in_f, dtype=torch.float32)

def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[Sequence[int], QFCModel, List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    encoder = QFCModel()
    weights = [_init_linear(i, o) for i, o in zip(qnn_arch[:-1], qnn_arch[1:])]
    target = weights[-1]
    training_data = random_training_data(target, samples)
    return qnn_arch, encoder, weights, training_data, target

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    data = []
    for _ in range(samples):
        inp = torch.randn(weight.size(1))
        out = weight @ inp
        data.append((inp, out))
    return data

def feedforward(qnn_arch: Sequence[int], encoder: QFCModel, weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
    activations_per_sample = []
    for inp, _ in samples:
        node_feats = encoder(inp.view(1,1,28,28)).squeeze(0)
        activations = [node_feats]
        cur = node_feats
        for w in weights:
            cur = torch.tanh(w @ cur)
            activations.append(cur)
        activations_per_sample.append(activations)
    return activations_per_sample

def state_fidelity(a: Tensor, b: Tensor) -> float:
    a_n = a / (a.norm() + 1e-12)
    b_n = b / (b.norm() + 1e-12)
    return float((a_n @ b_n).abs().item() ** 2)

def fidelity_adjacency(states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            g.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            g.add_edge(i, j, weight=secondary_weight)
    return g

class GraphQNNGen(nn.Module):
    """Hybrid classical graph neural network that uses a CNN encoder and linear layers."""
    def __init__(self, arch: Sequence[int]):
        super().__init__()
        self.arch = arch
        self.encoder = QFCModel()
        self.layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(arch[:-1], arch[1:])])

    def forward(self, x: Tensor) -> Tensor:
        node_feats = self.encoder(x)
        out = node_feats
        for layer in self.layers:
            out = torch.tanh(layer(out))
        return out

__all__ = [
    "QFCModel",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "GraphQNNGen",
]
