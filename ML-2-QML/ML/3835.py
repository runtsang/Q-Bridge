import torch
import torch.nn as nn
import networkx as nx
import itertools
from typing import List, Tuple, Sequence, Iterable

Tensor = torch.Tensor

def _random_lin(in_f: int, out_f: int) -> Tensor:
    return torch.randn(out_f, in_f, dtype=torch.float32)

def random_training_data(weight: Tensor, n_samples: int):
    data = []
    for _ in range(n_samples):
        x = torch.randn(weight.size(1), dtype=torch.float32)
        y = weight @ x
        data.append((x, y))
    return data

def random_classical_network(arch: Sequence[int], n_samples: int):
    weights = [_random_lin(a,b) for a,b in zip(arch[:-1], arch[1:])]
    target = weights[-1]
    training = random_training_data(target, n_samples)
    return list(arch), weights, training, target

def feedforward(arch: Sequence[int], weights: Sequence[Tensor], samples: Iterable[Tuple[Tensor, Tensor]]):
    activations = []
    for x,_ in samples:
        act = [x]
        cur = x
        for w in weights:
            cur = torch.tanh(w @ cur)
            act.append(cur)
        activations.append(act)
    return activations

def state_fidelity(a: Tensor, b: Tensor) -> float:
    an = a / (torch.norm(a)+1e-12)
    bn = b / (torch.norm(b)+1e-12)
    return float(torch.dot(an,bn).item()**2)

def fidelity_adjacency(states: Sequence[Tensor], thresh: float,
                       *, secondary: float | None=None, secondary_w: float=0.5):
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i,sa),(j,sb) in itertools.combinations(enumerate(states),2):
        fid = state_fidelity(sa,sb)
        if fid>=thresh:
            G.add_edge(i,j,weight=1.0)
        elif secondary is not None and fid>=secondary:
            G.add_edge(i,j,weight=secondary_w)
    return G

class GraphQNN__gen133(nn.Module):
    """Hybrid classical graph neural network with fidelity‑based adjacency."""
    def __init__(self, arch: Sequence[int], weights: Sequence[Tensor]):
        super().__init__()
        self.arch = arch
        self.weights = weights

    def forward(self, x: Tensor):
        act = [x]
        cur = x
        for w in self.weights:
            cur = torch.tanh(w @ cur)
            act.append(cur)
        return act[-1]

    @staticmethod
    def random_instance(arch: Sequence[int], samples: int):
        arch, weights, training, target = random_classical_network(arch, samples)
        return GraphQNN__gen133(arch, weights), training

__all__ = ['GraphQNN__gen133','feedforward','fidelity_adjacency',
           'random_classical_network','random_training_data','state_fidelity']
