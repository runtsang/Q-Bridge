"""GraphQNNGen224: Classical graph neural network with attention and QCNN‑style layers.

The class builds a random fully connected backbone that can be
evaluated via a lightweight estimator.  It also exposes a
self‑attention wrapper and a simple convolutional block that
mimics the QCNN structure.

The implementation merges the ideas from the original GraphQNN,
FastBaseEstimator, SelfAttention and QCNN modules.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

import networkx as nx
import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


class _RandomLinear(nn.Module):
    """Utility linear layer with random weights (used for network generation)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        nn.init.normal_(self.linear.weight)
        nn.init.normal_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x))


class SelfAttentionLayer(nn.Module):
    """Classical self‑attention implemented with PyTorch.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the node embedding.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v


class QCNNBlock(nn.Module):
    """QCNN‑style convolution followed by pooling."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Linear(in_features, out_features), nn.Tanh())
        self.pool = nn.Sequential(nn.Linear(out_features, out_features // 2), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(x))


class GraphQNNGen224(nn.Module):
    """Hybrid classical‑quantum graph neural network.

    The network consists of:
    * a random fully‑connected backbone (random_network)
    * a self‑attention layer
    * a QCNN‑style block
    """

    def __init__(self, arch: Sequence[int], embed_dim: int = 4) -> None:
        super().__init__()
        self.arch = list(arch)
        self.layers: nn.ModuleList = nn.ModuleList(
            [_RandomLinear(in_f, out_f) for in_f, out_f in zip(arch[:-1], arch[1:])]
        )
        self.attention = SelfAttentionLayer(embed_dim)
        self.qcnn = QCNNBlock(arch[-1], arch[-1] // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.attention(x)
        x = self.qcnn(x)
        return x


def random_network(arch: Sequence[int], samples: int) -> Tuple[List[int], nn.Module, List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Generate a random network and training data.

    Returns
    -------
    arch, model, training_data, target_weight
    """
    model = GraphQNNGen224(arch)
    # create synthetic data by applying the last layer weight to random inputs
    target_weight = model.layers[-1].linear.weight
    training_data: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        inp = torch.randn(arch[0])
        tgt = target_weight @ inp
        training_data.append((inp, tgt))
    return list(arch), model, training_data, target_weight


def feedforward(
    model: nn.Module,
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    """Collect activations for each layer."""
    outputs: List[List[torch.Tensor]] = []
    for inp, _ in samples:
        activations: List[torch.Tensor] = [inp]
        current = inp
        for layer in model.layers:
            current = layer(current)
            activations.append(current)
        current = model.attention(current)
        activations.append(current)
        current = model.qcnn(current)
        activations.append(current)
        outputs.append(activations)
    return outputs


def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap for normalized vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[torch.Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class FastBaseEstimator:
    """Wrapper that evaluates a torch model on batches of parameters."""

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
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs).squeeze(0)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().item())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        return results


__all__ = [
    "GraphQNNGen224",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "FastBaseEstimator",
]
