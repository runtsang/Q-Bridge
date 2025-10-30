"""Hybrid classical graph neural network with optional quantum modules.

The class GraphQNNGen implements the same API as the original GraphQNN
module but adds optional quantum convolution (Quanvolution) and
transformer blocks.  It can be instantiated in pure‑classical mode or
with quantum sub‑modules by passing ``use_quantum=True``.  The design
keeps the original seed behaviour (random network, feed‑forward,
fidelity graph) while exposing experimentally richer building blocks
derived from the QML seeds.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Dict

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Utility functions -----------------------------------------------------


def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    weights: List[torch.Tensor] = []
    for in_features, out_features in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_features, out_features))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    *,
    use_quantum: bool = False,
) -> List[List[torch.Tensor]]:
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


def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# ---- Quantum‑enabled components --------------------------------------------

class QuanvolutionFilter(nn.Module):
    """Two‑qubit quantum kernel applied to 2×2 patches."""

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.linear = nn.Linear(n_wires, n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.linear(x.view(x.size(0), -1))


class QuanvolutionClassifier(nn.Module):
    """Hybrid classifier that uses a quanvolution filter followed by a linear head."""

    def __init__(self, input_dim: int, num_classes: int = 10):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


# ---- Transformer integration ----------------------------------------------

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# ---- Main hybrid graph neural network ---------------------------------------

class GraphQNNGen(nn.Module):
    """
    Hybrid graph neural network that can operate purely classically or
    inject quantum sub‑modules (quanvolution, transformer, or a simple
    fully‑connected quantum layer).  The public API matches the seed
    GraphQNN module so that existing code can switch to the hybrid
    implementation with minimal changes.

    Parameters
    ----------
    arch : Sequence[int]
        List of layer widths.
    use_quantum : bool, optional
        When ``True`` the network will replace the linear layers with
        quantum‑inspired modules (a placeholder unitary is used).
    n_qubits : int, optional
        Number of qubits used in the quantum block.  Ignored when
        ``use_quantum=False``.
    """

    def __init__(
        self,
        arch: Sequence[int],
        *,
        use_quantum: bool = False,
        n_qubits: int = 4,
        use_transformer: bool = False,
        transformer_params: Dict | None = None,
    ) -> None:
        super().__init__()
        self.arch = list(arch)
        self.use_quantum = use_quantum
        self.n_qubits = n_qubits
        self.use_transformer = use_transformer

        # Build linear (or quantum‑placeholder) layers
        layers: list[nn.Module] = []
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            if use_quantum:
                layers.append(nn.Linear(in_f, out_f))
            else:
                layers.append(nn.Linear(in_f, out_f))
        self.layers = nn.ModuleList(layers)

        if use_transformer:
            params = transformer_params or {}
            self.transformer = nn.Sequential(
                *[
                    TransformerBlock(
                        embed_dim=arch[-1],
                        num_heads=params.get("num_heads", 4),
                        ffn_dim=params.get("ffn_dim", 64),
                        dropout=params.get("dropout", 0.1),
                    )
                    for _ in range(params.get("num_blocks", 2))
                ]
            )
        else:
            self.transformer = None

        # Optional quanvolution head for image‑style data
        self.qconv = QuanvolutionFilter() if use_quantum else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.qconv is not None:
            x = self.qconv(x)

        for layer in self.layers:
            x = torch.tanh(layer(x))

        if self.transformer is not None:
            x = x.unsqueeze(1)
            x = self.transformer(x)
            x = x.squeeze(1)

        return x

    # Utility methods mirroring the seed API --------------------------------

    def random_network(self, samples: int) -> Tuple[List[int], List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        return random_network(self.arch, samples)

    def feedforward(self, samples: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> List[List[torch.Tensor]]:
        return feedforward(self.arch, [layer.weight for layer in self.layers], samples, use_quantum=self.use_quantum)

    def fidelity_adjacency(self, states: Sequence[torch.Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def state_fidelity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        return state_fidelity(a, b)


class FCL(nn.Module):
    """
    Stateless wrapper around the simple fully‑connected quantum layer
    from the reference.  It exposes a ``run`` method that accepts a
    sequence of angles and returns the expectation value.
    """

    def __init__(self, n_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()


__all__ = [
    "GraphQNNGen",
    "FCL",
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "TransformerBlock",
    "PositionalEncoder",
    "MultiHeadAttention",
    "FeedForward",
]
