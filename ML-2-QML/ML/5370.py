"""
SelfAttentionQNN – classical implementation.

Provides the same public API as the quantum version but uses
efficient PyTorch operations for attention, convolution, and graph
neural‑network utilities.
"""

from __future__ import annotations

import itertools
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from typing import Iterable, Sequence, Tuple, List

# --------------------------------------------------------------------------- #
# Classical self‑attention module
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    """
    Fast, drop‑in replacement for the quantum self‑attention block.
    Parameters
    ----------
    embed_dim : int
        Size of the embedding (query/key/value space).
    """
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> torch.Tensor:
        """
        Compute the self‑attention output.
        The rotation and entangle arrays are reshaped to match the linear
        layers and are used as learned weight matrices.
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = torch.matmul(
            q,
            torch.as_tensor(rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32),
        )
        k = torch.matmul(
            k,
            torch.as_tensor(entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32),
        )
        scores = torch.softmax(q @ k.transpose(-1, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v


# --------------------------------------------------------------------------- #
# Classical convolution filter (quantum analogue)
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """
    2‑D convolutional filter that mimics the behaviour of a quanvolution layer.
    Parameters
    ----------
    kernel_size : int
        Size of the 2‑D filter kernel.
    threshold : float
        Threshold used to binarise the output before averaging.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> float:
        """
        Return the mean sigmoid activation over a single‑channel image patch.
        """
        tensor = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()


# --------------------------------------------------------------------------- #
# Graph‑based feed‑forward utilities
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: torch.Tensor, samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    dataset: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """
    Build a random classical network that mirrors the QNN architecture.
    Returns
    arch, weights, training_data, target_weight
    """
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    """
    Run a forward pass over the dataset and collect activations per layer.
    """
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
    """
    Classical analogue of quantum state fidelity.
    """
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """
    Build a weighted graph from pairwise state fidelities.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# Hybrid factory that produces both classical and quantum back‑ends
# --------------------------------------------------------------------------- #
def build_hybrid_attention(
    embed_dim: int = 4,
    kernel_size: int = 2,
    graph_arch: Sequence[int] = (4, 8, 4),
    quantum: bool = False,
) -> Tuple[object, object, object]:
    """
    Produce a tuple (attention, conv, graph) where each element is a
    classical object or a quantum variant depending on the *quantum* flag.
    The quantum versions are lazy‑loaded to avoid unnecessary imports.
    """
    if quantum:
        from.quantum_self_attention import QuantumSelfAttention
        from.conv_quantum import QuantumConvFilter
        from.graph_qnn import QuantumGraphQNN
        attention = QuantumSelfAttention(n_qubits=embed_dim)
        conv = QuantumConvFilter(kernel_size=kernel_size)
        graph = QuantumGraphQNN(arch=graph_arch)
    else:
        attention = ClassicalSelfAttention(embed_dim=embed_dim)
        conv = ConvFilter(kernel_size=kernel_size)
        graph = random_network(graph_arch, 100)
    return attention, conv, graph


__all__ = [
    "ClassicalSelfAttention",
    "ConvFilter",
    "random_network",
    "feedforward",
    "fidelity_adjacency",
    "build_hybrid_attention",
]
