import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import itertools
from typing import Iterable, List, Sequence, Tuple, Optional

Tensor = torch.Tensor

def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    weights = [_random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])]
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

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

def state_fidelity(a: Tensor, b: Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class UnifiedSelfAttentionGraphQNN:
    def __init__(self, embed_dim: int, num_heads: int = 4, use_quantum_mask: bool = False):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_quantum_mask = use_quantum_mask
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        if use_quantum_mask:
            try:
                from.quantum_module import QuantumSelfAttention
                self.q_attention = QuantumSelfAttention(n_qubits=embed_dim)
            except Exception:
                self.q_attention = None
        else:
            self.q_attention = None

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if self.use_quantum_mask and self.q_attention is not None and mask is None:
            mask = torch.zeros(x.shape[1], dtype=torch.bool)
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output
