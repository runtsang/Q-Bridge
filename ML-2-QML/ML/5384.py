"""
Classical hybrid estimator combining feedforward neural network, convolution filter,
self‑attention mechanism and graph‑based regularization.

The architecture is inspired by EstimatorQNN, GraphQNN, Conv and SelfAttention
reference pairs, but extends them with a hybrid regularization strategy that
leverages state fidelity between intermediate activations.
"""

import torch
from torch import nn
import numpy as np
import networkx as nx
import itertools

# ----- Core utilities -----
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: torch.Tensor, samples: int):
    dataset = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: list[int], samples: int):
    weights = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return qnn_arch, weights, training_data, target_weight

def feedforward(qnn_arch: list[int], weights: list[torch.Tensor], samples):
    activations = []
    for features, _ in samples:
        layer = features
        layerwise = [layer]
        for w in weights:
            layer = torch.tanh(w @ layer)
            layerwise.append(layer)
        activations.append(layerwise)
    return activations

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: list[torch.Tensor], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            g.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            g.add_edge(i, j, weight=secondary_weight)
    return g

# ----- Convolution filter -----
class ConvFilter(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[1,2,3])

# ----- Self‑attention -----
class ClassicalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.rotation = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = inputs @ self.rotation
        key = inputs @ self.entangle
        scores = torch.softmax(query @ key.t() / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs

# ----- Main Estimator -----
class EstimatorQNNGen275(nn.Module):
    def __init__(self, arch: list[int], conv_kernel: int = 2, conv_thresh: float = 0.0,
                 embed_dim: int = 4, graph_threshold: float = 0.8):
        super().__init__()
        self.arch = arch
        self.conv = ConvFilter(kernel_size=conv_kernel, threshold=conv_thresh)
        self.attention = ClassicalSelfAttention(embed_dim=embed_dim)
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(arch[:-1], arch[1:]):
            self.layers.append(nn.Linear(in_f, out_f))
        self.graph_threshold = graph_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(x)
        conv_vec = conv_out.view(conv_out.size(0), -1)
        attn_out = self.attention(conv_vec)
        out = attn_out
        activations = [attn_out]
        for layer in self.layers:
            out = torch.tanh(layer(out))
            activations.append(out)
        self.graph = fidelity_adjacency(activations, self.graph_threshold)
        return out

    def compute_graph(self, activations: list[torch.Tensor]) -> nx.Graph:
        return fidelity_adjacency(activations, self.graph_threshold)

__all__ = ["EstimatorQNNGen275"]
