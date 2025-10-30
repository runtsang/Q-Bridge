import torch
import torch.nn as nn
import networkx as nx
import itertools
import numpy as np
from typing import List, Tuple, Sequence, Iterable, Callable

Tensor = torch.Tensor

class GraphConv(nn.Module):
    """Classical graph convolution that aggregates neighbour states."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        aggregated = torch.matmul(adjacency, x)
        return torch.tanh(torch.matmul(aggregated, self.weight.t()))

class GraphPool(nn.Module):
    """Top‑k degree‑based pooling that reduces graph size."""
    def __init__(self, pool_size: int):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, x: Tensor, adjacency: Tensor) -> Tuple[Tensor, Tensor]:
        degrees = adjacency.sum(dim=1)
        _, idx = torch.topk(degrees, self.pool_size)
        return x[idx], adjacency[idx][:, idx]

class Autoencoder(nn.Module):
    """Fully‑connected autoencoder used as a latent bottleneck."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64),
                 dropout: float = 0.1):
        super().__init__()
        encoder_layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))

class GraphQNNHybrid(nn.Module):
    """Hybrid graph neural network that combines classical
    convolution‑pooling with a quantum‑style autoencoder."""
    def __init__(self,
                 arch: Sequence[int],
                 autoencoder_cfg: dict | None = None,
                 device: torch.device | None = None):
        super().__init__()
        self.arch = list(arch)
        self.conv_layers = nn.ModuleList(
            [GraphConv(arch[i], arch[i + 1]) for i in range(len(arch) - 1)]
        )
        self.pool_layers = nn.ModuleList(
            [GraphPool(pool_size=max(1, arch[i + 1] // 2)) for i in range(len(arch) - 1)]
        )
        self.autoencoder = Autoencoder(arch[-1], **(autoencoder_cfg or {}))
        self.device = device or torch.device("cpu")
        self.to(self.device)

    def forward(self, node_features: Tensor, adjacency: Tensor) -> Tensor:
        x = node_features.to(self.device)
        adj = adjacency.to(self.device)
        for conv, pool in zip(self.conv_layers, self.pool_layers):
            x = conv(x, adj)
            x, adj = pool(x, adj)
        latent = self.autoencoder.encode(x)
        return latent

    def fidelity_adjacency(self,
                           states: Sequence[Tensor],
                           threshold: float,
                           *,
                           secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = (a @ b).item() ** 2
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def kernel_matrix(self,
                      a: Sequence[Tensor],
                      b: Sequence[Tensor],
                      gamma: float = 1.0) -> np.ndarray:
        a_stack = torch.stack(a)
        b_stack = torch.stack(b)
        diff = a_stack.unsqueeze(1) - b_stack.unsqueeze(0)
        sq_norm = torch.sum(diff ** 2, dim=2)
        return torch.exp(-gamma * sq_norm).detach().cpu().numpy()

def random_network(arch: Sequence[int], samples: int):
    """Generate a random classical network matching the architecture."""
    weights = [torch.randn(out, in_) for in_, out in zip(arch[:-1], arch[1:])]
    target_weight = weights[-1]
    training_data = [(torch.randn(target_weight.size(1)), target_weight @ torch.randn(target_weight.size(1))) for _ in range(samples)]
    return list(arch), weights, training_data, target_weight
