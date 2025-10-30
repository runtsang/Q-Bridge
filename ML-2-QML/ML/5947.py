import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
from torch import nn
from torch.nn import functional as F

Tensor = torch.Tensor
Array = np.ndarray

# --------------------------------------------------------------------------- #
#  Core utilities – random network, training data and forward propagation
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a weight matrix with normalised columns."""
    w = torch.randn(out_features, in_features, dtype=torch.float32)
    return w / torch.norm(w, dim=1, keepdim=True).clamp_min(1e-12)

def random_training_data(
    weight: Tensor,
    samples: int,
    *,
    noise: float | None = None,
) -> List[Tuple[Tensor, Tensor]]:
    """Generate noisy linear targets for training."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        x = torch.randn(weight.size(1), dtype=torch.float32)
        y = weight @ x
        if noise is not None:
            y += torch.randn_like(y) * noise
        dataset.append((x, y))
    return dataset

def random_network(
    qnn_arch: Sequence[int],
    samples: int,
) -> Tuple[Sequence[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Create a random network and training data at one go."""
    weights = [_random_linear(a, b) for a, b in zip(qnn_arch[:-1], qnn_arch[1:])]
    target = weights[-1]
    training = random_training_data(target, samples)
    return qnn_arch, weights, training, target

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    batch: Tensor,
) -> Tensor:
    """Forward pass for a minibatch of samples."""
    out = batch
    for w in weights:
        out = torch.tanh(w @ out.t()).t()
    return out

# --------------------------------------------------------------------------- #
#  Fidelity helpers – state overlap and graph construction
# --------------------------------------------------------------------------- #

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap between two unit‑norm tensors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
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
#  Graph‑regularised loss – Laplacian penalty
# --------------------------------------------------------------------------- #

class GraphRegularizer:
    """Regularise a batch of embeddings using the graph Laplacian."""

    def __init__(self, adjacency: nx.Graph, lambda_: float = 1.0):
        self.adj = adjacency
        self.lambda_ = lambda_
        # Laplacian matrix
        self.L = nx.laplacian_matrix(adjacency).toarray().astype(np.float32)

    def loss(self, embeddings: Tensor) -> Tensor:
        """Return Laplacian regularisation term."""
        L = torch.from_numpy(self.L).to(embeddings.device)
        return self.lambda_ * torch.trace(embeddings @ L @ embeddings.t())

# --------------------------------------------------------------------------- #
#  Simple minibatch training loop
# --------------------------------------------------------------------------- #

def train_network(
    qnn_arch: Sequence[int],
    weights: List[Tensor],
    training: List[Tuple[Tensor, Tensor]],
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    graph: nx.Graph | None = None,
    lambda_: float = 0.0,
) -> List[Tensor]:
    """Simple minibatch training loop with optional graph regularisation."""
    device = torch.device("cpu")
    params = [w.clone().requires_grad_(True) for w in weights]
    optimizer = torch.optim.Adam(params, lr=lr)

    if graph is not None:
        reg = GraphRegularizer(graph, lambda_)
    else:
        reg = None

    for epoch in range(epochs):
        perm = torch.randperm(len(training))
        for i in range(0, len(training), batch_size):
            batch_idx = perm[i : i + batch_size]
            batch_x = torch.stack([training[j][0] for j in batch_idx])
            batch_y = torch.stack([training[j][1] for j in batch_idx])

            optimizer.zero_grad()
            pred = feedforward(qnn_arch, params, batch_x)
            loss = F.mse_loss(pred, batch_y)
            if reg is not None:
                loss += reg.loss(pred)
            loss.backward()
            optimizer.step()

    return [p.detach() for p in params]

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "train_network",
    "GraphRegularizer",
]
