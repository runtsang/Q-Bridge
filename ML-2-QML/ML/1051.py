import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Classical GCN backbone
# --------------------------------------------------------------------------- #
class ClassicalGCN(nn.Module):
    """
    Lightweight two‑layer GCN mirroring the original feed‑forward
    architecture.  The first layer transforms node features into a
    hidden representation; the second layer produces a linear output
    that matches the target weight matrix.
    """
    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden)
        self.conv2 = GCNConv(hidden, out_features)

    def forward(self, data: Data) -> Tensor:
        x, edge_index = data.x, data.edge_index
        h = F.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index)

# --------------------------------------------------------------------------- #
# Hybrid loss: fidelity between quantum and classical outputs
# --------------------------------------------------------------------------- #
def quantum_fidelity_loss(
    quantum_state: Tensor,
    classical_output: Tensor,
    eps: float = 1e-12,
) -> Tensor:
    """
    Compute a differentiable loss that keeps the quantum state
    (represented as a state vector) in‑class with the
    *target* (the classical output).  The loss is defined as
    1 - |<ψ_q|ψ_c>|², where |ψ_q> is the quantum state and |ψ_c>
    is the classical output projected onto the same Hilbert space.
    """
    # Normalize both states
    q_norm = quantum_state / (torch.norm(quantum_state) + eps)
    c_norm = classical_output / (torch.norm(classical_output) + eps)
    # Overlap magnitude
    overlap = torch.abs(torch.dot(q_norm.conj(), c_norm))
    return 1.0 - overlap ** 2

# --------------------------------------------------------------------------- #
# Random data generation utilities (classical)
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> list[tuple[Tensor, Tensor]]:
    dataset: list[tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """
    Generate a random hybrid network: a list of weight matrices for the
    classical GCN and a target weight matrix for the quantum layer.
    """
    weights: list[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

# --------------------------------------------------------------------------- #
# Fidelity‑based graph utilities
# --------------------------------------------------------------------------- #
def state_fidelity(a: Tensor, b: Tensor) -> float:
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
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

__all__ = [
    "ClassicalGCN",
    "quantum_fidelity_loss",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]
