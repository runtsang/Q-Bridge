import itertools
from typing import Iterable, Sequence, Tuple, List, Callable, Optional

import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim

Tensor = torch.Tensor


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a weight matrix initialized from a standard normal distribution."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate a synthetic dataset where the target is a linear transformation of the input."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random feed‑forward network and corresponding training data."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight


def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Perform a forward pass through the network and store intermediate activations."""
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
    """Return the squared overlap of two classical vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: Optional[float] = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a graph where edges reflect high‑fidelity similarities between states."""
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
#  New: Graph‑regularized loss and early‑stopping
# --------------------------------------------------------------------------- #
def _graph_regularizer(
    activations: List[List[Tensor]],
    graph: nx.Graph,
    gamma: float,
) -> Tensor:
    """Compute a graph‑based regularization term that penalises
    differences between node‑activations across edges in the graph."""
    loss = torch.tensor(0.0, device=activations[0][0].device)
    for edge in graph.edges:
        i, j = edge
        loss += gamma * torch.norm(
            *[t[i] - t[j] for t in activations],
            p=2
        )
    return loss


def _early_stopping(
    loss_history: List[float], patience: int, min_delta: float = 1e-4
) -> bool:
    """Return True if training should stop early."""
    if len(loss_history) < 2 * patience:
        return False
    recent = loss_history[-patience:]
    return all(
        all(abs(l - l_prev) < min_delta for l_prev, l in zip(loss_history[-patience:], loss_history[-patience-1:-1]))
        for _ in range(1))
