"""GraphQNN: Classical GNN with hybrid training utilities.

Improvements over the seed:
* End‑to‑end SGD training loop.
* Mini‑batch and epoch handling.
* Optional graph‑based regularisation via state fidelities.
* Checkpoint helpers for arch/weights persistence.

All code uses PyTorch and NetworkX; it runs on CPU only.
"""

from __future__ import annotations

import itertools
import random
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
from torch import Tensor

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "train_step",
    "train_loop",
    "evaluate",
    "save_checkpoint",
    "load_checkpoint",
]


# --------------------------------------------------------------------- #
#  Core feed‑forward + utilities
# --------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a weight tensor of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32, requires_grad=True)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic labelled pairs (x, y = W·x)."""
    data: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        x = torch.randn(weight.size(1), dtype=torch.float32)
        y = weight @ x
        data.append((x, y))
    return data


def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[Tensor], List[Tuple[Tensor, Tensor]], Tensor]:
    """Build a random feed‑forward network and create training data for its last layer."""
    weights: List[Tensor] = [
        _random_linear(in_f, out_f) for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])
    ]
    target = weights[-1]
    training = random_training_data(target, samples)
    return list(qnn_arch), weights, training, target


# --------------------------------------------------------------------- #
#  Forward pass
# --------------------------------------------------------------------- #
def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Return a list of layer‑wise activations for each sample."""
    outputs: List[List[Tensor]] = []
    for x, _ in samples:
        activations = [x]
        current = x
        for w in weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        outputs.append(activations)
    return outputs


# --------------------------------------------------------------------- #
#  Fidelity utilities
# --------------------------------------------------------------------- #
def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap between two normalised vectors."""
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
    """Create a weighted adjacency graph from state fidelities.

    Edges with fidelity >= threshold receive weight 1.
    When secondary is provided, fidelities between secondary and threshold
    are added with secondary_weight.
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


# --------------------------------------------------------------------- #
#  Training helpers
# --------------------------------------------------------------------- #
def train_step(
    weights: List[Tensor],
    batch: List[Tuple[Tensor, Tensor]],
    lr: float = 1e-3,
) -> float:
    """Perform one SGD step on the given batch and return the loss."""
    loss = 0.0
    for x, y in batch:
        activations = [x]
        current = x
        for w in weights:
            current = torch.tanh(w @ current)
            activations.append(current)
        pred = activations[-1]
        loss += torch.mean((pred - y) ** 2)
    loss /= len(batch)
    loss.backward()
    with torch.no_grad():
        for w in weights:
            w -= lr * w.grad
            w.grad.zero_()
    return loss.item()


def train_loop(
    arch: List[int],
    weights: List[Tensor],
    train_dataset: List[Tuple[Tensor, Tensor]],
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    verbose: bool = True,
) -> List[float]:
    """Run a full training loop, returning the loss history."""
    history: List[float] = []
    for epoch in range(1, epochs + 1):
        random.shuffle(train_dataset)
        epoch_loss = 0.0
        batches = [
            train_dataset[i : i + batch_size]
            for i in range(0, len(train_dataset), batch_size)
        ]
        for batch in batches:
            loss = train_step(weights, batch, lr)
            epoch_loss += loss
        epoch_loss /= len(batches)
        history.append(epoch_loss)
        if verbose:
            print(f"Epoch {epoch:3d}/{epochs}  Loss: {epoch_loss:.6f}")
    return history


def evaluate(weights: List[Tensor], dataset: List[Tuple[Tensor, Tensor]]) -> float:
    """Compute mean squared error on the given dataset."""
    mse = 0.0
    for x, y in dataset:
        current = x
        for w in weights:
            current = torch.tanh(w @ current)
        pred = current
        mse += torch.mean((pred - y) ** 2).item()
    return mse / len(dataset)


def save_checkpoint(path: str, arch: List[int], weights: List[Tensor]) -> None:
    """Persist architecture and weights to disk."""
    torch.save(
        {"arch": arch, "weights": [w.detach().clone() for w in weights]}, path
    )


def load_checkpoint(path: str) -> Tuple[List[int], List[Tensor]]:
    """Load architecture and weights from disk."""
    ckpt = torch.load(path)
    arch = ckpt["arch"]
    weights = [w.clone().requires_grad_(True) for w in ckpt["weights"]]
    return arch, weights
