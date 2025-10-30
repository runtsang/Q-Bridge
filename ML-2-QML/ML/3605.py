"""Hybrid classical regressor that integrates a lightweight MLP with a quantum estimator.

The class exposes a `forward` method that accepts a batch of 2‑dimensional inputs,
produces a 3‑dimensional embedding, and then feeds that embedding into a
user‑provided quantum estimator. The quantum estimator is expected to
implement a `__call__(self, inputs: torch.Tensor) -> torch.Tensor` signature
and return a scalar expectation value for each input.

The module also re‑exports graph‑based utilities from the original GraphQNN
seed, allowing the generation of synthetic training data and the construction
of fidelity‑based adjacency graphs.
"""
from __future__ import annotations

import itertools
import networkx as nx
from typing import Iterable, Sequence, Tuple, List

import torch
from torch import nn

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
#  Graph‑based utilities – copied from the original GraphQNN seed
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
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
    "EstimatorQNNFusion",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]

# --------------------------------------------------------------------------- #
#  Hybrid estimator – classical front‑end
# --------------------------------------------------------------------------- #

class EstimatorQNNFusion(nn.Module):
    """
    A two‑stage hybrid regressor that first maps the raw 2‑D input to a
    compact embedding using a lightweight MLP, and then feeds that embedding
    into a user‑provided quantum estimator that refines the prediction.

    Parameters
    ----------
    quantum_estimator : callable
        A callable that accepts a torch.Tensor of shape (batch, dim)
        and returns a torch.Tensor of shape (batch, 1) containing the
        quantum‑derived expectation values.
    embedding_dim : int, default 8
        Size of the hidden embedding produced by the MLP.
    post_norm : bool, default False
        Whether to apply a final layer‑norm before returning the output.
    """

    def __init__(
        self,
        quantum_estimator: callable,
        embedding_dim: int = 8,
        post_norm: bool = False,
    ) -> None:
        super().__init__()
        self.quantum_estimator = quantum_estimator

        # Lightweight embedding MLP
        self.embedding = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
        )

        self.post_norm = post_norm
        if self.post_norm:
            self.norm = nn.LayerNorm(1)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass through the hybrid model.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch, 2).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, 1).
        """
        # Classical embedding
        embed = self.embedding(inputs)

        # Quantum refinement – the estimator must be able to handle batched input
        with torch.no_grad():
            quantum_out = self.quantum_estimator(embed)

        # Ensure we have a 2‑D tensor
        quantum_out = quantum_out.view(-1, 1)

        if self.post_norm:
            quantum_out = self.norm(quantum_out)

        return quantum_out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(embedding_dim={self.embedding[0].out_features}, "
            f"quantum_estimator={self.quantum_estimator!r})"
        )
