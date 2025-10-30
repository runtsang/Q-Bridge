"""Graph‑based neural network with a quantum‑inspired head.

The module defines a single :class:`GraphQNNHybrid` that
* computes node embeddings with a classical feed‑forward network
  (mimicking the GNN logic of the first seed)
* optionally builds a fidelity‑based adjacency graph from the embeddings
* passes the final layer weight vector through a variational quantum
  circuit (implemented with Pennylane) and uses the expectation value
  as a classification score.
The design follows the two reference seeds but merges them into a
single, extensible interface.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Classical utilities (adapted from the first seed)
# ----------------------------------------------------------------------
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_network(qnn_arch: Sequence[int], samples: int):
    """Generate a random network, training data and target weight.

    Parameters
    ----------
    qnn_arch: sequence of int
        Layer sizes, e.g. [10, 20, 5].
    samples: int
        Number of training examples to create.

    Returns
    -------
    arch: list[int]
        The architecture list.
    weights: list[torch.Tensor]
        Weight matrices for each layer.
    training_data: list[tuple[torch.Tensor, torch.Tensor]]
        (input, target) pairs where target = W_last @ input.
    target_weight: torch.Tensor
        The last weight matrix (used as ground truth).
    """
    weights: List[torch.Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    # Generate training data using the target weight
    training_data: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(samples):
        inp = torch.randn(target_weight.size(1), dtype=torch.float32)
        tgt = target_weight @ inp
        training_data.append((inp, tgt))
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[torch.Tensor],
    samples: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> List[List[torch.Tensor]]:
    """Forward propagate inputs through the classical network.

    Returns a list of activation lists per sample.
    """
    activations: List[List[torch.Tensor]] = []
    for inp, _ in samples:
        current = inp
        layer_outputs = [current]
        for w in weights:
            current = torch.tanh(w @ current)
            layer_outputs.append(current)
        activations.append(layer_outputs)
    return activations

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap between two normalized vectors."""
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
    """Build a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# ----------------------------------------------------------------------
# Quantum utilities (adapted from the second seed, Pennylane version)
# ----------------------------------------------------------------------
import pennylane as qml

# A simple one‑qubit ansatz: apply RY for each input parameter
dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev, interface="torch")
def _qnode(params: torch.Tensor) -> torch.Tensor:
    """Return the expectation value of PauliZ after applying RY gates."""
    for p in params:
        qml.RY(p, wires=0)
    return qml.expval(qml.PauliZ(0))

def quantum_expectation(inputs: torch.Tensor) -> torch.Tensor:
    """Compute quantum expectation for each row in ``inputs``."""
    # ``inputs`` shape: (batch, features)
    return torch.stack([_qnode(row) for row in inputs])

# ----------------------------------------------------------------------
# Hybrid model
# ----------------------------------------------------------------------
class GraphQNNHybrid(nn.Module):
    """Hybrid graph‑neural‑network with a quantum head.

    Parameters
    ----------
    qnn_arch : sequence of int
        Layer sizes for the classical network.
    use_fidelity : bool, optional
        Whether to compute a fidelity‑based adjacency graph.
    fidelity_threshold : float, optional
        Threshold for edge creation.
    secondary_threshold : float, optional
        Secondary fidelity threshold.
    """

    def __init__(
        self,
        qnn_arch: Sequence[int],
        *,
        use_fidelity: bool = False,
        fidelity_threshold: float = 0.8,
        secondary_threshold: float | None = None,
    ) -> None:
        super().__init__()
        self.arch = list(qnn_arch)
        self.use_fidelity = use_fidelity
        self.fidelity_threshold = fidelity_threshold
        self.secondary_threshold = secondary_threshold

        # Initialize classical weights
        self.weights = nn.ParameterList(
            nn.Parameter(_random_linear(in_f, out_f))
            for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])
        )

        # Quantum head: we treat the last layer's output as parameters
        # for the quantum circuit. For a single‑qubit circuit we
        # restrict the number of parameters to 1; if the last layer
        # has more features we average them.
        self.quantum_dim = self.arch[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: classical propagation → optional graph → quantum head."""
        # Classical feed‑forward
        activations = [x]
        current = x
        for w in self.weights:
            current = torch.tanh(w @ current)
            activations.append(current)

        # Optional fidelity graph (just for demonstration)
        if self.use_fidelity:
            states = [act.detach() for act in activations]
            self.graph = fidelity_adjacency(
                states,
                self.fidelity_threshold,
                secondary=self.secondary_threshold,
            )
        else:
            self.graph = None

        # Prepare parameters for quantum circuit
        last_layer = activations[-1]
        if last_layer.shape[1] == 1:
            q_params = last_layer
        else:
            q_params = last_layer.mean(dim=1, keepdim=True)

        # Quantum expectation
        q_out = quantum_expectation(q_params)

        # Classification head: softmax over two classes
        logits = torch.cat([q_out, 1 - q_out], dim=-1)
        probs = F.softmax(logits, dim=-1)
        return probs

    def extra_repr(self) -> str:
        return f"qnn_arch={self.arch}, use_fidelity={self.use_fidelity}"

__all__ = ["GraphQNNHybrid", "random_network", "feedforward",
           "state_fidelity", "fidelity_adjacency"]
