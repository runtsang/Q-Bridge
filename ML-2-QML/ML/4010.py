"""Hybrid graph‑based classical neural network with an optional quantum head.

The module defines ``QuantumGraphClassifier`` which:
1.  Builds a dense feed‑forward backbone mirroring the ML seed.
2.  Accepts an optional quantum head callable (built in the QML module) that
    transforms the final hidden state into a 2‑class probability.
3.  Provides a fidelity‑based graph regularizer that connects hidden
    representations, inspired by the QML seed.
4.  Exposes ``fit`` and ``predict`` methods compatible with sklearn‑style
    estimators.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple, Optional

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# 1. Classical backbone
# --------------------------------------------------------------------------- #
class _ClassicalBackbone(nn.Module):
    """Simple dense network mirrored from the ML seed."""

    def __init__(self, num_features: int, depth: int) -> None:
        super().__init__()
        self.layers: List[nn.Module] = []
        self.activations: List[nn.Module] = []
        in_dim = num_features
        for _ in range(depth):
            lin = nn.Linear(in_dim, num_features)
            self.layers.append(lin)
            self.activations.append(nn.ReLU())
            in_dim = num_features
        self.out = nn.Linear(in_dim, 2)

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        activations: List[Tensor] = []
        for lin, act in zip(self.layers, self.activations):
            x = act(lin(x))
            activations.append(x)
        logits = self.out(x)
        return logits, activations


# --------------------------------------------------------------------------- #
# 2. Fidelity helpers (re‑used from the ML seed)
# --------------------------------------------------------------------------- #
def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared overlap between two unit‑normed tensors."""
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
    """Build a weighted graph from state fidelities."""
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
# 3. Hybrid model
# --------------------------------------------------------------------------- #
class QuantumGraphClassifier(nn.Module):
    """
    Hybrid classical‑quantum classifier.

    Parameters
    ----------
    num_features : int
        Size of the input feature vector.
    depth : int
        Depth of the classical dense backbone.
    quantum_head : Optional[Callable[[Tensor], Tensor]]
        Callable that takes the final hidden state and returns a
        probability distribution.  Typically a quantum variational
        circuit defined in the QML module.  If None, the model reduces
        to a pure classical classifier.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        quantum_head: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        super().__init__()
        self.backbone = _ClassicalBackbone(num_features, depth)
        self.quantum_head = quantum_head

    def forward(self, x: Tensor) -> Tensor:
        logits, activations = self.backbone(x)
        # If a quantum head is provided, use it to refine logits
        if self.quantum_head is not None:
            # The quantum head expects a unit‑normed vector
            hidden = activations[-1]
            hidden_norm = hidden / (torch.norm(hidden) + 1e-12)
            q_logits = self.quantum_head(hidden_norm)
            # Combine classical logits with quantum output
            logits = logits + q_logits
        return logits

    def fit(
        self,
        X: Tensor,
        y: Tensor,
        epochs: int = 20,
        lr: float = 1e-3,
        reg_graph: Optional[nx.Graph] = None,
        reg_weight: float = 1e-4,
    ) -> None:
        """
        Simple training loop with optional graph‑based regularization.

        Parameters
        ----------
        X : Tensor
            Input features (N × D).
        y : Tensor
            Binary labels (N,).
        epochs : int
            Number of epochs.
        lr : float
            Learning rate.
        reg_graph : nx.Graph, optional
            Graph constructed from hidden states.  If provided, the
            adjacency weights are used as a regularization term.
        reg_weight : float
            Weight of the regularization term.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            logits, activations = self.backbone(X)
            loss = loss_fn(logits, y)
            if reg_graph is not None:
                # Simple Laplacian regularization on hidden states
                hidden = activations[-1]
                laplacian = torch.eye(hidden.size(0))
                for u, v, data in reg_graph.edges(data=True):
                    w = data.get("weight", 1.0)
                    laplacian[u, u] -= w
                    laplacian[v, v] -= w
                    laplacian[u, v] += w
                    laplacian[v, u] += w
                reg = torch.trace(hidden.t() @ laplacian @ hidden)
                loss += reg_weight * reg
            loss.backward()
            optimizer.step()

    def predict(self, X: Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(X)
            return torch.argmax(logits, dim=1)

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int, depth: int
    ) -> Tuple[QuantumCircuit, Sequence, Sequence, List[SparsePauliOp]]:
        """
        Wrapper that forwards to the quantum implementation in the
        QML module.  The function is imported lazily to keep the ML
        module free of quantum imports.
        """
        from.quantum_classifier_head import build_classifier_circuit
        return build_classifier_circuit(num_qubits, depth)


__all__ = ["QuantumGraphClassifier"]
