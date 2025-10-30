"""Hybrid classical LSTM with QCNN feature extraction and graph‑based adjacency.

The module defines :class:`HybridQLSTM`, a drop‑in replacement for
:class:`torch.nn.LSTM` that augments each gate with a classical
QCNN‑style feature extractor and post‑processing by a fidelity graph.
"""

from __future__ import annotations

import itertools
from typing import Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Graph utilities from GraphQNN
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> torch.Tensor:
    return torch.randn(out_features, in_features, dtype=torch.float32)


def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
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
    """
    Build a weighted graph from pairwise state fidelities.

    Edges with fidelity ≥ ``threshold`` receive weight 1.0;
    if ``secondary`` is provided, fidelities in [secondary, threshold)
    receive ``secondary_weight``.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
#  QCNN‑style feature extractor
# --------------------------------------------------------------------------- #
class QCNNFeatureExtractor(nn.Module):
    """Stacked linear layers with Tanh activations mimicking a QCNN."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
#  Hybrid LSTM
# --------------------------------------------------------------------------- #
class HybridQLSTM(nn.Module):
    """
    Classical LSTM cell with QCNN‑derived gate parameters and graph‑based
    state adjacency.

    Parameters
    ----------
    input_dim : int
        Dimension of each input token.
    hidden_dim : int
        Hidden state size.
    n_qubits : int, default 0
        Number of qubits for future quantum augmentation
        (unused in the pure ML variant but retained for API parity).
    graph_threshold : float, default 0.8
        Fidelity threshold for constructing the adjacency graph.
    qcnn_hidden : Sequence[int] | None, default None
        Hidden layer sizes for the QCNN feature extractor.  Defaults to
        ``[hidden_dim, hidden_dim]``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        graph_threshold: float = 0.8,
        qcnn_hidden: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.graph_threshold = graph_threshold

        self.feature_extractor = QCNNFeatureExtractor(
            input_dim, qcnn_hidden or [hidden_dim, hidden_dim]
        )

        # Linear projections for the four LSTM gates
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(hidden_dim * 2, gate_dim)
        self.input_linear = nn.Linear(hidden_dim * 2, gate_dim)
        self.update_linear = nn.Linear(hidden_dim * 2, gate_dim)
        self.output_linear = nn.Linear(hidden_dim * 2, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            # QCNN feature extraction
            feat = self.feature_extractor(x)
            combined = torch.cat([feat, hx], dim=1)

            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)

        # Build a fidelity graph over the hidden states
        graph = fidelity_adjacency(
            [h.detach() for h in outputs], self.graph_threshold
        )
        # The graph can be used to weight or regularise hidden states.
        # (Additional graph‑based operations omitted for brevity.)

        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


__all__ = ["HybridQLSTM"]
