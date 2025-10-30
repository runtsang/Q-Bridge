"""Hybrid classical LSTM tagger with graph‑based state analysis.

This module mirrors the original QLSTMTagger but adds:
  * a quantum‑style LSTM cell (purely classical implementation)
  * optional quantum‑style fully‑connected classifier
  * a helper that builds a weighted graph of hidden‑state fidelities
    using the GraphQNN fidelity functions.

The :class:`HybridQLSTMTagger` can be instantiated with
``n_qubits>0`` to use the quantum‑style cell and with
``use_qfc=True`` to employ a linear classifier (placeholder for a
quantum variant).  The class remains fully compatible with the
original API.
"""

from __future__ import annotations

import itertools
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx


# --------------------------------------------------------------------------- #
# Classical LSTM cell that mimics the interface of the quantum version
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """Classical implementation of a quantum‑style LSTM cell.

    The gates are realised with linear layers followed by sigmoid/tanh
    activations, mirroring the structure of the quantum variant but
    operating entirely on classical tensors.  The ``n_qubits`` argument
    is retained for API compatibility but is unused.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


# --------------------------------------------------------------------------- #
# Utility functions for fidelity‑based graph construction
# --------------------------------------------------------------------------- #
def _state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap between two unit‑norm vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)


def fidelity_graph(
    states: Tuple[torch.Tensor,...],
    threshold: float,
    *,
    secondary: Optional[float] = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Return a weighted graph whose nodes are the hidden‑state vectors.

    Edges are added if the fidelity exceeds ``threshold`` (weight 1).  If
    ``secondary`` is provided, states with fidelity between
    ``secondary`` and ``threshold`` receive ``secondary_weight``.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = _state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# Hybrid tagger that can use a classical or quantum LSTM and an optional
# quantum fully‑connected classifier.
# --------------------------------------------------------------------------- #
class HybridQLSTMTagger(nn.Module):
    """Drop‑in replacement for the original ``QLSTMTagger`` with added
    quantum‑style gates and graph analysis.

    Parameters
    ----------
    embedding_dim : int
        Dimension of word embeddings.
    hidden_dim : int
        Hidden size of the LSTM.
    vocab_size : int
        Size of the vocabulary.
    tagset_size : int
        Number of output tags.
    n_qubits : int, default 0
        If > 0 the LSTM gates are replaced by a quantum‑style cell.
    use_qfc : bool, default False
        If ``True`` the classification layer is a simple linear
        projection; otherwise it is a quantum fully‑connected module
        (see :class:`QuantumFullyConnected` in the QML variant).
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_qfc: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        if use_qfc:
            # Simple linear classifier (placeholder for a quantum variant)
            self.fc = nn.Linear(hidden_dim, tagset_size)
        else:
            self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        logits = self.fc(lstm_out.view(len(sentence), -1))
        return F.log_softmax(logits, dim=1)

    def fidelity_graph(
        self,
        hidden_states: torch.Tensor,
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a graph from the hidden states of a single sequence.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Tensor of shape ``(seq_len, batch, hidden_dim)``.
        threshold : float
            Fidelity threshold for edge creation.
        secondary : float | None
            Secondary threshold for weighted edges.
        secondary_weight : float
            Weight used for secondary edges.

        Returns
        -------
        networkx.Graph
            Weighted adjacency graph of hidden states.
        """
        seq_len = hidden_states.size(0)
        states = tuple(hidden_states[i, 0, :] for i in range(seq_len))
        return fidelity_graph(states, threshold, secondary=secondary, secondary_weight=secondary_weight)


__all__ = ["HybridQLSTMTagger", "QLSTM", "fidelity_graph"]
