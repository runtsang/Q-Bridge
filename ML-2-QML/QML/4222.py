"""Hybrid quantum‑classical LSTM tagger with graph‑based state analysis.

This module mirrors the classical version but replaces the LSTM cell
with a quantum‑style implementation and the classifier with a
quantum fully‑connected layer.  It also exposes the same
``fidelity_graph`` helper.  The class name is kept identical for
drop‑in compatibility with the original API.
"""

from __future__ import annotations

import itertools
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import networkx as nx


# --------------------------------------------------------------------------- #
# Quantum‑style LSTM cell using a small variational circuit per gate
# --------------------------------------------------------------------------- #
class QLSTM(tq.QuantumModule):
    """LSTM cell where gates are realised by small quantum circuits."""

    class QGate(tq.QuantumModule):
        """Gate module that encodes a real‑valued vector into qubits,
        applies a trainable circuit, and measures the Pauli‑Z basis.
        """

        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            # Simple encoder: one RX per wire
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            # Trainable rotation layer
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev)
            for gate in self.params:
                gate(qdev)
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Each gate is a QGate; the output dimension is n_qubits
        self.forget_gate = self.QGate(n_qubits)
        self.input_gate = self.QGate(n_qubits)
        self.update_gate = self.QGate(n_qubits)
        self.output_gate = self.QGate(n_qubits)

        # Linear projections to the qubit register
        self.forget_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_lin = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(self.forget_lin(combined)))
            i = torch.sigmoid(self.input_gate(self.input_lin(combined)))
            g = torch.tanh(self.update_gate(self.update_lin(combined)))
            o = torch.sigmoid(self.output_gate(self.output_lin(combined)))
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
# Quantum fully‑connected layer inspired by the Quantum‑NAT example
# --------------------------------------------------------------------------- #
class QuantumFullyConnected(tq.QuantumModule):
    """Quantum fully‑connected layer that maps a vector of length ``input_dim``
    to ``output_dim`` using a trainable circuit and a linear readout.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.n_wires = input_dim
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_ryzxy"][:input_dim]
        )
        self.circuit = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.fc = nn.Linear(self.n_wires, output_dim)
        self.norm = nn.BatchNorm1d(output_dim)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev)
        self.circuit(qdev)
        return self.measure(qdev)

    def forward_batch(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        out = self.forward(qdev)
        out = self.fc(out)
        return self.norm(out)


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
        If ``True`` the classification layer is a quantum fully‑connected
        module; otherwise it is a classical linear layer.
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
            self.qfc = QuantumFullyConnected(hidden_dim, tagset_size)
        else:
            self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        if hasattr(self, "qfc"):
            logits = self.qfc.forward_batch(lstm_out.view(len(sentence), -1))
        else:
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


__all__ = ["HybridQLSTMTagger", "QLSTM", "QuantumFullyConnected", "fidelity_graph"]
