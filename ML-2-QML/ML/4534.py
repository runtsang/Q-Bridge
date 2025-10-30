"""Hybrid classical LSTM model with optional regression and graph utilities.

This module combines the classical LSTM implementation from the original
QLSTM.py, adds a lightweight regression head inspired by EstimatorQNN and
QuantumRegression, and provides graph‑based adjacency construction from
GraphQNN.  The class can act as a sequence tagger or a regression model
depending on the `regression` flag."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for regression experiments."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset wrapper for the synthetic regression data."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute squared overlap between two real vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

# --------------------------------------------------------------------------- #
# Classical LSTM cell (same as original QLSTM)
# --------------------------------------------------------------------------- #

class QLSTM(nn.Module):
    """Drop‑in classical replacement for the quantum LSTM gate implementation."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
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
# Hybrid model combining tagging, regression, and graph utilities
# --------------------------------------------------------------------------- #

class HybridQLSTM(nn.Module):
    """
    A versatile LSTM‑based model that can act as a sequence tagger,
    a regression head, or a graph‑aware encoder.

    Parameters
    ----------
    embedding_dim : int
        Dimension of word embeddings.
    hidden_dim : int
        Hidden size of the LSTM.
    vocab_size : int
        Vocabulary size for embeddings.
    tagset_size : int
        Number of tags for the tagging head.
    n_qubits : int, default 0
        If >0, the LSTM gates are implemented by a lightweight quantum module
        (see the quantum implementation in the qml_code section).
    regression : bool, default False
        If True, a regression head is added on top of the LSTM outputs.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        regression: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Classical LSTM or quantum‑inspired gates
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.regression = regression
        if regression:
            self.regression_head = nn.Linear(hidden_dim, 1)

    def forward(self, sentence: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass returning tag logits and optional regression output.
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_logits = F.log_softmax(tag_logits, dim=1)
        regression_out = None
        if self.regression:
            # Use the last hidden state for regression
            regression_out = self.regression_head(lstm_out[-1].squeeze(0))
        return tag_logits, regression_out

    def compute_graph_adjacency(
        self,
        hidden_states: torch.Tensor,
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> "nx.Graph":
        """
        Build a weighted graph from hidden states using fidelity as edge weight.
        """
        import networkx as nx
        graph = nx.Graph()
        graph.add_nodes_from(range(hidden_states.size(0)))
        for i in range(hidden_states.size(0)):
            for j in range(i + 1, hidden_states.size(0)):
                fid = state_fidelity(hidden_states[i], hidden_states[j])
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph

__all__ = ["HybridQLSTM", "RegressionDataset", "generate_superposition_data"]
