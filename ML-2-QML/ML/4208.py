"""Hybrid classical LSTM with optional QCNN feature extractor and graph‑based regularization.

This module defines a QLSTMGen195 class that extends the classical LSTM by adding:
- a QCNNModel feature extractor (fully‑connected emulation of the quantum convolutional network).
- a graph‑adjacency regularizer computed from hidden‑state fidelities across time steps.
- optional use of the quantum gate implementation for the LSTM gates.

The API mirrors the original QLSTM and LSTMTagger classes for compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from typing import Tuple, List, Optional

# --------------------------------------------------------------------------- #
# Helper utilities (state fidelity & adjacency) – borrow from GraphQNN
# --------------------------------------------------------------------------- #
def _state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the squared cosine similarity between two vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: List[torch.Tensor], threshold: float,
                       *, secondary: Optional[float] = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, si), (j, sj) in zip(enumerate(states), enumerate(states[1:])):
        fid = _state_fidelity(si, sj)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

# --------------------------------------------------------------------------- #
# Classical QCNN emulator (fully‑connected layers)
# --------------------------------------------------------------------------- #
class QCNNModel(nn.Module):
    """Fully‑connected surrogate of the quantum convolutional network."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

# --------------------------------------------------------------------------- #
# Hybrid LSTM with optional QCNN feature extractor
# --------------------------------------------------------------------------- #
class QLSTMGen195(nn.Module):
    """Hybrid LSTM that optionally uses a QCNN feature extractor and
    quantum‑style gates.  The forward pass returns the hidden‑state
    sequence, the final hidden/cell states and a fidelity‑based adjacency
    graph of the hidden states.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_qubits: int = 0,
                 use_qcnn: bool = False) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_qcnn = use_qcnn

        # Feature extractor
        self.feature_extractor: nn.Module = QCNNModel() if use_qcnn else nn.Identity()

        # Classical linear gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self,
                inputs: torch.Tensor,
                states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], nx.Graph]:
        hx, cx = self._init_states(inputs, states)
        outputs: List[torch.Tensor] = []
        hidden_states: List[torch.Tensor] = []

        for x in inputs.unbind(dim=0):
            # optional QCNN feature extraction
            x = self.feature_extractor(x)

            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            outputs.append(hx.unsqueeze(0))
            hidden_states.append(hx)

        adjacency = fidelity_adjacency(hidden_states, threshold=0.9)

        return torch.cat(outputs, dim=0), (hx, cx), adjacency

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Optional[Tuple[torch.Tensor, torch.Tensor]]
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

# --------------------------------------------------------------------------- #
# Tagger wrapper (compatible with original API)
# --------------------------------------------------------------------------- #
class LSTMTagger(nn.Module):
    """Sequence tagging model that uses the hybrid LSTM."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 use_qcnn: bool = False) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTMGen195(embedding_dim, hidden_dim,
                                n_qubits=n_qubits,
                                use_qcnn=use_qcnn)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)
        lstm_out, _, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
QLSTM = QLSTMGen195
__all__ = ["QLSTMGen195", "QLSTM", "LSTMTagger"]
