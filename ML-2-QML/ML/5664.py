"""Classical LSTM with optional fully connected gating and optional quantum‑inspired gating via FCL.

This module implements the classical counterpart of the hybrid quantum‑classical
QLSTM__gen041.  It retains the original API of QLSTM.py while adding a
fully‑connected layer (FCL) that can be used to generate gate parameters
in a quantum‑aware fashion.  The class can be instantiated with n_qubits=0
to use ordinary linear gates, or with n_qubits>0 to produce a *classical*
approximation of the quantum gates by passing the linear output through
a small fully connected network.  This mirrors the behaviour of the
quantum version but keeps the implementation fully classical.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --------------------------------------------------------------------------- #
# Helper: Fully connected layer used in the quantum version
# --------------------------------------------------------------------------- #
class FCL(nn.Module):
    """Simple fully‑connected layer that mimics the quantum FCL example."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return torch.tanh(self.linear(values)).mean(dim=0)

# --------------------------------------------------------------------------- #
# Classical QLSTM
# --------------------------------------------------------------------------- #
class QLSTM__gen041(nn.Module):
    """Classical LSTM cell with optional fully‑connected gate generator."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Linear layers for the gates
        self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Optional fully‑connected layer to generate gate parameters
        if n_qubits > 0:
            self.fcl = FCL(n_qubits)
        else:
            self.fcl = None

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            # Optionally generate gate parameters via FCL
            if self.fcl is not None:
                # flatten linear outputs to a vector and feed into FCL
                f_raw = self.forget(combined)
                i_raw = self.input(combined)
                g_raw = self.update(combined)
                o_raw = self.output(combined)

                f = torch.sigmoid(self.fcl(f_raw))
                i = torch.sigmoid(self.fcl(i_raw))
                g = torch.tanh(self.fcl(g_raw))
                o = torch.sigmoid(self.fcl(o_raw))
            else:
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
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

# --------------------------------------------------------------------------- #
# Tagger
# --------------------------------------------------------------------------- #
class LSTMTagger(nn.Module):
    """Sequence‑tagging model that uses the hybrid QLSTM__gen041 cell."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM__gen041(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTM__gen041", "LSTMTagger"]
