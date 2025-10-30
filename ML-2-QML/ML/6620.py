"""
Hybrid classical LSTM with cell abstraction and stackable architecture.
The module preserves the original API but introduces a lightweight
QLSTMCell that can be reused or stacked for research experiments.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QLSTMCell(nn.Module):
    """Classic LSTM cell suitable for stacking."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gates = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = states
        combined = torch.cat([x, hx], dim=1)
        g = self.gates(combined)
        f, i, g_, o = g.chunk(4, dim=1)
        f = torch.sigmoid(f)
        i = torch.sigmoid(i)
        g_ = torch.tanh(g_)
        o = torch.sigmoid(o)
        cx = f * cx + i * g_
        hx = o * torch.tanh(cx)
        return hx, (hx, cx)


class QLSTMStack(nn.Module):
    """Stack of LSTM cells."""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [QLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim)
             for i in range(num_layers)]
        )

    def forward(
        self,
        inputs: torch.Tensor,
        init_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = inputs.size(1)
        if init_states is None:
            hx = torch.zeros(batch_size, self.cells[0].hidden_dim, device=inputs.device)
            cx = torch.zeros(batch_size, self.cells[0].hidden_dim, device=inputs.device)
        else:
            hx, cx = init_states

        outputs = []
        for t in range(inputs.size(0)):
            x = inputs[t]
            for cell in self.cells:
                x, (hx, cx) = cell(x, (hx, cx))
            outputs.append(x.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)


class QLSTM(nn.Module):
    """Drop‑in replacement for the original QLSTM but built from QLSTMCell."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        # n_qubits is ignored in the classical version but kept for API parity
        self.lstm_stack = QLSTMStack(input_dim, hidden_dim, num_layers=1)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.lstm_stack(inputs, init_states=states)


class LSTMTagger(nn.Module):
    """Sequence tagging model that can use either classical QLSTM or nn.LSTM."""
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
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "QLSTMCell", "QLSTMStack", "LSTMTagger"]
