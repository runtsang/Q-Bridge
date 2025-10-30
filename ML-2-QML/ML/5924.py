"""Hybrid classical LSTM with optional quantum‑inspired gates and a quantum‑style estimator head.

The module mirrors the original QLSTM implementation but replaces the
quantum gates with lightweight feed‑forward networks that emulate the
behaviour of a small quantum circuit.  It also exposes a
`HybridLSTMTagger` that can be used for sequence tagging tasks.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EstimatorNN(nn.Module):
    """Tiny regressor that mimics the behaviour of a quantum EstimatorQNN.

    The network maps a 2‑dimensional input to a single scalar and is
    used inside each gate to provide a learnable, yet fully classical,
    approximation of a quantum circuit.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected shape: (batch, 2) -> (batch, 1)
        return self.net(x)


class ClassicalGate(nn.Module):
    """Linear‑plus‑MLP gate that emulates a quantum LSTM gate."""
    def __init__(self, input_dim: int, hidden_dim: int, n_wires: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim + hidden_dim, n_wires)
        self.estimator = EstimatorNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, n_wires)
        z = self.estimator(self.linear(x))
        return torch.sigmoid(z)  # gate activation


class HybridQLSTM(nn.Module):
    """Classical LSTM where each gate is powered by a ClassicalGate."""
    def __init__(self, input_dim: int, hidden_dim: int, n_wires: int = 4) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_wires = n_wires

        self.forget_gate = ClassicalGate(input_dim, hidden_dim, n_wires)
        self.input_gate = ClassicalGate(input_dim, hidden_dim, n_wires)
        self.update_gate = ClassicalGate(input_dim, hidden_dim, n_wires)
        self.output_gate = ClassicalGate(input_dim, hidden_dim, n_wires)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = self.forget_gate(combined)
            i = self.input_gate(combined)
            g = torch.tanh(self.update_gate(combined))
            o = self.output_gate(combined)

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class HybridLSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_wires: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_wires > 0:
            self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_wires=n_wires)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["HybridQLSTM", "HybridLSTMTagger"]
