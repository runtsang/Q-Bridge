"""Hybrid quantum‑classical LSTM (classical backbone).

This module extends the original QLSTM by adding a
`HybridTagger` that can be used in downstream NLP tasks.
The LSTM cell is fully classical and can be switched out
for a quantum version by importing the quantum implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HybridQLSTMConfig:
    """Configuration for the hybrid LSTM."""
    input_dim: int
    hidden_dim: int
    n_qubits: int  # kept for API compatibility; unused in classical mode
    batch_size: int = 32
    dropout: float = 0.0


class HybridQLSTM(nn.Module):
    """Classical LSTM cell that mimics the interface of the quantum version."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int,
        mode: str = "classical",
    ) -> None:
        super().__init__()
        if mode!= "classical":
            raise ValueError("Classical implementation only supports mode='classical'")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Classical linear gates
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
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class HybridTagger(nn.Module):
    """Sequence tagging model that uses the hybrid LSTM."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        mode: str = "classical",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if mode == "quantum" and n_qubits > 0:
            # In classical module we cannot load quantum implementation;
            # users should import the quantum module instead.
            raise RuntimeError(
                "Quantum mode is not available in the classical implementation. "
                "Import the quantum module to use a quantum LSTM."
            )
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_qubits, mode="classical")
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["HybridQLSTM", "HybridTagger", "HybridQLSTMConfig"]
