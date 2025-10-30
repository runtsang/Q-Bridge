"""Hybrid LSTM implementation combining strengths from the classical and quantum LSTM seeds."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _QuantumGate(nn.Module):
    """A lightweight quantum‑like gate that can be reused for each LSTM gate."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.lin = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mimics a quantum transformation: linear followed by sigmoid
        return torch.sigmoid(self.lin(x))


class QLSTM(nn.Module):
    """Hybrid LSTM that can operate in classical, quantum, or mixed modes."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0, mode: str = "mixed"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.mode = mode.lower()
        if self.mode not in {"classical", "quantum", "mixed"}:
            raise ValueError("mode must be one of 'classical', 'quantum', or'mixed'")

        # Linear projection for gates
        self.proj = nn.Linear(input_dim + hidden_dim, hidden_dim * 4)

        if self.mode == "classical":
            # Standard linear layers for each gate
            self.forget = nn.Linear(hidden_dim, hidden_dim)
            self.input = nn.Linear(hidden_dim, hidden_dim)
            self.update = nn.Linear(hidden_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, hidden_dim)
        else:
            # Shared quantum‑like gate for all four gates
            self.qgate = _QuantumGate(hidden_dim)

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            gate_inputs = self.proj(combined)  # shape (batch, hidden_dim*4)
            f_in, i_in, g_in, o_in = gate_inputs.chunk(4, dim=1)

            if self.mode == "classical":
                f = torch.sigmoid(self.forget(f_in))
                i = torch.sigmoid(self.input(i_in))
                g = torch.tanh(self.update(g_in))
                o = torch.sigmoid(self.output(o_in))
            else:
                # quantum or mixed mode
                f = torch.sigmoid(self.qgate(f_in))
                i = torch.sigmoid(self.qgate(i_in))
                g = torch.tanh(self.qgate(g_in))
                o = torch.sigmoid(self.qgate(o_in))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)


class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical, quantum, or mixed LSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        mode: str = "mixed",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, mode=mode)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
