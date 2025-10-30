"""Hybrid classical LSTM with optional fully connected gate layer.

This module implements a classical LSTM cell that can operate in two modes:
- Standard linear gates (default), identical to a textbook LSTM.
- Fully connected gates via a simple tanh‑activated linear layer (FCL),
  providing a lightweight quantum‑inspired transformation.

The API mirrors the original QLSTM module so that existing code can
swap in this implementation without modification.  The `n_qubits`
parameter is retained for API stability but ignored in the purely
classical setting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class QLSTM(nn.Module):
    """Classical LSTM cell with optional fully connected gate transformation."""

    class GateLayer(nn.Module):
        """Linear gate layer, optionally wrapped in a tanh‑activated fully connected layer."""
        def __init__(self, input_size: int, output_size: int, use_fcl: bool = False):
            super().__init__()
            self.use_fcl = use_fcl
            self.linear = nn.Linear(input_size, output_size)
            if use_fcl:
                self.activation = nn.Tanh()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.linear(x)
            if self.use_fcl:
                y = self.activation(y)
            return y

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0, use_fcl: bool = False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits  # retained for API compatibility
        gate_input = input_dim + hidden_dim
        self.forget_gate = self.GateLayer(gate_input, hidden_dim, use_fcl)
        self.input_gate = self.GateLayer(gate_input, hidden_dim, use_fcl)
        self.update_gate = self.GateLayer(gate_input, hidden_dim, use_fcl)
        self.output_gate = self.GateLayer(gate_input, hidden_dim, use_fcl)

    def _init_states(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    def forward(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass over a sequence of inputs (T, N, D)."""
        batch_size = inputs.size(1)
        device = inputs.device
        hx, cx = self._init_states(batch_size, device) if states is None else states

        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(combined))
            i = torch.sigmoid(self.input_gate(combined))
            g = torch.tanh(self.update_gate(combined))
            o = torch.sigmoid(self.output_gate(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)


class LSTMTagger(nn.Module):
    """Sequence tagging model that uses the hybrid QLSTM cell."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0, use_fcl: bool = False) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, use_fcl=use_fcl)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.unsqueeze(1))
        tag_logits = self.hidden2tag(lstm_out.squeeze(1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
