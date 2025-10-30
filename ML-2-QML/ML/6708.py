"""Hybrid classical LSTM with optional quantum‑inspired gates and a regression head."""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumGate(nn.Module):
    """A lightweight MLP that mimics a quantum gate’s non‑linear behaviour."""

    def __init__(self, dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QLSTM(nn.Module):
    """
    Classical LSTM cell with optional quantum‑inspired gates.
    If ``use_quantum`` is True, each gate is processed through a
    QuantumGate module that adds expressive non‑linearities
    inspired by small quantum circuits.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        use_quantum: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum

        # Classical linear transforms for gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        if use_quantum:
            # Quantum‑inspired gates
            self.forget_gate = QuantumGate(hidden_dim)
            self.input_gate = QuantumGate(hidden_dim)
            self.update_gate = QuantumGate(hidden_dim)
            self.output_gate = QuantumGate(hidden_dim)

        # Auxiliary regression head (EstimatorQNN analogue)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in torch.unbind(inputs, dim=0):
            combined = torch.cat([x, hx], dim=1)

            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            if self.use_quantum:
                f = self.forget_gate(f)
                i = self.input_gate(i)
                g = self.update_gate(g)
                o = self.output_gate(o)

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
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    def predict_regression(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return a scalar regression value for each batch element."""
        return self.regressor(hidden)


class LSTMTagger(nn.Module):
    """
    Sequence tagging model that supports classical, hybrid, or
    pure quantum LSTM layers.  When ``use_quantum`` is True,
    the underlying LSTM cell includes quantum‑inspired gates.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_quantum: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(
            embedding_dim, hidden_dim, n_qubits=n_qubits, use_quantum=use_quantum
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
