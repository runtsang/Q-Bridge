"""Hybrid classical LSTM with optional stochastic gating via SamplerQNN.

This module defines QLSTM and LSTMTagger classes that are drop‑in
replacements for the original QLSTM.py.  The LSTM gates are realised
by linear layers, but an optional SamplerQNN module can be attached
to modulate the gate activations.  The SamplerQNN outputs a softmax
distribution over four gate probabilities; during the forward pass
the gates are multiplied by these probabilities, providing a
stochastic regulariser that can improve generalisation for small
datasets.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# SamplerQNN – a lightweight probabilistic mask generator
# --------------------------------------------------------------------------- #
class SamplerQNN(nn.Module):
    """
    Produces a probability for each of the four LSTM gates.
    Input shape: (batch, input_dim + hidden_dim)
    Output shape: (batch, 4)
    """
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )

    def forward(self, combined: torch.Tensor) -> torch.Tensor:
        # Softmax over 4 gate probabilities
        return F.softmax(self.net(combined), dim=-1)


# --------------------------------------------------------------------------- #
# Classical QLSTM with optional stochastic gating
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """
    Drop‑in replacement using classical linear gates.
    If ``use_sampler`` is True, the gates are multiplied by a
    sampled probability vector from SamplerQNN.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        use_sampler: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_sampler = use_sampler

        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

        if self.use_sampler:
            self.sampler = SamplerQNN(input_dim, hidden_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            if self.use_sampler:
                probs = self.sampler(combined)            # (batch, 4)
                f = f * probs[:, 0:1]
                i = i * probs[:, 1:2]
                g = g * probs[:, 2:3]
                o = o * probs[:, 3:4]

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
# Sequence tagging model
# --------------------------------------------------------------------------- #
class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between classical and quantum LSTM.
    The ``use_sampler`` flag activates the stochastic gating in the classical LSTM.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_sampler: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = QLSTM(embedding_dim, hidden_dim, use_sampler=use_sampler)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
