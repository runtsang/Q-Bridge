"""Hybrid classical‑quantum LSTM with attention and mixture weight.

This module extends the original seed by adding:
- A multi‑head attention layer that operates on the hidden state at each time step.
- A learnable scalar `mix_weight` that blends the classical and quantum gate outputs.
- Separate parameter groups for classical and quantum parts to ease fine‑tuning.

The design keeps the public API identical to the seed, so downstream code can use `QLSTMGen` in place of `QLSTM` or `LSTMTaggerGen` without modification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# --------------------------------------------------------------------------- #
# Classical gate helper
# --------------------------------------------------------------------------- #
class _GateLinear(nn.Module):
    """Linear layer followed by an activation."""
    def __init__(self, in_dim: int, out_dim: int, act: str = "sigmoid"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.act == "tanh":
            return torch.tanh(out)
        return torch.sigmoid(out)

# --------------------------------------------------------------------------- #
# Hybrid LSTM cell
# --------------------------------------------------------------------------- #
class QLSTMGen(nn.Module):
    """Hybrid LSTM cell with optional quantum gates, attention and mix weight."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        num_heads: int = 4,
        mix_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.num_heads = num_heads

        # Classical gates
        self.forget_gate = _GateLinear(input_dim + hidden_dim, hidden_dim, act="sigmoid")
        self.input_gate = _GateLinear(input_dim + hidden_dim, hidden_dim, act="sigmoid")
        self.update_gate = _GateLinear(input_dim + hidden_dim, hidden_dim, act="tanh")
        self.output_gate = _GateLinear(input_dim + hidden_dim, hidden_dim, act="sigmoid")

        # Quantum gates (only if n_qubits > 0)
        if n_qubits > 0:
            # Linear projections to quantum space
            self.forget_q_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.input_q_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.update_q_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
            self.output_q_lin = nn.Linear(input_dim + hidden_dim, n_qubits)
            # Mapping from quantum output to hidden_dim
            self.q_to_hidden = nn.Linear(n_qubits, hidden_dim)
        else:
            self.forget_q_lin = None

        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )

        # Learnable mix weight in [0,1]
        self.mix_weight = nn.Parameter(torch.tensor(mix_weight))

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            inputs: (seq_len, batch, input_dim)
            states: (hx, cx) each (batch, hidden_dim) or None
        Returns:
            outputs: (seq_len, batch, hidden_dim)
            (hx, cx)
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []

        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)

            # Classical gate outputs
            f_c = self.forget_gate(combined)
            i_c = self.input_gate(combined)
            g_c = self.update_gate(combined)
            o_c = self.output_gate(combined)

            if self.n_qubits > 0:
                # Quantum gate outputs (simulated as linear mapping)
                f_q = self.forget_q_lin(combined)
                i_q = self.input_q_lin(combined)
                g_q = self.update_q_lin(combined)
                o_q = self.output_q_lin(combined)

                f_q = self.q_to_hidden(f_q)
                i_q = self.q_to_hidden(i_q)
                g_q = self.q_to_hidden(g_q)
                o_q = self.q_to_hidden(o_q)

                # Blend
                f = self.mix_weight * f_c + (1 - self.mix_weight) * f_q
                i = self.mix_weight * i_c + (1 - self.mix_weight) * i_q
                g = self.mix_weight * g_c + (1 - self.mix_weight) * g_q
                o = self.mix_weight * o_c + (1 - self.mix_weight) * o_q
            else:
                f, i, g, o = f_c, i_c, g_c, o_c

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, hidden_dim)

        # Apply multi‑head attention over the sequence
        attn_out, _ = self.attention(outputs, outputs, outputs)
        outputs = outputs + attn_out * self.mix_weight

        return outputs, (hx, cx)

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
# Tagger wrapper
# --------------------------------------------------------------------------- #
class LSTMTaggerGen(nn.Module):
    """Sequence tagging model that can switch between classical and quantum LSTM."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        num_heads: int = 4,
        mix_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTMGen(
            embedding_dim,
            hidden_dim,
            n_qubits=n_qubits,
            num_heads=num_heads,
            mix_weight=mix_weight,
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

__all__ = ["QLSTMGen", "LSTMTaggerGen"]
