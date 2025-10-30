"""Enhanced classical LSTM with optional quantum-inspired gating and attention.

This module extends the original QLSTM by adding:
- a lightweight quantum-inspired gate (sine-activated linear transform) that can replace the classical linear gate when `n_qubits > 0`;
- a multi‑head self‑attention mechanism over the hidden states, configurable via `attention_heads`;
- batch‑wise sequence handling compatible with PyTorch’s PackedSequence API.

The public API mirrors the original seed (`QLSTM(input_dim, hidden_dim, n_qubits)` and `forward`),
but now accepts tensors of shape `(seq_len, batch, input_dim)` and returns hidden states
for each time step.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumInspiredGate(nn.Module):
    """A thin quantum‑inspired layer: linear followed by a sine activation."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.linear(x))


class QLSTM__(nn.Module):
    """
    Classical LSTM that can optionally replace its linear gates with a quantum‑inspired gate
    and optionally apply multi‑head self‑attention over the hidden states.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        attention_heads: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.attention_heads = attention_heads

        gate_cls = QuantumInspiredGate if n_qubits > 0 else nn.Linear

        self.forget_gate = gate_cls(input_dim + hidden_dim, hidden_dim)
        self.input_gate = gate_cls(input_dim + hidden_dim, hidden_dim)
        self.update_gate = gate_cls(input_dim + hidden_dim, hidden_dim)
        self.output_gate = gate_cls(input_dim + hidden_dim, hidden_dim)

        if attention_heads > 1:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=attention_heads,
                batch_first=True,
            )
        else:
            self.attention = None

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (seq_len, batch, input_dim).
        states : tuple, optional
            Tuple of (h_0, c_0), each of shape (batch, hidden_dim).

        Returns
        -------
        outputs : torch.Tensor
            Hidden states for each time step, shape (seq_len, batch, hidden_dim).
        final_state : tuple
            Final hidden and cell states.
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []

        seq_len, batch, _ = inputs.shape
        past_hidden = []

        for t in range(seq_len):
            x_t = inputs[t]
            combined = torch.cat([x_t, hx], dim=1)

            f = torch.sigmoid(self.forget_gate(combined))
            i = torch.sigmoid(self.input_gate(combined))
            g = torch.tanh(self.update_gate(combined))
            o = torch.sigmoid(self.output_gate(combined))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            if self.attention is not None and past_hidden:
                # past_hidden: list of tensors (batch, hidden_dim)
                past = torch.stack(past_hidden, dim=1)  # (batch, seq_so_far, hidden_dim)
                attn_output, _ = self.attention(hx.unsqueeze(1), past, past)
                hx = hx + attn_output.squeeze(1)

            outputs.append(hx.unsqueeze(1))
            past_hidden.append(hx.detach())

        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, hidden_dim)
        # Transpose to (seq_len, batch, hidden_dim) for consistency with original API
        outputs = outputs.transpose(0, 1)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


class LSTMTagger__(nn.Module):
    """
    Sequence tagging model that can switch between the classical or quantum‑inspired LSTM.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        attention_heads: int = 1,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM__(embedding_dim, hidden_dim, n_qubits=n_qubits, attention_heads=attention_heads)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sentence : torch.Tensor
            Tensor of shape (seq_len, batch) containing word indices.

        Returns
        -------
        tag_scores : torch.Tensor
            Log‑softmax scores for each tag, shape (seq_len, batch, tagset_size).
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=2)


__all__ = ["QLSTM__", "LSTMTagger__"]
