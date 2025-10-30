"""Enhanced classical LSTM with quantum kernel approximation and attention.

This module keeps the same public API as the original `QLSTM` and
`LSTMTagger` classes but extends them with:
    * a quantum‑kernel‑style gate generator implemented as a small MLP,
    * an optional attention layer over the hidden states,
    * a multi‑task head that can output two tag sets.

The implementation is fully classical and can be used as a drop‑in
replacement for the original seed.  The new features are optional
and can be toggled via keyword arguments.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _QuantumKernelGate(nn.Module):
    """A tiny MLP that mimics a quantum kernel for gate generation."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QLSTM(nn.Module):
    """Drop‑in replacement that optionally uses a quantum‑kernel gate
    generator and an attention mechanism."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        use_quantum_gate: bool = False,
        use_attention: bool = False,
        multi_task: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_quantum_gate = use_quantum_gate
        self.use_attention = use_attention
        self.multi_task = multi_task

        if self.use_quantum_gate:
            self.forget_gate = _QuantumKernelGate(
                input_dim + hidden_dim, hidden_dim
            )
            self.input_gate = _QuantumKernelGate(
                input_dim + hidden_dim, hidden_dim
            )
            self.update_gate = _QuantumKernelGate(
                input_dim + hidden_dim, hidden_dim
            )
            self.output_gate = _QuantumKernelGate(
                input_dim + hidden_dim, hidden_dim
            )
        else:
            self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
            self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        if self.use_attention:
            # Attention MLP that outputs a scalar per hidden state
            self.attention_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            if self.use_quantum_gate:
                f = torch.sigmoid(self.forget_gate(combined))
                i = torch.sigmoid(self.input_gate(combined))
                g = torch.tanh(self.update_gate(combined))
                o = torch.sigmoid(self.output_gate(combined))
            else:
                f = torch.sigmoid(self.forget_linear(combined))
                i = torch.sigmoid(self.input_linear(combined))
                g = torch.tanh(self.update_linear(combined))
                o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)

        if self.use_attention:
            # Compute attention weights over the sequence
            attn_weights = self.attention_mlp(outputs).squeeze(-1)  # (seq_len,)
            attn_weights = F.softmax(attn_weights, dim=0)
            context = torch.sum(attn_weights.unsqueeze(-1) * outputs, dim=0)
            # Broadcast context to all time steps
            outputs = context.unsqueeze(0).repeat(outputs.size(0), 1)

        return outputs, (hx, cx)


class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and
    quantum‑kernel LSTM and supports optional attention and multi‑task
    heads."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        use_quantum_gate: bool = False,
        use_attention: bool = False,
        multi_task: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(
            embedding_dim,
            hidden_dim,
            n_qubits=n_qubits,
            use_quantum_gate=use_quantum_gate,
            use_attention=use_attention,
            multi_task=multi_task,
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        if multi_task:
            self.hidden2chunk = nn.Linear(hidden_dim, tagset_size)

        self.use_attention = use_attention
        self.multi_task = multi_task

    def forward(self, sentence: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        if self.multi_task:
            chunk_logits = self.hidden2chunk(lstm_out.view(len(sentence), -1))
            return F.log_softmax(tag_logits, dim=1), F.log_softmax(chunk_logits, dim=1)
        else:
            return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
