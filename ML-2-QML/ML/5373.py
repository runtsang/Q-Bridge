"""Hybrid LSTM with optional quantum gates and an attention‑based decoder.

The implementation extends the classical QLSTM seed by adding a lightweight
self‑attention decoder that can optionally be quantum‑augmented.  The
`HybridQLSTMTagger` class provides a drop‑in replacement for the original
`QLSTMTagger` and can be used with the `FastBaseEstimator` utilities for
evaluation.  Quantum gates are implemented via a tiny variational circuit
using TorchQuantum, but the module falls back to classical linear gates when
the optional dependency is unavailable or when `n_qubits` is zero.
"""

from __future__ import annotations

from typing import Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Optional quantum dependency; used only when `n_qubits > 0`
try:
    import torchquantum as tq
    import torchquantum.functional as tqf
except Exception:  # pragma: no cover
    tq = None
    tqf = None


class _QuantumGateModule(nn.Module):
    """Small variational quantum circuit producing a vector of length `n_qubits`."""
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        if tq:
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)
        else:
            self.encoder = None
            self.params = None
            self.measure = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_qubits)
        returns: (batch, n_qubits)
        """
        if not tq:
            # Fallback: identity mapping
            return x
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for gate in self.params:
            gate(qdev)
        return self.measure(qdev)


class HybridAttention(nn.Module):
    """Self‑attention with optional quantum augmentation."""
    def __init__(self, embed_dim: int, n_heads: int = 1, use_quantum: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.use_quantum = use_quantum
        self.d_k = embed_dim // n_heads
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        if use_quantum and tq:
            self.q_gate = _QuantumGateModule(embed_dim)
        else:
            self.q_gate = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (seq_len, batch, embed_dim)
        returns: (seq_len, batch, embed_dim)
        """
        batch_size = x.size(1)
        seq_len = x.size(0)
        q = self.linear_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim).transpose(0, 1)

        if self.use_quantum and self.q_gate:
            out = []
            for token in attn_output.unbind(dim=0):
                token = token.squeeze(0)  # (batch, embed_dim)
                mod = self.q_gate(token)
                out.append(mod.unsqueeze(0))
            attn_output = torch.cat(out, dim=0)
        return attn_output


class HybridQLSTM(nn.Module):
    """Hybrid LSTM cell with classical or quantum gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0, n_attention_heads: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.n_attention_heads = n_attention_heads

        # Classical linear gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Quantum gate modules
        if n_qubits > 0:
            self.forget_gate = _QuantumGateModule(n_qubits)
            self.input_gate = _QuantumGateModule(n_qubits)
            self.update_gate = _QuantumGateModule(n_qubits)
            self.output_gate = _QuantumGateModule(n_qubits)
        else:
            self.forget_gate = None

        # Decoder
        self.decoder = HybridAttention(embed_dim=hidden_dim,
                                       n_heads=n_attention_heads,
                                       use_quantum=(n_qubits > 0))

    def _init_states(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

    def _gate(self, combined: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.forget_gate is not None:
            f = torch.sigmoid(self.forget_gate(combined[:, :self.n_qubits]))
            i = torch.sigmoid(self.input_gate(combined[:, :self.n_qubits]))
            g = torch.tanh(self.update_gate(combined[:, :self.n_qubits]))
            o = torch.sigmoid(self.output_gate(combined[:, :self.n_qubits]))
        else:
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
        return f, i, g, o

    def forward(self, inputs: torch.Tensor, states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        inputs: (seq_len, batch, input_dim)
        returns: (seq_len, batch, hidden_dim), (hx, cx)
        """
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f, i, g, o = self._gate(combined)
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        lstm_out = torch.cat(outputs, dim=0)
        decoded = self.decoder(lstm_out)
        return decoded, (hx, cx)


class HybridQLSTMTagger(nn.Module):
    """Sequence tagging model using the hybrid LSTM and optional quantum attention."""
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0,
                 n_attention_heads: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, n_attention_heads=n_attention_heads)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        sentence: (seq_len, batch)
        returns log‑softmaxed tag logits
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)


__all__ = ["HybridQLSTM", "HybridQLSTMTagger"]
