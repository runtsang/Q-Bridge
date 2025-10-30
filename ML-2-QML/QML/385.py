"""Quantum‑enhanced LSTM with variational gates and optional quantum attention.

This module mirrors the classical implementation but replaces the
gate projections with small variational quantum circuits.  It also
provides an optional quantum attention layer that produces soft
attention weights from a variational circuit, and a multi‑task
head similar to the classical version.

The public API matches the original `QLSTM` and `LSTMTagger`
classes so that existing code can be switched to the quantum
implementation by setting `n_qubits > 0` and the new flags.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QLayer(tq.QuantumModule):
    """Variational quantum layer used for the LSTM gates."""

    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Simple encoder that applies RX rotations to each wire
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        return self.measure(qdev)


class QAttention(tq.QuantumModule):
    """Quantum attention layer that outputs a scalar weight per time step."""

    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [i], "func": "rx", "wires": [i]}
                for i in range(n_wires)
            ]
        )
        self.params = nn.ModuleList(
            [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, n_wires)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
        self.encoder(qdev, x)
        for wire, gate in enumerate(self.params):
            gate(qdev, wires=wire)
        return self.measure(qdev)  # (seq_len, n_wires)


class QLSTM(nn.Module):
    """Quantum‑enhanced LSTM cell."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Use the same number of qubits for gates and attention
        self.forget = QLayer(n_qubits)
        self.input = QLayer(n_qubits)
        self.update = QLayer(n_qubits)
        self.output = QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        self.attention = QAttention(n_qubits)

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
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)

        # Quantum attention over the hidden states
        attn_logits = self.attention(outputs)  # (seq_len, n_qubits)
        # Collapse to a scalar per time step (e.g., sum over wires)
        attn_logits = attn_logits.sum(dim=-1)  # (seq_len,)
        attn_weights = F.softmax(attn_logits, dim=0)
        context = torch.sum(attn_weights.unsqueeze(-1) * outputs, dim=0)
        outputs = context.unsqueeze(0).repeat(outputs.size(0), 1)

        return outputs, (hx, cx)


class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between classical and
    quantum LSTM and supports optional multi‑task heads."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        multi_task: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            # Fallback to a classical LSTM if no qubits are requested
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        if multi_task:
            self.hidden2chunk = nn.Linear(hidden_dim, tagset_size)

        self.multi_task = multi_task

    def forward(self, sentence: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        embeds = self.word_embeddings(sentence)
        if isinstance(self.lstm, nn.LSTM):
            lstm_out, _ = self.lstm(embeds.unsqueeze(0))
            lstm_out = lstm_out.squeeze(0)
        else:
            lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out)
        if self.multi_task:
            chunk_logits = self.hidden2chunk(lstm_out)
            return F.log_softmax(tag_logits, dim=1), F.log_softmax(chunk_logits, dim=1)
        else:
            return F.log_softmax(tag_logits, dim=1)


__all__ = ["QLSTM", "LSTMTagger"]
