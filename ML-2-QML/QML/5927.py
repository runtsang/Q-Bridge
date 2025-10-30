"""Quantum‑enhanced LSTM‑Transformer implementation.

This module implements the same public API as the classical version but
uses TorchQuantum to replace LSTM gates and transformer sub‑modules.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# Classical LSTM cell (fallback when quantum flag is off)
# --------------------------------------------------------------------------- #
class ClassicalQLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(self, inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

# --------------------------------------------------------------------------- #
# Quantum‑augmented LSTM gate (from reference 1)
# --------------------------------------------------------------------------- #
class QuantumGateQLSTM(nn.Module):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
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
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(self, inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None
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
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

# --------------------------------------------------------------------------- #
# Quantum attention and feed‑forward (from reference 2)
# --------------------------------------------------------------------------- #
class MultiHeadAttentionQuantum(nn.Module):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 8) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer()
        self.q_device = q_device
        self.combine_heads = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        if embed_dim!= self.embed_dim:
            raise ValueError("Input embedding does not match layer size")
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)

        def separate_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch_size, seq_len, self.num_heads,
                          self.embed_dim // self.num_heads).transpose(1, 2)

        k = separate_heads(k)
        q = separate_heads(q)
        v = separate_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim // self.num_heads) ** 0.5
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        out = torch.matmul(scores, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        qdev = self.q_device or tq.QuantumDevice(n_wires=self.q_layer.n_wires)
        out = self.q_layer(out, qdev)
        return self.combine_heads(out)

class FeedForwardQuantum(nn.Module):
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_qubits)]
            )
            self.params = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            return self.measure(qdev)

    def __init__(self,
                 embed_dim: int,
                 ffn_dim: int,
                 n_qubits: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer(n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.dropout(F.relu(x))
        qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires)
        x = self.q_layer(x, qdev)
        return self.linear2(x)

class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockQuantum(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_qubits_transformer: int,
                 n_qubits_ffn: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads,
                                               dropout)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim,
                                          n_qubits_ffn, dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# --------------------------------------------------------------------------- #
# Positional encoding (identical to classical)
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# --------------------------------------------------------------------------- #
# Hybrid model (quantum‑enabled) – same public API as classical
# --------------------------------------------------------------------------- #
class QuantumEnhancedLSTMTransformer(nn.Module):
    """
    Hybrid LSTM‑Transformer that uses quantum‑augmented LSTM gates and
    transformer blocks when the corresponding flags are set.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 tagset_size: int,
                 dropout: float = 0.1,
                 use_quantum_lstm: bool = False,
                 use_quantum_transformer: bool = False,
                 n_qubits_lstm: int = 4,
                 n_qubits_transformer: int = 8,
                 n_qubits_ffn: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if use_quantum_lstm:
            self.lstm = QuantumGateQLSTM(embed_dim, hidden_dim, n_qubits_lstm)
        else:
            self.lstm = ClassicalQLSTM(embed_dim, hidden_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        if use_quantum_transformer:
            self.transformer_blocks = nn.ModuleList(
                [TransformerBlockQuantum(embed_dim, num_heads,
                                         ffn_dim, n_qubits_transformer,
                                         n_qubits_ffn, dropout)
                 for _ in range(num_blocks)])
        else:
            self.transformer_blocks = nn.ModuleList(
                [TransformerBlockClassical(embed_dim, num_heads,
                                           ffn_dim, dropout)
                 for _ in range(num_blocks)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, tagset_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.embedding(x)
        lstm_out, _ = self.lstm(tokens)
        x = self.pos_encoder(lstm_out)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

__all__ = ["QuantumEnhancedLSTMTransformer"]
