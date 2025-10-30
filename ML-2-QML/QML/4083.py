"""Hybrid Self‑Attention Transformer – quantum‑centric implementation.

This module mirrors the classical version but replaces the
self‑attention, LSTM gates, and transformer blocks with
variational quantum circuits implemented via torchquantum.
It can be executed on simulators or quantum hardware.
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
# Quantum self‑attention helper
# --------------------------------------------------------------------------- #
class QuantumSelfAttention(nn.Module):
    """Self‑attention block where query/key/value projections are quantum‑derived."""

    class _QLayer(tq.QuantumModule):
        """Variational circuit that produces projection vectors."""
        def __init__(self, n_wires: int):
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

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for i, gate in enumerate(self.params):
                gate(qdev, wires=i)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, n_wires: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_wires = n_wires
        self.q_layer = self._QLayer(n_wires)

    def forward(
        self,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        # Project inputs through quantum circuits
        batch = inputs.size(0)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=batch, device=inputs.device)
        proj = self.q_layer(inputs, qdev)
        query = torch.matmul(proj, rotation_params)
        key = torch.matmul(proj, entangle_params)
        scores = F.softmax(query @ key.t() / math.sqrt(self.embed_dim), dim=-1)
        return scores @ proj


# --------------------------------------------------------------------------- #
# Quantum LSTM cell
# --------------------------------------------------------------------------- #
class QuantumQLSTM(nn.Module):
    """LSTM cell where each gate is realised by a small quantum module."""

    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
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

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for i, gate in enumerate(self.params):
                gate(qdev, wires=i)
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        # Linear projections into quantum sub‑space
        self.forget_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.input_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.update_proj = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.output_proj = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum circuits for gates
        self.forget_gate = self._QLayer(n_qubits)
        self.input_gate  = self._QLayer(n_qubits)
        self.update_gate = self._QLayer(n_qubits)
        self.output_gate = self._QLayer(n_qubits)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch, self.hidden_dim, device=device),
            torch.zeros(batch, self.hidden_dim, device=device),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            qdev = tq.QuantumDevice(n_wires=self.n_qubits,
                                    bsz=combined.size(0),
                                    device=combined.device)
            f = torch.sigmoid(
                self.forget_gate(
                    self.forget_proj(combined), qdev
                )
            )
            i = torch.sigmoid(
                self.input_gate(
                    self.input_proj(combined), qdev
                )
            )
            g = torch.tanh(
                self.update_gate(
                    self.update_proj(combined), qdev
                )
            )
            o = torch.sigmoid(
                self.output_gate(
                    self.output_proj(combined), qdev
                )
            )
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)


# --------------------------------------------------------------------------- #
# Quantum transformer components
# --------------------------------------------------------------------------- #
class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Multi‑head attention with quantum‑derived projections."""

    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
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

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for i, gate in enumerate(self.params):
                gate(qdev, wires=i)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1, n_wires: int = 8):
        super().__init__(embed_dim, num_heads, dropout)
        self.n_wires = n_wires
        self.q_layer = self._QLayer(n_wires)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def _apply_quantum_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        seq_len = x.size(1)
        proj = torch.zeros(batch, seq_len, self.n_wires, device=x.device)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=batch, device=x.device)
        for i in range(seq_len):
            proj[:, i, :] = self.q_layer(x[:, i, :], qdev)
        return proj

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch = x.size(0)
        q = self._apply_quantum_heads(x)
        k = self._apply_quantum_heads(x)
        v = self._apply_quantum_heads(x)
        out = self.downstream(q, k, v, batch, mask)
        return self.combine_heads(out)


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network realised by a quantum module."""

    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "ry", "wires": [i]}
                    for i in range(n_wires)
                ]
            )
            self.params = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for i, gate in enumerate(self.params):
                gate(qdev, wires=i)
            return self.measure(qdev)

    def __init__(self, embed_dim: int, ffn_dim: int,
                 n_qubits: int = 8, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.n_qubits = n_qubits
        self.q_layer = self._QLayer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=batch, device=x.device)
        outputs = []
        for i in range(seq_len):
            outputs.append(self.q_layer(x[:, i, :], qdev))
        q_out = torch.stack(outputs, dim=1)
        q_out = self.linear1(self.dropout(q_out))
        return self.linear2(F.relu(q_out))


class TransformerBlockQuantum(TransformerBlockBase):
    """Transformer block that uses quantum attention and feed‑forward."""

    def __init__(self, embed_dim: int, num_heads: int,
                 ffn_dim: int, n_qubits: int = 8,
                 dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads,
                                              dropout, n_wires=n_qubits)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim,
                                      n_qubits=n_qubits, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
# HybridSelfAttentionTransformer – main class (quantum version)
# --------------------------------------------------------------------------- #
class HybridSelfAttentionTransformer(nn.Module):
    """Hybrid transformer that replaces classical sub‑modules with quantum ones.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary.
    embed_dim : int
        Dimension of token embeddings.
    num_heads : int
        Number of attention heads in each transformer block.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Hidden dimension of the feed‑forward network.
    hidden_dim : int
        Hidden dimension of the recurrent unit.
    num_classes : int
        Number of output classes.
    dropout : float, optional
        Drop‑out probability.
    n_qubits : int, optional
        Number of qubits per quantum module.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        n_qubits: int = 8,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.self_attention = QuantumSelfAttention(embed_dim, n_wires=n_qubits)

        # Recurrent component
        self.lstm = QuantumQLSTM(embed_dim, hidden_dim, n_qubits=n_qubits)

        # Transformer backbone
        self.transformer = nn.Sequential(
            *[
                TransformerBlockQuantum(
                    embed_dim, num_heads, ffn_dim, n_qubits=n_qubits, dropout=dropout
                )
                for _ in range(num_blocks)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input token indices of shape (seq_len, batch).

        Returns
        -------
        torch.Tensor
            Log‑probabilities for each class.
        """
        tokens = self.token_embedding(x.t())  # (batch, seq_len, embed_dim)
        tokens = self.pos_encoder(tokens)

        # Self‑attention block
        seq_len, batch, embed_dim = tokens.size(1), tokens.size(0), tokens.size(2)
        inputs = tokens.reshape(-1, embed_dim)
        rotation_params = torch.eye(embed_dim, device=inputs.device)
        entangle_params = torch.eye(embed_dim, device=inputs.device)
        attn_out = self.self_attention(rotation_params, entangle_params, inputs)
        attn_out = attn_out.reshape(batch, seq_len, embed_dim)

        # LSTM
        lstm_out, _ = self.lstm(attn_out.permute(1, 0, 2))  # (seq_len, batch, embed_dim)

        # Transformer
        x = lstm_out.permute(1, 0, 2)  # (batch, seq_len, embed_dim)
        x = self.transformer(x)
        x = self.dropout(x.mean(dim=1))
        logits = self.classifier(x)
        return F.log_softmax(logits, dim=1)
