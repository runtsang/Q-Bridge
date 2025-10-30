"""UnifiedSelfAttentionTransformer – Quantum implementation using TorchQuantum.

This module mirrors the classical ``UnifiedSelfAttentionTransformer`` but
replaces the attention and feed‑forward sub‑components with variational
quantum circuits implemented via TorchQuantum.  The API is identical,
so the same downstream training loop can be used for both
classical and quantum experiments.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# Quantum self‑attention – a lightweight variational circuit.
# --------------------------------------------------------------------------- #
class QuantumSelfAttention(tq.QuantumModule):
    """Variational self‑attention block implemented with TorchQuantum.

    The circuit encodes each token vector as rotation angles on a set of
    qubits, entangles neighbouring wires with CNOT gates, and measures
    Pauli‑Z.  The measurement outcomes are interpreted as a continuous
    attention vector with the same dimensionality as the input embedding.
    """
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.params = nn.ParameterList(
            [nn.Parameter(torch.rand(1) * 2 * math.pi) for _ in range(n_qubits)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        # x shape: (batch, seq_len, embed_dim)
        batch, seq_len, embed_dim = x.shape
        flat = x.view(batch * seq_len, embed_dim)
        self.encoder(q_device, flat)
        for idx, par in enumerate(self.params):
            tq.RX(par, wires=idx)(q_device)
        # entangle with a simple chain of CNOTs
        for i in range(self.n_qubits - 1):
            tqf.cnot(q_device, wires=[i, i + 1])
        out = self.measure(q_device)
        return out.view(batch, seq_len, embed_dim)

# --------------------------------------------------------------------------- #
# Quantum feed‑forward – a two‑layer variational circuit.
# --------------------------------------------------------------------------- #
class QuantumFeedForward(tq.QuantumModule):
    """Two‑layer variational feed‑forward network realised by a parameterised
    circuit followed by a classical linear layer.
    """
    def __init__(self, n_qubits: int, output_dim: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.params = nn.ParameterList(
            [nn.Parameter(torch.rand(1) * 2 * math.pi) for _ in range(n_qubits)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear = nn.Linear(n_qubits, output_dim, bias=False)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        # x shape: (batch, seq_len, embed_dim)
        batch, seq_len, embed_dim = x.shape
        flat = x.view(batch * seq_len, embed_dim)
        self.encoder(q_device, flat)
        for idx, par in enumerate(self.params):
            tq.RX(par, wires=idx)(q_device)
        out = self.measure(q_device)
        out = out.view(batch, seq_len, self.n_qubits)
        return self.linear(out)

# --------------------------------------------------------------------------- #
# Quantum transformer block – combines the two quantum modules.
# --------------------------------------------------------------------------- #
class QuantumTransformerBlock(nn.Module):
    """Residual transformer block using quantum attention and feed‑forward."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_attn: int,
        n_qubits_ffn: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.attn = QuantumSelfAttention(n_qubits_attn)
        self.ffn = QuantumFeedForward(n_qubits_ffn, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, embed_dim = x.shape
        # quantum attention
        q_device_attn = tq.QuantumDevice(n_wires=self.attn.n_qubits, bsz=batch * seq_len, device=x.device)
        attn_out = self.attn(x, q_device_attn)  # shape: (batch, seq_len, embed_dim)
        x = self.norm1(x + self.dropout(attn_out))

        # quantum feed‑forward
        q_device_ffn = tq.QuantumDevice(n_wires=self.ffn.n_qubits, bsz=batch * seq_len, device=x.device)
        ffn_out = self.ffn(x, q_device_ffn)  # shape: (batch, seq_len, ffn_dim)
        ffn_out = self.ffn_out(ffn_out)  # map back to embed_dim
        x = self.norm2(x + self.dropout(ffn_out))
        return x

# --------------------------------------------------------------------------- #
# Positional encoder – identical to the classical version.
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

# --------------------------------------------------------------------------- #
# Quantum UnifiedSelfAttentionTransformer – drop‑in replacement.
# --------------------------------------------------------------------------- #
class UnifiedSelfAttentionTransformer(nn.Module):
    """Quantum‑enhanced transformer‑style classifier.

    The architecture mirrors the classical version but all attention and
    feed‑forward sub‑components are realised by variational circuits.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        n_qubits_attn: int = 8,
        n_qubits_ffn: int = 8,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[
                QuantumTransformerBlock(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_qubits_attn,
                    n_qubits_ffn,
                    dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

__all__ = ["UnifiedSelfAttentionTransformer"]
