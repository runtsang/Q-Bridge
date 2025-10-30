"""Quantum‑enhanced Transformer implementation using torchquantum.

This module provides the same public API as the classical module but
replaces the attention and feed‑forward sub‑modules with quantum
circuit‑based alternatives.  The quantum blocks consist of a shallow
parameterised circuit that can be trained end‑to‑end with the rest of
the model.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class MultiHeadAttentionBase(nn.Module):
    """Base class for multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 mask: Optional[torch.Tensor] = None, use_bias: bool = False) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.attn_weights: Optional[torch.Tensor] = None

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores

    def downstream(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                   batch_size: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.separate_heads(query)
        k = self.separate_heads(key)
        v = self.separate_heads(value)
        out, self.attn_weights = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented using PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 mask: Optional[torch.Tensor] = None, use_bias: bool = False) -> None:
        super().__init__(embed_dim, num_heads, dropout, mask, use_bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        batch_size, seq_len, embed_dim = x.shape
        if embed_dim!= self.embed_dim:
            raise ValueError(f"Input embedding {embed_dim} does not match layer embedding size {self.embed_dim}")
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        k = k.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.combine_heads(attn_output)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum‑encoded multi‑head attention."""
    class _QLayer(tq.QuantumModule):
        """Quantum circuit that operates on a single head."""
        def __init__(self, n_wires: int, n_layers: int = 1):
            super().__init__()
            self.n_wires = n_wires
            self.n_layers = n_layers
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ParameterList(
                [nn.Parameter(torch.randn(n_wires)) for _ in range(n_layers)]
            )
            self.cnot_pattern = [(i, (i + 1) % n_wires) for i in range(n_wires)]
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, input_vec: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, input_vec)
            for layer_idx in range(self.n_layers):
                param = self.params[layer_idx]
                for wire in range(self.n_wires):
                    tqf.rx(q_device, wires=[wire], params=param[wire])
                for src, tgt in self.cnot_pattern:
                    tqf.cnot(q_device, wires=[src, tgt])
            return self.measure(q_device)

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 mask: Optional[torch.Tensor] = None,
                 use_bias: bool = False,
                 n_wires_per_head: int = 4,
                 n_layers: int = 1):
        super().__init__(embed_dim, num_heads, dropout, mask, use_bias)
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.d_k = embed_dim // num_heads
        if self.d_k!= n_wires_per_head:
            raise ValueError("d_k must equal n_wires_per_head for quantum attention")
        self.n_wires_per_head = n_wires_per_head
        self.n_layers = n_layers
        self.q_layer = self._QLayer(n_wires_per_head, n_layers)
        self.q_device = tq.QuantumDevice(n_wires=n_wires_per_head, bsz=0, device="cpu")
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def _apply_quantum_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        token_heads = x.view(batch, seq_len, self.num_heads, -1)
        outputs = []
        for token_idx in range(seq_len):
            heads = token_heads[:, token_idx, :, :]  # (batch, num_heads, d_k)
            heads_flat = heads.reshape(batch * self.num_heads, self.d_k)
            qdev = self.q_device.copy(bsz=batch, device=heads_flat.device)
            quantum_out = self.q_layer(heads_flat, qdev)
            quantum_out = quantum_out.reshape(batch, self.num_heads, self.d_k)
            outputs.append(quantum_out)
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        if embed_dim!= self.embed_dim:
            raise ValueError(f"Input embedding {embed_dim} does not match layer embedding size {self.embed_dim}")
        k = self._apply_quantum_heads(x)
        q = self._apply_quantum_heads(x)
        v = self._apply_quantum_heads(x)
        out = self.downstream(q, k, v, batch_size, mask)
        return self.combine_heads(out)


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward networks."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network realized by a quantum circuit."""
    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int, n_layers: int = 1):
            super().__init__()
            self.n_wires = n_wires
            self.n_layers = n_layers
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ParameterList(
                [nn.Parameter(torch.randn(n_wires)) for _ in range(n_layers)]
            )
            self.cnot_pattern = [(i, (i + 1) % n_wires) for i in range(n_wires)]
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, input_vec: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, input_vec)
            for layer_idx in range(self.n_layers):
                param = self.params[layer_idx]
                for wire in range(self.n_wires):
                    tqf.rx(q_device, wires=[wire], params=param[wire])
                for src, tgt in self.cnot_pattern:
                    tqf.cnot(q_device, wires=[src, tgt])
            return self.measure(q_device)

    def __init__(self,
                 embed_dim: int,
                 ffn_dim: int,
                 n_qubits: int,
                 dropout: float = 0.1,
                 n_layers: int = 1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.q_layer = self._QLayer(n_qubits, n_layers)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits, bsz=0, device="cpu")
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            out = self.q_layer(token, qdev)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)  # (batch, seq_len, n_qubits)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockBase(nn.Module):
    """Base transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    """Standard transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_wires_per_head: int,
                 n_qubits_ffn: int,
                 n_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(
            embed_dim,
            num_heads,
            dropout,
            n_wires_per_head=n_wires_per_head,
            n_layers=n_layers
        )
        self.ffn = FeedForwardQuantum(
            embed_dim,
            ffn_dim,
            n_qubits=n_qubits_ffn,
            dropout=dropout,
            n_layers=n_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class QuantumEnhancedTransformer(nn.Module):
    """Transformer‑based classifier that can optionally use quantum sub‑modules.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embed_dim : int
        Dimension of token embeddings.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Hidden dimension of the feed‑forward network.
    num_classes : int
        Number of output classes.
    dropout : float, default 0.1
        Dropout probability.
    use_quantum_attention : bool, default False
        If True, use the quantum attention module.
    use_quantum_ffn : bool, default False
        If True, use the quantum feed‑forward module.
    n_wires_per_head : int, default 4
        Number of qubits per attention head.
    n_qubits_ffn : int, default 8
        Number of qubits in the feed‑forward quantum circuit.
    n_layers : int, default 1
        Depth of the quantum circuits.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_quantum_attention: bool = False,
                 use_quantum_ffn: bool = False,
                 n_wires_per_head: int = 4,
                 n_qubits_ffn: int = 8,
                 n_layers: int = 1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        blocks = []
        for _ in range(num_blocks):
            if use_quantum_attention or use_quantum_ffn:
                blocks.append(
                    TransformerBlockQuantum(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        n_wires_per_head=n_wires_per_head,
                        n_qubits_ffn=n_qubits_ffn,
                        n_layers=n_layers,
                        dropout=dropout
                    )
                )
            else:
                blocks.append(TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout))
        self.transformer_blocks = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_encoder(tokens)
        x = self.transformer_blocks(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardClassical",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "QuantumEnhancedTransformer",
]
