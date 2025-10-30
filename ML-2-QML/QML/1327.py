"""Quantum-enhanced transformer with TorchQuantum."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class MultiHeadAttentionBase(nn.Module):
    """Base class for attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_bias: bool = True) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.use_bias = use_bias

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        return F.softmax(scores, dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with linear layers."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, use_bias: bool = True) -> None:
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        k = self.k_proj(x)
        q = self.q_proj(x)
        v = self.v_proj(x)
        qk = self.separate_heads(q)
        kv = self.separate_heads(k)
        v = self.separate_heads(v)
        scores = self.attention(qk, kv, v, mask)
        attn = torch.matmul(scores, v)
        attn = attn.transpose(1, 2).reshape(batch, seq, self.embed_dim)
        return self.out_proj(attn)


class QuantumMultiHeadAttention(nn.Module):
    """Multi‑head attention where each head is a small quantum circuit."""
    class QHead(tq.QuantumModule):
        def __init__(self, n_wires: int, depth: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.depth = depth
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for _ in range(self.depth):
                for wire in range(self.n_wires - 1):
                    tqf.cnot(qdev, wires=[wire, wire + 1])
                tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 depth: int = 1, quantum_device: Optional[tq.QuantumDevice] = None,
                 use_bias: bool = True) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.use_bias = use_bias
        self.depth = depth
        self.q_device = quantum_device
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.heads = nn.ModuleList([self.QHead(self.d_k, depth) for _ in range(num_heads)])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        k = self.k_proj(x)
        q = self.q_proj(x)
        v = self.v_proj(x)
        qk = self.separate_heads(q)
        kv = self.separate_heads(k)
        v = self.separate_heads(v)
        qh = torch.zeros(batch, self.num_heads, seq, self.d_k, device=x.device)
        kh = torch.zeros(batch, self.num_heads, seq, self.d_k, device=x.device)
        vh = torch.zeros(batch, self.num_heads, seq, self.d_k, device=x.device)
        for h_idx, head in enumerate(self.heads):
            for b in range(batch):
                for t in range(seq):
                    qdev = self.q_device or tq.QuantumDevice(n_wires=self.d_k, bsz=1, device=x.device)
                    qh[b, h_idx, t] = head(qk[b, h_idx, t], qdev)
                    kh[b, h_idx, t] = head(kv[b, h_idx, t], qdev)
                    vh[b, h_idx, t] = head(v[b, h_idx, t], qdev)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        attn = torch.matmul(scores, vh)
        attn = attn.transpose(1, 2).reshape(batch, seq, self.embed_dim)
        return self.out_proj(attn)


class FeedForwardBase(nn.Module):
    """Base feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Standard feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class QuantumFeedForward(nn.Module):
    """Feed‑forward network realized by a quantum module."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int, depth: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.depth = depth
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for _ in range(self.depth):
                for wire in range(self.n_wires - 1):
                    tqf.cnot(qdev, wires=[wire, wire + 1])
                tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int,
                 dropout: float = 0.1, depth: int = 1, quantum_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)
        self.n_qubits = n_qubits
        self.depth = depth
        self.q_device = quantum_device
        self.q_layer = self.QLayer(n_qubits, depth)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        outputs = []
        for b in range(batch):
            for t in range(seq):
                qdev = self.q_device or tq.QuantumDevice(n_wires=self.n_qubits, bsz=1, device=x.device)
                q_out = self.q_layer(x[b, t], qdev)
                outputs.append(q_out)
        out = torch.stack(outputs, dim=0).reshape(batch, seq, self.n_qubits)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockBase(nn.Module):
    """Base transformer block."""
    def __init__(self, embed_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    """Transformer block with classical attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockHybrid(TransformerBlockBase):
    """Transformer block that can switch between classical and quantum sub‑modules."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 use_quantum: bool = False, quantum_depth: int = 1,
                 dropout: float = 0.1, n_qubits_attn: int = 8, n_qubits_ffn: int = 8,
                 quantum_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__(embed_dim, dropout)
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.use_quantum = use_quantum
        self.quantum_depth = quantum_depth
        if use_quantum:
            self.attn = QuantumMultiHeadAttention(embed_dim, num_heads, dropout,
                                                  depth=quantum_depth,
                                                  quantum_device=quantum_device)
            self.ffn = QuantumFeedForward(embed_dim, ffn_dim, n_qubits_ffn,
                                          dropout=dropout, depth=quantum_depth,
                                          quantum_device=quantum_device)
        else:
            self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def set_mode(self, use_quantum: bool, quantum_depth: Optional[int] = None) -> None:
        if use_quantum!= self.use_quantum:
            self.use_quantum = use_quantum
            depth = quantum_depth or self.quantum_depth
            if use_quantum:
                self.attn = QuantumMultiHeadAttention(self.embed_dim, self.attn.num_heads,
                                                      dropout=self.dropout,
                                                      depth=depth)
                self.ffn = QuantumFeedForward(self.embed_dim, self.ffn_dim,
                                              n_qubits_ffn=8,
                                              dropout=self.dropout,
                                              depth=depth)
            else:
                self.attn = MultiHeadAttentionClassical(self.embed_dim, self.attn.num_heads,
                                                        dropout=self.dropout)
                self.ffn = FeedForwardClassical(self.embed_dim, self.ffn_dim,
                                                dropout=self.dropout)

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
        return x + self.pe[:, :x.size(1)]


class QuantumTransformerHybrid(nn.Module):
    """Transformer‑based text classifier that can use quantum sub‑modules."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_quantum: bool = False,
        quantum_depth: int = 1,
        n_qubits_attn: int = 8,
        n_qubits_ffn: int = 8,
        quantum_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlockHybrid(
                embed_dim,
                num_heads,
                ffn_dim,
                use_quantum=use_quantum,
                quantum_depth=quantum_depth,
                dropout=dropout,
                n_qubits_attn=n_qubits_attn,
                n_qubits_ffn=n_qubits_ffn,
                quantum_device=quantum_device,
            )
            for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def set_mode(self, use_quantum: bool, quantum_depth: Optional[int] = None) -> None:
        for block in self.blocks:
            block.set_mode(use_quantum, quantum_depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "QuantumMultiHeadAttention",
    "FeedForwardBase",
    "FeedForwardClassical",
    "QuantumFeedForward",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "TransformerBlockHybrid",
    "PositionalEncoder",
    "QuantumTransformerHybrid",
]
