import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class MultiHeadAttentionBase(nn.Module):
    """Base class for attention variants."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.d_k = embed_dim // num_heads

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, l, e = x.size()
        return x.view(b, l, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionHybrid(MultiHeadAttentionBase):
    """Classical dot‑product attention with an optional quantum‑style projection."""
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1, use_quantum: bool = False,
                 n_qubits: Optional[int] = None, q_device: Optional[tq.QuantumDevice] = None):
        super().__init__(embed_dim, num_heads, dropout)
        self.use_quantum = use_quantum
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=False)
        if self.use_quantum:
            self.n_qubits = n_qubits if n_qubits is not None else self.d_k
            self.q_layer = self._build_qlayer()
            self.q_device = q_device or tq.QuantumDevice(n_wires=self.n_qubits)
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _build_qlayer(self):
        class QLayer(tq.QuantumModule):
            def __init__(self, n_qubits):
                super().__init__()
                self.n_qubits = n_qubits
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
                )
                self.parameters = nn.ModuleList(
                    [tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)]
                )
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
                self.encoder(q_device, x)
                for wire, gate in enumerate(self.parameters):
                    gate(q_device, wires=wire)
                for wire in range(self.n_qubits - 1):
                    tqf.cnot(q_device, wires=[wire, wire + 1])
                tqf.cnot(q_device, wires=[self.n_qubits - 1, 0])
                return self.measure(q_device)
        return QLayer(self.n_qubits)

    def _apply_quantum_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, heads, d_k)
        batch, seq, heads, d_k = x.size()
        projections = []
        for token in x.unbind(dim=1):  # per sequence position
            head_outputs = []
            for head in token.unbind(dim=1):  # per head
                qdev = self.q_device.copy(bsz=head.size(0))
                head_out = self.q_layer(head, qdev)
                head_outputs.append(head_out)
            projections.append(torch.stack(head_outputs, dim=1))
        return torch.stack(projections, dim=1)  # (batch, seq, heads, n_qubits)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)

        if self.use_quantum:
            q = self._apply_quantum_heads(q)
            k = self._apply_quantum_heads(k)
            v = self._apply_quantum_heads(v)
            # reshape back to (batch, seq, embed_dim)
            q = q.view(batch_size, seq_len, self.embed_dim)
            k = k.view(batch_size, seq_len, self.embed_dim)
            v = v.view(batch_size, seq_len, self.embed_dim)

        attn_out = self.attention(q, k, v, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        attn_out = self.combine_heads(attn_out)

        if self.use_quantum:
            attn_out = self.q_proj(attn_out)

        return attn_out


class FeedForwardBase(nn.Module):
    """Base feed‑forward module."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardHybrid(FeedForwardBase):
    """Two‑layer feed‑forward network with optional quantum‑style projection."""
    def __init__(self, embed_dim: int, ffn_dim: int,
                 dropout: float = 0.1, use_quantum: bool = False,
                 n_qubits: Optional[int] = None, q_device: Optional[tq.QuantumDevice] = None):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.use_quantum = use_quantum
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)
        if self.use_quantum:
            self.n_qubits = n_qubits if n_qubits is not None else ffn_dim
            self.q_layer = self._build_qlayer()
            self.q_device = q_device or tq.QuantumDevice(n_wires=self.n_qubits)
            self.q_proj = nn.Linear(self.n_qubits, ffn_dim, bias=False)

    def _build_qlayer(self):
        class QLayer(tq.QuantumModule):
            def __init__(self, n_qubits):
                super().__init__()
                self.n_qubits = n_qubits
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_qubits)]
                )
                self.parameters = nn.ModuleList(
                    [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
                )
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
                self.encoder(q_device, x)
                for wire, gate in enumerate(self.parameters):
                    gate(q_device, wires=wire)
                return self.measure(q_device)
        return QLayer(self.n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear1(x)
        out = F.relu(out)
        out = self.dropout(out)
        if self.use_quantum:
            # apply quantum projection per token
            batch, seq, dim = out.size()
            q_outputs = []
            for i in range(batch):
                token_outputs = []
                for j in range(seq):
                    qdev = self.q_device.copy(bsz=1)
                    token_outputs.append(self.q_layer(out[i, j, :].unsqueeze(0), qdev))
                q_outputs.append(torch.stack(token_outputs, dim=0))
            q_out = torch.stack(q_outputs, dim=0)  # (batch, seq, n_qubits)
            out = self.q_proj(q_out)
        out = self.linear2(out)
        return out


class TransformerBlockBase(nn.Module):
    """Base transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockHybrid(TransformerBlockBase):
    """Hybrid transformer block using the attention and feed‑forward variants."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, use_quantum: bool = False,
                 n_qubits: Optional[int] = None, q_device: Optional[tq.QuantumDevice] = None):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionHybrid(embed_dim, num_heads,
                                             dropout=dropout,
                                             use_quantum=use_quantum,
                                             n_qubits=n_qubits,
                                             q_device=q_device)
        self.ffn = FeedForwardHybrid(embed_dim, ffn_dim,
                                     dropout=dropout,
                                     use_quantum=use_quantum,
                                     n_qubits=n_qubits,
                                     q_device=q_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TextClassifierHybrid(nn.Module):
    """Transformer‑based text classifier supporting optional quantum sub‑modules."""
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_quantum: bool = False,
                 n_qubits: Optional[int] = None,
                 q_device: Optional[tq.QuantumDevice] = None):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlockHybrid(embed_dim, num_heads, ffn_dim,
                                   dropout=dropout,
                                   use_quantum=use_quantum,
                                   n_qubits=n_qubits,
                                   q_device=q_device)
            for _ in range(num_blocks)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

    def quantum_regularization(self, x: torch.Tensor) -> torch.Tensor:
        """Computes an expectation‑based regularizer over the last block's quantum circuit."""
        if not hasattr(self, 'blocks') or len(self.blocks) == 0:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        last_block = self.blocks[-1]
        if not getattr(last_block.attn, 'use_quantum', False):
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        # sample a few tokens to compute expectation
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        for block in self.blocks[:-1]:
            x = block(x)
        attn_out = last_block.attn(x)
        # compute mean absolute value as a simple penalty
        return torch.mean(torch.abs(attn_out))


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionHybrid",
    "FeedForwardBase",
    "FeedForwardHybrid",
    "TransformerBlockBase",
    "TransformerBlockHybrid",
    "PositionalEncoder",
    "TextClassifierHybrid",
]
