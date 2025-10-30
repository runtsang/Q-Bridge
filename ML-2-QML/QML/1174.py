import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class MultiHeadAttentionBase(nn.Module):
    """Base class that implements the head‑splitting logic used by both
    classical and quantum variants.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, T, E) → (B, H, T, d_k)."""
        return x.view(x.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, H, T, d_k) → (B, T, E)."""
        return x.transpose(1, 2).contiguous().view(x.size(0), -1, self.embed_dim)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        return F.softmax(scores, dim=-1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention using linear projections."""
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1, bias: bool = True) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, _ = x.shape
        k = self.k_proj(x)
        q = self.q_proj(x)
        v = self.v_proj(x)
        k, q, v = map(self.separate_heads, (k, q, v))
        attn_weights = self.attention(q, k, v, mask)
        attn_output = torch.matmul(attn_weights, v)
        return self.merge_heads(self.out_proj(attn_output))


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Multi‑head attention that processes each head through a variational
    quantum circuit.  The circuit is a shallow RX‑RY‑CNOT chain that
    operates on `d_k` qubits per head.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, d_k: int):
            super().__init__()
            self.d_k = d_k
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(d_k)]
            )
            self.params = nn.Parameter(torch.randn(d_k, 2))
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            # x shape (N, d_k)
            self.encoder(q_device, x)
            for i in range(self.d_k):
                tqf.ry(q_device, wires=[i], params=self.params[i, 0])
                tqf.rx(q_device, wires=[i], params=self.params[i, 1])
            for i in range(self.d_k - 1):
                tqf.cnot(q_device, wires=[i, i + 1])
            tqf.cnot(q_device, wires=[self.d_k - 1, 0])
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1, q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.q_layer = self.QLayer(self.d_k)
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.d_k)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, _ = x.shape
        k = self.k_proj(x)
        q = self.q_proj(x)
        v = self.v_proj(x)
        k, q, v = map(self.separate_heads, (k, q, v))

        def apply_q(layer: torch.Tensor) -> torch.Tensor:
            # layer shape (B, H, T, d_k)
            b_, h_, t_, dk_ = layer.shape
            flat = layer.reshape(b_ * t_ * h_, dk_)
            qdev = self.q_device.copy(bsz=flat.size(0), device=flat.device)
            out_flat = self.q_layer(flat, qdev)
            out = out_flat.reshape(b_, h_, t_, dk_)
            return out

        k_q = apply_q(k)
        q_q = apply_q(q)
        v_q = apply_q(v)

        attn_weights = self.attention(q_q, k_q, v_q, mask)
        attn_output = torch.matmul(attn_weights, v_q)
        return self.merge_heads(self.out_proj(attn_output))


class FeedForwardBase(nn.Module):
    """Base feed‑forward that can be swapped for a quantum variant."""
    def __init__(self, embed_dim: int, ffn_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Standard two‑layer MLP."""
    def __init__(self, embed_dim: int, ffn_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.lin1 = nn.Linear(embed_dim, ffn_dim)
        self.lin2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.dropout(F.relu(self.lin1(x))))


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network realized by a variational quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.params = nn.Parameter(torch.randn(n_qubits, 2))
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            # x shape (N, n_qubits)
            self.encoder(q_device, x)
            for i in range(self.n_qubits):
                tqf.ry(q_device, wires=[i], params=self.params[i, 0])
                tqf.rx(q_device, wires=[i], params=self.params[i, 1])
            for i in range(self.n_qubits - 1):
                tqf.cnot(q_device, wires=[i, i + 1])
            tqf.cnot(q_device, wires=[self.n_qubits - 1, 0])
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int,
                 dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.n_qubits = n_qubits
        self.proj = nn.Linear(embed_dim, n_qubits)
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, e = x.shape
        proj = self.proj(x)  # (b, t, n_qubits)
        flat = proj.reshape(b * t, self.n_qubits)
        qdev = self.q_device.copy(bsz=flat.size(0), device=flat.device)
        out_flat = self.q_layer(flat, qdev)
        out = out_flat.reshape(b, t, self.n_qubits)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockBase(nn.Module):
    """Common infrastructure for all transformer blocks."""
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise


class TransformerBlockClassical(TransformerBlockBase):
    """Classical transformer block."""
    def __init__(self, embed_dim: int, num_heads: int,
                 ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads,
                                                dropout=dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockQuantum(TransformerBlockBase):
    """Hybrid transformer block that can mix classical and quantum sub‑modules."""
    def __init__(self, embed_dim: int, num_heads: int,
                 ffn_dim: int, n_qubits_transformer: int,
                 n_qubits_ffn: int, n_qlayers: int,
                 q_device: Optional[tq.QuantumDevice] = None,
                 dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        if n_qubits_transformer > 0:
            self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads,
                                                  dropout=dropout,
                                                  q_device=q_device)
        else:
            self.attn = MultiHeadAttentionClassical(embed_dim, num_heads,
                                                    dropout=dropout)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn,
                                          dropout=dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

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


class TextClassifier(nn.Module):
    """Transformer‑based text classifier that can leverage quantum sub‑modules."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
        n_qlayers: int = 1,
        q_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        if n_qubits_transformer > 0:
            q_device = q_device or tq.QuantumDevice(
                n_wires=max(n_qubits_transformer, n_qubits_ffn)
            )
            blocks = [
                TransformerBlockQuantum(
                    embed_dim, num_heads, ffn_dim,
                    n_qubits_transformer, n_qubits_ffn,
                    n_qlayers, q_device=q_device, dropout=dropout
                )
                for _ in range(num_blocks)
            ]
        else:
            blocks = [
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim,
                                          dropout=dropout)
                for _ in range(num_blocks)
            ]
        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        if num_classes > 2:
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


# expose the shared class name
QuantumTransformerEnhanced = TextClassifier

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
    "TextClassifier",
    "QuantumTransformerEnhanced",
]
