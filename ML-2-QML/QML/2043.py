import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class QuantumFeatureExtractor(nn.Module):
    """Variational quantum circuit producing a measurement vector."""
    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_wires = n_qubits
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        )
        self.qgates = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_qubits)])
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(q_device, x)
        for i, gate in enumerate(self.qgates):
            gate(q_device, wires=[i])
        return self.measure(q_device)


class MultiHeadAttentionBase(nn.Module):
    """Base class for attention mechanisms."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.d_k = embed_dim // num_heads

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        return torch.matmul(probs, v), probs

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        attn_output, _ = self.attention(q, k, v, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq, self.embed_dim)
        return self.out_proj(attn_output)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Hybrid attention that uses a variational quantum circuit."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1,
                 n_qubits: int = 4) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_qm = nn.Linear(embed_dim, n_qubits, bias=False)
        self.out_proj = nn.Linear(n_qubits, embed_dim, bias=False)
        self.n_qubits = n_qubits
        self.qc = QuantumFeatureExtractor(n_qubits)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        q = self.proj_qm(q)
        k = self.proj_qm(k)
        v = self.proj_qm(v)
        q_flat = q.reshape(batch * seq, -1)
        k_flat = k.reshape(batch * seq, -1)
        v_flat = v.reshape(batch * seq, -1)
        q_device = tq.QuantumDevice(n_wires=self.n_qubits, bsz=batch * seq)
        k_device = tq.QuantumDevice(n_wires=self.n_qubits, bsz=batch * seq)
        v_device = tq.QuantumDevice(n_wires=self.n_qubits, bsz=batch * seq)
        q_qm = self.qc(q_flat, q_device)
        k_qm = self.qc(k_flat, k_device)
        v_qm = self.qc(v_flat, v_device)
        q_qm = q_qm.reshape(batch, seq, self.n_qubits)
        k_qm = k_qm.reshape(batch, seq, self.n_qubits)
        v_qm = v_qm.reshape(batch, seq, self.n_qubits)
        d_k = self.n_qubits // self.num_heads
        if d_k * self.num_heads!= self.n_qubits:
            raise ValueError("n_qubits must be divisible by num_heads")
        q_h = q_qm.view(batch, seq, self.num_heads, d_k).transpose(1, 2)
        k_h = k_qm.view(batch, seq, self.num_heads, d_k).transpose(1, 2)
        v_h = v_qm.view(batch, seq, self.num_heads, d_k).transpose(1, 2)
        attn_output, _ = self.attention(q_h, k_h, v_h, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq, self.n_qubits)
        return self.out_proj(attn_output)


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
    """Standard two‑layer feed‑forward."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network with a variational quantum circuit."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.to_qubits = nn.Linear(ffn_dim, n_qubits, bias=False)
        self.qc = QuantumFeatureExtractor(n_qubits)
        self.linear2 = nn.Linear(n_qubits, ffn_dim, bias=False)
        self.out_proj = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.to_qubits(x)
        batch, seq, _ = x.shape
        x_flat = x.reshape(batch * seq, -1)
        q_device = tq.QuantumDevice(n_wires=self.qc.n_wires, bsz=batch * seq)
        qm_out = self.qc(x_flat, q_device)
        qm_out = qm_out.reshape(batch, seq, -1)
        x = self.linear2(qm_out)
        x = F.relu(x)
        return self.out_proj(self.dropout(x))


class TransformerBlockBase(nn.Module):
    """Base transformer block with normalisation."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    """Classical transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockQuantum(TransformerBlockBase):
    """Transformer block with quantum‑augmented attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1,
                 n_qubits_attention: int = 4,
                 n_qubits_ffn: int = 4) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout,
                                              n_qubits_attention)
        self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)

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
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class HybridTransformer(nn.Module):
    """
    Transformer‑based classifier with optional quantum‑augmented blocks
    and a multi‑task head for classification and regression.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 num_regress: int = 0,
                 dropout: float = 0.1,
                 use_quantum: bool = False,
                 n_qubits_attention: int = 4,
                 n_qubits_ffn: int = 4) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if use_quantum:
                self.blocks.append(
                    TransformerBlockQuantum(embed_dim, num_heads, ffn_dim,
                                            dropout, n_qubits_attention, n_qubits_ffn))
            else:
                self.blocks.append(
                    TransformerBlockClassical(embed_dim, num_heads, ffn_dim,
                                            dropout))
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        self.reg_head = None
        if num_regress > 0:
            self.reg_head = nn.Linear(embed_dim, num_regress)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        out = self.cls_head(x)
        if self.reg_head is not None:
            out = (out, self.reg_head(x))
        return out


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
    "HybridTransformer",
]
