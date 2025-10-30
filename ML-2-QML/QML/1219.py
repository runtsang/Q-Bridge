import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class MultiHeadAttentionBase(nn.Module):
    """Base class that validates dimensions and provides utilities for splitting heads."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape to (batch, heads, seq, head_dim)."""
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Re‑assemble heads into the original embedding dimension."""
        batch, heads, seq, dim = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq, heads * dim)

    def attention(self,
                   q: torch.Tensor,
                   k: torch.Tensor,
                   v: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Scaled dot‑product attention."""
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        return self.dropout(F.softmax(scores, dim=-1))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention using linear projections."""

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 use_bias: bool = True) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        attn_weights = self.attention(q, k, v, mask)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = self.merge_heads(attn_output)
        return self.out_proj(attn_output)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum‑aware multi‑head attention that projects each head through a variational circuit."""

    class QHead(tq.QuantumModule):
        def __init__(self, n_qubits: int):
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
            for gate in self.parameters:
                gate(q_device)
            return self.measure(q_device)

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 use_bias: bool = True,
                 n_qubits_per_head: Optional[int] = None,
                 q_device: Optional[tq.QuantumDevice] = None):
        super().__init__(embed_dim, num_heads, dropout)
        self.q_device = q_device or tq.QuantumDevice(n_wires=8)
        self.n_qubits_per_head = n_qubits_per_head or self.head_dim
        self.q_head = self.QHead(self.n_qubits_per_head)
        self.q_proj = nn.Linear(self.n_qubits_per_head, embed_dim, bias=use_bias)
        self.k_proj = nn.Linear(self.n_qubits_per_head, embed_dim, bias=use_bias)
        self.v_proj = nn.Linear(self.n_qubits_per_head, embed_dim, bias=use_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        # Separate heads
        x_heads = self.separate_heads(x)  # (batch, heads, seq, head_dim)
        # Quantum projection for each head
        q_head_outs = []
        for h in range(self.num_heads):
            head_input = x_heads[:, h, :, :]  # (batch, seq, head_dim)
            head_out = []
            for token in head_input.unbind(dim=1):
                qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
                out = self.q_head(token, qdev)
                head_out.append(out)
            head_out = torch.stack(head_out, dim=1)  # (batch, seq, n_qubits)
            q_head_outs.append(head_out)
        # Stack heads back
        quantum_proj = torch.stack(q_head_outs, dim=1)  # (batch, heads, seq, n_qubits)
        # Flatten heads to apply linear projections
        quantum_proj = quantum_proj.view(batch, self.num_heads * seq, self.n_qubits_per_head)
        q = self.q_proj(quantum_proj)
        k = self.k_proj(quantum_proj)
        v = self.v_proj(quantum_proj)
        # Reshape back to multi‑head
        q = q.view(batch, self.num_heads, seq, self.head_dim)
        k = k.view(batch, self.num_heads, seq, self.head_dim)
        v = v.view(batch, self.num_heads, seq, self.head_dim)
        attn_weights = self.attention(q, k, v, mask)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = self.merge_heads(attn_output)
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
    """Two‑layer perceptron feed‑forward network."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward network realised by a variational quantum circuit."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int, n_layers: int = 1):
            super().__init__()
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_qubits * n_layers)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            idx = 0
            for _ in range(self.n_layers):
                for gate in self.parameters[idx:idx + self.n_qubits]:
                    gate(q_device)
                idx += self.n_qubits
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
        self.q_layer = self.QLayer(n_qubits, n_layers)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            out = self.q_layer(token, qdev)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)  # (batch, seq, n_qubits)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
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
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_qubits_transformer: int,
                 n_qubits_ffn: int,
                 n_qlayers: int = 1,
                 dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(embed_dim,
                                              num_heads,
                                              dropout,
                                              n_qubits_per_head=n_qubits_transformer)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim,
                                          ffn_dim,
                                          n_qubits_ffn,
                                          dropout,
                                          n_layers=n_qlayers)
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
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class QuantumTransformerHybrid(nn.Module):
    """Hybrid transformer that can optionally use quantum submodules."""

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 use_quantum: bool = False,
                 n_qubits_transformer: int = 0,
                 n_qubits_ffn: int = 0,
                 n_qlayers: int = 1):
        super().__init__()
        self.use_quantum = use_quantum
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        blocks = []
        for _ in range(num_blocks):
            if use_quantum and n_qubits_transformer > 0:
                blocks.append(TransformerBlockQuantum(embed_dim,
                                                      num_heads,
                                                      ffn_dim,
                                                      n_qubits_transformer,
                                                      n_qubits_ffn,
                                                      n_qlayers,
                                                      dropout=dropout))
            else:
                blocks.append(TransformerBlockClassical(embed_dim,
                                                       num_heads,
                                                       ffn_dim,
                                                       dropout))
        self.transformer = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim,
                                    num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
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
    "QuantumTransformerHybrid",
]
