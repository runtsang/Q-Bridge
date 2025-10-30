"""
QTransformerHybrid – a transformer‑based classifier with optional quantum submodules.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Base abstractions
# ------------------------------------------------------------------
class MultiHeadAttentionBase(nn.Module):
    """Shared logic for attention layers."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.size()
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, v)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardBase(nn.Module):
    """Shared interface for feed‑forward layers."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


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
        return x + self.pe[:, : x.size(1)]


# ------------------------------------------------------------------
# Classical implementations
# ------------------------------------------------------------------
class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)
        out = self.attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(x.size(0), -1, self.embed_dim)
        return self.combine_heads(out)


class FeedForwardClassical(FeedForwardBase):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockClassical(TransformerBlockBase):
    def __init__(self, embed_dim: int, num_heads: int,
                 ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# ------------------------------------------------------------------
# Quantum implementations (torchquantum)
# ------------------------------------------------------------------
try:
    import torchquantum as tq
    import torchquantum.functional as tqf
except ImportError:  # pragma: no cover
    tq = None
    tqf = None


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Attention layer that uses a small quantum circuit for each head."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 8) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1, q_device: Optional[tq.QuantumDevice] = None) -> None:
        if tq is None:
            raise ImportError("torchquantum is required for quantum attention")
        super().__init__(embed_dim, num_heads, dropout)
        self.q_layer = self.QLayer()
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.q_layer.n_wires)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def _apply_quantum_projection(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the quantum encoder to each token."""
        batch, seq, _ = x.size()
        projections = []
        for token in x.unbind(dim=1):  # iterate over sequence
            token = token.view(token.size(0), self.num_heads, -1)
            head_outs = []
            for head in token.unbind(dim=1):
                qdev = self.q_device.copy(bsz=head.size(0), device=head.device)
                head_outs.append(self.q_layer(head, qdev))
            projections.append(torch.stack(head_outs, dim=1))
        return torch.stack(projections, dim=1)  # (batch, seq, num_heads, wires)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        # quantum projections
        k_q = self._apply_quantum_projection(k)
        q_q = self._apply_quantum_projection(q)
        v_q = self._apply_quantum_projection(v)
        out = self.attention(q_q, k_q, v_q, mask)
        out = out.transpose(1, 2).contiguous().view(x.size(0), -1, self.embed_dim)
        return self.combine_heads(out)


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward implemented by a small quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int,
                 dropout: float = 0.1) -> None:
        if tq is None:
            raise ImportError("torchquantum is required for quantum feed‑forward")
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class TransformerBlockQuantum(TransformerBlockBase):
    """Transformer block that can mix classical and quantum sub‑modules."""
    def __init__(self, embed_dim: int, num_heads: int,
                 ffn_dim: int, n_qubits_attn: int = 0,
                 n_qubits_ffn: int = 0,
                 q_device: Optional[tq.QuantumDevice] = None,
                 dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        if n_qubits_attn > 0:
            self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads,
                                                  dropout, q_device=q_device)
        else:
            self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# ------------------------------------------------------------------
# Hybrid head (classical or quantum)
# ------------------------------------------------------------------
class HybridHead(nn.Module):
    """A flexible head that can either be a simple sigmoid or a quantum expectation."""
    class QuantumCircuit(tq.QuantumModule):
        def __init__(self, n_qubits: int = 1) -> None:
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [0], "func": "rx", "wires": [0]}]
            )
            self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True)
                                             for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, in_features: int, shift: float = 0.0,
                 use_quantum: bool = False, n_qubits: int = 1) -> None:
        super().__init__()
        self.shift = shift
        self.use_quantum = use_quantum
        if use_quantum:
            if tq is None:
                raise ImportError("torchquantum is required for quantum head")
            self.circuit = self.QuantumCircuit(n_qubits)
            self.q_device = tq.QuantumDevice(n_wires=n_qubits)
            self.linear = nn.Linear(in_features, 1)  # mapping to single parameter
        else:
            self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_quantum:
            # scalar parameter per batch element
            param = self.linear(x).view(-1, 1)
            out = []
            for p in param:
                qdev = self.q_device.copy(bsz=1, device=p.device)
                out.append(self.circuit(p, qdev))
            out = torch.stack(out, dim=0).squeeze(-1)
            return out
        else:
            logits = self.linear(x).squeeze(-1)
            return torch.sigmoid(logits + self.shift)


# ------------------------------------------------------------------
# Main hybrid transformer
# ------------------------------------------------------------------
class QTransformerHybrid(nn.Module):
    """
    Transformer‑based classifier that can optionally use quantum attention,
    quantum feed‑forward, and a hybrid quantum/classical head.

    Parameters
    ----------
    vocab_size : int
        Size of the input vocabulary.
    embed_dim : int
        Dimensionality of token embeddings.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer layers.
    ffn_dim : int
        Hidden dimension of the feed‑forward network.
    num_classes : int
        Number of output classes (1 for binary, >1 for multi‑class).
    dropout : float
        Drop‑out probability.
    n_qubits_attn : int, default 0
        If >0, each attention head is implemented quantum‑wise with this many qubits.
    n_qubits_ffn : int, default 0
        If >0, each feed‑forward sub‑network uses a quantum module with this many qubits.
    n_qubits_head : int, default 0
        If >0, the classification head is quantum‑based.
    shift : float, default 0.0
        Bias shift applied to the sigmoid in the hybrid head.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int = 2,
                 dropout: float = 0.1,
                 n_qubits_attn: int = 0,
                 n_qubits_ffn: int = 0,
                 n_qubits_head: int = 0,
                 shift: float = 0.0) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                TransformerBlockQuantum(
                    embed_dim, num_heads, ffn_dim,
                    n_qubits_attn=n_qubits_attn,
                    n_qubits_ffn=n_qubits_ffn,
                    q_device=None,  # let quantum modules create their own device
                    dropout=dropout
                )
            )
        self.dropout = nn.Dropout(dropout)
        self.head = HybridHead(
            in_features=embed_dim,
            shift=shift,
            use_quantum=(n_qubits_head > 0),
            n_qubits=n_qubits_head
        )
        if num_classes > 2:
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = nn.Identity()  # head already produces probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x)
        x = self.pos_encoder(x)
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        if isinstance(self.classifier, nn.Identity):
            out = self.head(x)
            return torch.cat((out.unsqueeze(-1), 1 - out.unsqueeze(-1)), dim=-1)
        else:
            logits = self.classifier(x)
            return logits


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
    "HybridHead",
    "QTransformerHybrid",
]
