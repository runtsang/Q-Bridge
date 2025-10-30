"""Extended transformer with optional quantum modules and attention visualization."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionBase(nn.Module):
    """Base class for attention mechanisms with optional weight extraction."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.attn_weights: Optional[torch.Tensor] = None

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape input for multi‑head attention."""
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot‑product attention."""
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # mask shape: (batch, seq_len) -> (batch, 1, 1, seq_len)
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        self.attn_weights = scores.detach().clone()
        return torch.matmul(scores, value), scores

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with torch.nn.MultiheadAttention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_out


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Multi‑head attention where each projection is passed through a small quantum module."""
    # The quantum implementation uses torchquantum.  For brevity we only
    # import it when the module is instantiated.

    class _QLayer(nn.Module):
        """Simple quantum layer that applies a parameterised RX on each wire."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            import torchquantum as tq
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ParameterList(
                [nn.Parameter(torch.randn(n_wires)) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: "tq.QuantumDevice") -> torch.Tensor:
            import torchquantum as tq
            self.encoder(q_device, x)
            for i, param in enumerate(self.parameters):
                tq.RX(param, wires=i)(q_device)
            return self.measure(q_device)

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        n_wires_per_head: int = 8,
        q_device: Optional["tq.QuantumDevice"] = None,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.n_wires_per_head = n_wires_per_head
        self.q_layer = self._QLayer(n_wires_per_head)
        self.q_device = q_device
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def _apply_quantum(self, x: torch.Tensor) -> torch.Tensor:
        """Apply quantum layer to each head projection."""
        import torchquantum as tq
        batch, seq_len, _ = x.shape
        proj = self.separate_heads(x)  # (batch, heads, seq_len, d_k)
        outputs = []
        for i in range(self.num_heads):
            head = proj[:, i]  # (batch, seq_len, d_k)
            out = []
            for token in head.unbind(dim=1):  # iterate over seq_len
                qdev = self.q_device or tq.QuantumDevice(n_wires=self.n_wires_per_head, bsz=token.size(0), device=token.device)
                out.append(self.q_layer(token, qdev))
            out = torch.stack(out, dim=1)  # (batch, seq_len, n_wires)
            outputs.append(out)
        return torch.stack(outputs, dim=1)  # (batch, heads, seq_len, n_wires)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        batch_size, seq_len, _ = x.size()
        # obtain quantum projections
        quantum_proj = self._apply_quantum(x)  # (batch, heads, seq_len, n_wires)
        # reshape to match classical attention shape
        quantum_proj = quantum_proj.view(batch_size, self.num_heads, seq_len, -1).transpose(1, 2)
        # compute attention over these quantum‑encoded projections
        attn_out, _ = self.attention(quantum_proj, quantum_proj, quantum_proj, mask)
        return self.combine_heads(attn_out)


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward layers."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Standard two‑layer feed‑forward network."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class FeedForwardQuantum(FeedForwardBase):
    """Feed‑forward that uses a quantum module for the first linear transformation."""
    class _QLayer(nn.Module):
        def __init__(self, n_qubits: int):
            super().__init__()
            import torchquantum as tq
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.parameters = nn.ParameterList(
                [nn.Parameter(torch.randn(n_qubits)) for _ in range(n_qubits)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: "tq.QuantumDevice") -> torch.Tensor:
            import torchquantum as tq
            self.encoder(q_device, x)
            for i, param in enumerate(self.parameters):
                tq.RX(param, wires=i)(q_device)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self._QLayer(n_qubits)
        self.q_device = None  # will be created per batch
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply quantum layer token‑wise
        outs = []
        for token in x.unbind(dim=1):
            import torchquantum as tq
            qdev = tq.QuantumDevice(n_wires=self.q_layer.n_qubits, bsz=token.size(0), device=token.device)
            outs.append(self.q_layer(token, qdev))
        q_out = torch.stack(outs, dim=1)  # (batch, seq_len, n_qubits)
        q_out = self.linear1(self.dropout(q_out))
        return self.linear2(F.relu(q_out))


class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return the most recently computed attention weights, if any."""
        return getattr(self, "attn", None).attn_weights if hasattr(self, "attn") else None


class TransformerBlockClassical(TransformerBlockBase):
    """Classic transformer block."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockQuantum(TransformerBlockBase):
    """Quantum‑augmented transformer block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_wires_per_head: int = 8,
        n_ffn_qubits: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(
            embed_dim, num_heads, dropout, n_wires_per_head
        )
        self.ffn = (
            FeedForwardQuantum(embed_dim, ffn_dim, n_ffn_qubits, dropout)
            if n_ffn_qubits > 0
            else FeedForwardClassical(embed_dim, ffn_dim, dropout)
        )

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
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TextClassifier(nn.Module):
    """Transformer‑based text classifier with optional quantum sub‑modules."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        n_wires_per_head: int = 0,
        n_ffn_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        blocks = []
        for _ in range(num_blocks):
            if n_wires_per_head > 0:
                blocks.append(
                    TransformerBlockQuantum(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        n_wires_per_head=n_wires_per_head,
                        n_ffn_qubits=n_ffn_qubits,
                        dropout=dropout,
                    )
                )
            else:
                blocks.append(
                    TransformerBlockClassical(
                        embed_dim, num_heads, ffn_dim, dropout=dropout
                    )
                )
        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = (
            nn.Linear(embed_dim, num_classes)
            if num_classes > 2
            else nn.Linear(embed_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
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
    "TextClassifier",
]
