"""Quantum‑refined transformer with modern quantum circuit design."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class MultiHeadAttentionBase(nn.Module):
    """Base class for multi‑head attention, shared by classical and quantum variants."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_bias: bool = True,
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
        """Reshape [B, L, D] → [B, H, L, D/H]."""
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot‑product attention with optional mask."""
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value), scores

    def downstream(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        batch_size: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply attention to all heads and return a reshaped tensor."""
        q = self.separate_heads(query)
        k = self.separate_heads(key)
        v = self.separate_heads(value)
        out, self.attn_weights = self.attention(q, k, v, mask)
        return out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return the attention weights from the last forward pass."""
        return self.attn_weights


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention using PyTorch linear layers."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_bias: bool = True,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.combine = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        batch_size, seq_len, embed_dim = x.size()
        if embed_dim!= self.embed_dim:
            raise ValueError(f"Expected embedding {self.embed_dim} but got {embed_dim}")
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        y = self.downstream(q, k, v, batch_size, mask)
        return self.combine(y)


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Multi‑head attention that encodes each head using a variational quantum circuit."""

    class _QHead(tq.QuantumModule):
        """Quantum encoding of a single head."""
        def __init__(self, d_k: int, n_wires: int = 8):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(d_k)]
            )
            self.gates = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            # Encode classical data into qubit states
            self.encoder(qdev, x)
            # Apply trainable rotations
            for wire, gate in enumerate(self.gates):
                gate(qdev, wires=[wire % self.n_wires])
            # Entangle qubits with a simple ring pattern
            for i in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[i, i + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_bias: bool = True,
        q_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout, use_bias)
        self.q_head = self._QHead(self.d_k)
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.q_head.n_wires)
        self.combine = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def _apply_quantum(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the same quantum circuit to every head and token."""
        batch_size, seq_len, _ = x.shape
        # reshape to [B, L, H, D/H]
        heads = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        outputs = []
        for h in range(self.num_heads):
            head = heads[:, :, h, :]  # [B, L, D/H]
            # For each token in the batch, use a quantum device
            qdev = self.q_device.copy(bsz=batch_size, device=head.device)
            outputs.append(self.q_head(head, qdev))
        # stack heads back
        return torch.stack(outputs, dim=2).view(batch_size, seq_len, self.embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        k = self._apply_quantum(x)
        q = self._apply_quantum(x)
        v = self._apply_quantum(x)
        y = self.downstream(q, k, v, x.size(0), mask)
        return self.combine(y)


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward networks."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)


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

    class _QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int, n_layers: int = 1):
            super().__init__()
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.gates = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
            # Encode
            self.encoder(qdev, x)
            # Apply trainable rotations
            for wire, gate in enumerate(self.gates):
                gate(qdev, wires=[wire % self.n_qubits])
            # Simple entanglement pattern
            for i in range(self.n_qubits - 1):
                tqf.cnot(qdev, wires=[i, i + 1])
            tqf.cnot(qdev, wires=[self.n_qubits - 1, 0])
            return self.measure(qdev)

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        n_qubits: int,
        dropout: float = 0.1,
        n_layers: int = 1,
    ) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self._QLayer(n_qubits, n_layers)
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


class TransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface only
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockQuantum(TransformerBlockBase):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_transformer: int,
        n_qubits_ffn: int,
        n_qlayers: int,
        q_device: Optional[tq.QuantumDevice] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionQuantum(
            embed_dim, num_heads, dropout, q_device=q_device
        )
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(
                embed_dim, ffn_dim, n_qubits_ffn, dropout, n_layers=n_qlayers
            )
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding with optional learnable bias."""

    def __init__(self, embed_dim: int, max_len: int = 5000, learnable_bias: bool = False):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.learnable_bias = learnable_bias
        if learnable_bias:
            self.bias = nn.Parameter(torch.zeros(1, max_len, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.pe[:, : x.size(1)]
        if self.learnable_bias:
            out = out + self.bias[:, : x.size(1)]
        return out


class TextClassifier(nn.Module):
    """Transformer‑based text classifier supporting quantum sub‑modules."""

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
        learnable_pos_bias: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim, learnable_bias=learnable_pos_bias)
        if n_qubits_transformer > 0:
            q_device = q_device or tq.QuantumDevice(n_wires=max(n_qubits_transformer, n_qubits_ffn))
            blocks = [
                TransformerBlockQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_qubits_transformer,
                    n_qubits_ffn,
                    n_qlayers,
                    q_device=q_device,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        else:
            blocks = [
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

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
