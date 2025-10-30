"""QTransformerTorchGen415: quantum‑enhanced transformer with stochastic depth.

This module implements a transformer‑based text classifier that can use
quantum sub‑modules for attention and feed‑forward sub‑layers.  It
preserves the original API while adding a learnable gating scalar to
mix classical and quantum outputs, a learnable positional bias, and
stochastic depth regularisation for each block.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


# --------------------------------------------------------------------------- #
# 1. Multi‑Head Attention
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Base class for multi‑head attention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.d_k = embed_dim // num_heads

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with torch.nn.MultiheadAttention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output


class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Multi‑head attention that mixes classical and quantum projections."""
    class QLayer(tq.QuantumModule):
        """Quantum module that processes a single head vector."""
        def __init__(self, n_wires: int = 8) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.parameters = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(q_device, wires=[wire, wire + 1])
            tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
            return self.measure(q_device)

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        q_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.q_layer = self.QLayer()
        self.q_device = q_device
        self.combine_heads = nn.Linear(embed_dim, embed_dim)
        self.gate = nn.Parameter(torch.tensor(1.0))  # learnable gate between classical and quantum

    def _apply_quantum_head(self, head: torch.Tensor) -> torch.Tensor:
        """Apply quantum processing to a single head vector."""
        # head shape: (batch, d_k)
        # Map to n_wires qubits; if d_k > n_wires, truncate; if less, pad with zeros
        n_wires = self.q_layer.n_wires
        batch = head.size(0)
        if head.size(1) > n_wires:
            head = head[:, :n_wires]
        elif head.size(1) < n_wires:
            pad = torch.zeros(batch, n_wires - head.size(1), device=head.device, dtype=head.dtype)
            head = torch.cat([head, pad], dim=1)
        qdev = self.q_device or tq.QuantumDevice(n_wires=n_wires, bsz=batch, device=head.device)
        return self.q_layer(head, qdev)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        batch_size = x.size(0)
        # Classical projections
        k = nn.Linear(self.embed_dim, self.embed_dim)(x)
        q = nn.Linear(self.embed_dim, self.embed_dim)(x)
        v = nn.Linear(self.embed_dim, self.embed_dim)(x)
        # Separate heads
        def separate(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        qh = separate(q)
        kh = separate(k)
        vh = separate(v)
        # Classical attention
        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        classical_out = torch.matmul(scores, vh).transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        # Quantum attention per head
        quantum_heads = []
        for h in range(self.num_heads):
            head_q = qh[:, h]  # shape (batch, d_k)
            quantum_head = self._apply_quantum_head(head_q)
            quantum_heads.append(quantum_head)
        quantum_out = torch.stack(quantum_heads, dim=1).transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        # Gated mix
        output = self.gate * quantum_out + (1.0 - self.gate) * classical_out
        return self.combine_heads(output)


# --------------------------------------------------------------------------- #
# 2. Feed‑Forward Network
# --------------------------------------------------------------------------- #
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
    """Feed‑forward network realised by a quantum module."""
    class QLayer(tq.QuantumModule):
        """Quantum module that processes a single token vector."""
        def __init__(self, n_qubits: int) -> None:
            super().__init__()
            self.n_wires = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.parameters = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.parameters):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.gate = nn.Parameter(torch.tensor(1.0))  # learnable gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        batch = x.size(0)
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            out = self.q_layer(token, qdev)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        classical_out = self.linear2(F.relu(self.linear1(x)))  # classical path
        quantum_out = self.linear2(F.relu(out))
        return self.gate * quantum_out + (1.0 - self.gate) * classical_out


# --------------------------------------------------------------------------- #
# 3. Transformer Block
# --------------------------------------------------------------------------- #
class TransformerBlockBase(nn.Module):
    """Base transformer block with optional stochastic depth."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.drop_path_rate = drop_path_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def _stochastic_depth(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.drop_path_rate > 0.0 and random.random() < self.drop_path_rate:
            return x
        return x


class TransformerBlockQuantum(TransformerBlockBase):
    """Transformer block that uses quantum attention and feed‑forward."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_transformer: int,
        n_qubits_ffn: int,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        q_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout, drop_path_rate)
        self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, q_device=q_device)
        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self._stochastic_depth(x)
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class TransformerBlockClassical(TransformerBlockBase):
    """Standard transformer block with classical modules."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout, drop_path_rate)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self._stochastic_depth(x)
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
# 4. Positional Encoding
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding with a learnable bias."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.pos_bias = nn.Parameter(torch.zeros(1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)] + self.pos_bias


# --------------------------------------------------------------------------- #
# 5. Text Classifier
# --------------------------------------------------------------------------- #
class QTransformerTorchGen415(nn.Module):
    """Transformer‑based text classifier supporting quantum sub‑modules.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary.
    embed_dim : int
        Hidden dimension of the model.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Hidden dimension of the feed‑forward network.
    num_classes : int
        Number of target classes.
    dropout : float, optional
        Dropout probability.
    use_quantum : bool, optional
        If True, use quantum layers for attention and feed‑forward.
    n_qubits_transformer : int, optional
        Number of qubits per transformer block.
    n_qubits_ffn : int, optional
        Number of qubits for the feed‑forward sub‑module.
    n_qlayers : int, optional
        Number of quantum layers per block (unused in this simplified example).
    q_device : Optional[torchquantum.QuantumDevice], optional
        Quantum device to use.
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
        use_quantum: bool = False,
        n_qubits_transformer: int = 0,
        n_qubits_ffn: int = 0,
        n_qlayers: int = 1,
        q_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        blocks = []
        drop_path_rate = 0.1
        for _ in range(num_blocks):
            if use_quantum and n_qubits_transformer > 0:
                block = TransformerBlockQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    n_qubits_transformer,
                    n_qubits_ffn,
                    dropout,
                    drop_path_rate,
                    q_device=q_device,
                )
            else:
                block = TransformerBlockClassical(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    dropout,
                    drop_path_rate,
                )
            blocks.append(block)
        self.transformers = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
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
    "QTransformerTorchGen415",
]
