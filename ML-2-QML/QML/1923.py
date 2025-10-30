"""Quantum‑enhanced transformer implementation using TorchQuantum."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class HybridAttentionBase(nn.Module):
    """Base class for hybrid attention that supports a mix between classical and quantum paths."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        mix_ratio: float = 1.0,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.mix_ratio = mix_ratio
        self.attn_weights: Optional[torch.Tensor] = None
        self.use_bias = use_bias

    def separate_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape and transpose for multi‑head attention."""
        batch_size = tensor.size(0)
        return tensor.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def compute_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Standard dot‑product attention."""
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        return torch.matmul(scores, value)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class HybridAttentionClassical(HybridAttentionBase):
    """Purely classical attention – identical to the original."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_bias: bool = False,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout, mix_ratio=1.0, use_bias=use_bias)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        if embed_dim!= self.embed_dim:
            raise ValueError(f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})")
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        k = self.separate_heads(k)
        q = self.separate_heads(q)
        v = self.separate_heads(v)
        x = self.compute_attention(q, k, v)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.combine_heads(x)


class HybridAttentionQuantum(HybridAttentionBase):
    """Hybrid attention that mixes classical and quantum attention."""

    class QLayer(tq.QuantumModule):
        """Quantum module that implements a simple attention head."""

        def __init__(self, n_wires: int, n_heads: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.n_heads = n_heads
            # Encode each input dimension into a RX gate
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [idx], "func": "rx", "wires": [idx]}
                    for idx in range(n_wires)
                ]
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

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        mix_ratio: float = 0.5,
        use_bias: bool = False,
        q_device: Optional[tq.QuantumDevice] = None,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout, mix_ratio, use_bias)
        self.q_layer = self.QLayer(self.d_k, num_heads)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.q_device = q_device or tq.QuantumDevice(n_wires=self.d_k)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        classical_out = super(HybridAttentionClassical, self).forward(x, mask)
        quantum_out = self._apply_quantum_heads(x)
        return self.mix_ratio * classical_out + (1.0 - self.mix_ratio) * quantum_out

    def _apply_quantum_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the quantum layer to each head."""
        batch_size, seq_len, embed_dim = x.size()
        if embed_dim!= self.embed_dim:
            raise ValueError(f"Input embedding ({embed_dim}) does not match layer embedding size ({self.embed_dim})")
        # Split into heads
        x_heads = self.separate_heads(x)  # shape (B, H, L, d_k)
        # Process each head with a separate quantum device
        outputs = []
        for h in range(self.num_heads):
            head_tensor = x_heads[:, h, :, :]  # (B, L, d_k)
            # Flatten batch and seq for quantum device
            head_flat = head_tensor.reshape(-1, self.d_k).transpose(0, 1)  # (d_k, B*L)
            qdev = self.q_device.copy(bsz=head_flat.size(1), device=head_flat.device)
            q_out = self.q_layer(head_flat, qdev)  # (d_k, B*L)
            q_out = q_out.transpose(0, 1).contiguous().view(batch_size, seq_len, self.d_k)
            outputs.append(q_out)
        # Stack heads
        quantum_heads = torch.stack(outputs, dim=1)  # (B, H, L, d_k)
        quantum_heads = quantum_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.combine_heads(quantum_heads)


class HybridFeedForwardBase(nn.Module):
    """Base class for feed‑forward blocks."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class HybridFeedForwardClassical(HybridFeedForwardBase):
    """Standard two‑layer feed‑forward network."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class HybridFeedForwardQuantum(HybridFeedForwardBase):
    """Feed‑forward network realized by a quantum module."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [idx], "func": "rx", "wires": [idx]}
                    for idx in range(n_wires)
                ]
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

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (B, L, embed_dim)
        outputs = []
        for token in x.unbind(dim=1):  # iterate over sequence
            # token shape (B, embed_dim)
            token_flat = token.transpose(0, 1)  # (embed_dim, B)
            qdev = self.q_device.copy(bsz=token_flat.size(1), device=token_flat.device)
            q_out = self.q_layer(token_flat, qdev)  # (embed_dim, B)
            q_out = q_out.transpose(0, 1)  # (B, embed_dim)
            outputs.append(q_out)
        out = torch.stack(outputs, dim=1)  # (B, L, embed_dim)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


class HybridTransformerBlockBase(nn.Module):
    """Base transformer block containing attention and feed‑forward parts."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class HybridTransformerBlockClassical(HybridTransformerBlockBase):
    """Purely classical transformer block."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = HybridAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = HybridFeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class HybridTransformerBlockQuantum(HybridTransformerBlockBase):
    """Transformer block that mixes classical and quantum attention and feed‑forward."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        mix_ratio: float = 0.5,
        use_quantum_ffn: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = HybridAttentionQuantum(embed_dim, num_heads, dropout, mix_ratio)
        if use_quantum_ffn:
            # Use a quantum feed‑forward block with n_qubits equal to ffn_dim
            self.ffn = HybridFeedForwardQuantum(embed_dim, ffn_dim, ffn_dim, dropout)
        else:
            self.ffn = HybridFeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoding(nn.Module):
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


class TextClassifierHybrid(nn.Module):
    """Transformer‑based text classifier supporting hybrid quantum‑classical layers."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        mix_ratio: float = 0.5,
        use_quantum_ffn: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoding(embed_dim)
        blocks = [
            HybridTransformerBlockQuantum(
                embed_dim,
                num_heads,
                ffn_dim,
                mix_ratio=mix_ratio,
                use_quantum_ffn=use_quantum_ffn,
                dropout=dropout,
            )
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


__all__ = [
    "HybridAttentionBase",
    "HybridAttentionClassical",
    "HybridAttentionQuantum",
    "HybridFeedForwardBase",
    "HybridFeedForwardClassical",
    "HybridFeedForwardQuantum",
    "HybridTransformerBlockBase",
    "HybridTransformerBlockClassical",
    "HybridTransformerBlockQuantum",
    "PositionalEncoding",
    "TextClassifierHybrid",
]
