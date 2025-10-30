"""Quantum‑enhanced transformer layers implemented with TorchQuantum."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


# --------------------------------------------------------------------------- #
#  Quantum attention
# --------------------------------------------------------------------------- #
class MultiHeadAttentionQuantum(nn.Module):
    """Multi‑head attention using variational quantum circuits for each projection."""

    class QLayer(tq.QuantumModule):
        """Variational layer that encodes a classical vector into a quantum state,
        applies a trainable rotation, and measures expectation values."""

        def __init__(self, n_wires: int = 8) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [idx], "func": "rx", "wires": [idx]}
                    for idx in range(n_wires)
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.params):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        n_wires: int = 8,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer(n_wires)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = F.softmax(scores, dim=-1)
        return self.dropout(scores) @ v

    def _apply_quantum_proj(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the quantum layer to each token independently
        outputs = []
        for token in x.unbind(dim=1):
            qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self._apply_quantum_proj(x)
        k = self._apply_quantum_proj(x)
        v = self._apply_quantum_proj(x)
        qk = self.separate_heads(q)
        kq = self.separate_heads(k)
        vq = self.separate_heads(v)
        out = self.attention(qk, kq, vq)
        out = out.transpose(1, 2).contiguous().view(x.shape)
        return self.out_proj(out)


# --------------------------------------------------------------------------- #
#  Quantum feed‑forward
# --------------------------------------------------------------------------- #
class FeedForwardQuantum(nn.Module):
    """Feed‑forward network realised by a variational quantum circuit."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 8):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [idx], "func": "rx", "wires": [idx]} for idx in range(n_wires)]
            )
            self.params = nn.ModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for wire, gate in enumerate(self.params):
                gate(q_device, wires=wire)
            return self.measure(q_device)

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        n_qubits: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)
        self.q_layer = self.QLayer(n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def _apply_quantum(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_out = self._apply_quantum(x)
        out = self.linear1(self.dropout(q_out))
        return self.linear2(F.relu(out))


# --------------------------------------------------------------------------- #
#  Quantum transformer block
# --------------------------------------------------------------------------- #
class TransformerBlockQuantum(nn.Module):
    """Transformer block that uses quantum attention and/or quantum feed‑forward."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        use_quantum_attention: bool = False,
        use_quantum_ffn: bool = False,
        n_wires: int = 8,
        n_qubits_ffn: int = 8,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = (
            MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, n_wires=n_wires)
            if use_quantum_attention
            else nn.Identity()
        )
        self.ffn = (
            FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn, dropout)
            if use_quantum_ffn
            else nn.Identity()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(self.attn, nn.Identity):
            attn_out = self.attn(x)
            x = self.norm1(x + self.dropout(attn_out))
        if not isinstance(self.ffn, nn.Identity):
            ffn_out = self.ffn(x)
            x = self.norm2(x + self.dropout(ffn_out))
        return x


# --------------------------------------------------------------------------- #
#  Positional encoding
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
#  Text classifier
# --------------------------------------------------------------------------- #
class TextClassifierQuantum(nn.Module):
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
        use_quantum_attention: bool = False,
        use_quantum_ffn: bool = False,
        n_wires: int = 8,
        n_qubits_ffn: int = 8,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlockQuantum(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    dropout,
                    use_quantum_attention,
                    use_quantum_ffn,
                    n_wires,
                    n_qubits_ffn,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = (
            nn.Linear(embed_dim, num_classes)
            if num_classes > 2
            else nn.Linear(embed_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.positional(tokens)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionQuantum",
    "FeedForwardQuantum",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "TextClassifierQuantum",
]
