"""
Hybrid Transformer – Quantum‑enabled implementation.

Quantum sub‑modules are built on TorchQuantum and can replace the
classical attention and feed‑forward layers when the corresponding
flags are set.  The API remains identical to the classical version
(`TextClassifier`) and accepts the same arguments; only the
`use_quantum`, `n_qubits_transformer` and `n_qubits_ffn` flags
activate the quantum stack.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


# --------------------------------------------------------------------------- #
#  Quantum linear mapping – reusable quantum module
# --------------------------------------------------------------------------- #

class QLinear(tq.QuantumModule):
    """
    Linear projection implemented via a quantum circuit.
    * input_dim → linear → n_wires → RandomLayer → measurement → linear_out
    """
    def __init__(self, input_dim: int, output_dim: int, n_wires: int = 4, n_ops: int = 50):
        super().__init__()
        self.n_wires = n_wires
        self.linear_in = nn.Linear(input_dim, n_wires)
        # encoder uses a fixed Ry/Z/X pattern – works for any n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}x{n_wires}_ryzxy"])
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear_out = nn.Linear(n_wires, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        encoded = self.linear_in(x)
        self.encoder(qdev, encoded)
        self.random_layer(qdev)
        out = self.measure(qdev)
        out = self.linear_out(out)
        return out


# --------------------------------------------------------------------------- #
#  Quantum attention – replaces MultiHeadAttentionClassical
# --------------------------------------------------------------------------- #

class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """
    Multi‑head attention where the Q, K, V projections are quantum linear layers.
    The attention computation itself remains classical.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, n_wires: int = 4):
        super().__init__(embed_dim, num_heads, dropout)
        self.k_proj = QLinear(embed_dim, embed_dim, n_wires)
        self.q_proj = QLinear(embed_dim, embed_dim, n_wires)
        self.v_proj = QLinear(embed_dim, embed_dim, n_wires)
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        k = self.k_proj(x)
        q = self.q_proj(x)
        v = self.v_proj(x)

        def separate_heads(t: torch.Tensor) -> torch.Tensor:
            batch_size = t.size(0)
            return t.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        qh = separate_heads(q)
        kh = separate_heads(k)
        vh = separate_heads(v)

        scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, vh)
        out = out.transpose(1, 2).contiguous().view(x.size(0), -1, self.embed_dim)
        return self.combine(out)


# --------------------------------------------------------------------------- #
#  Quantum feed‑forward – replaces FeedForwardClassical
# --------------------------------------------------------------------------- #

class FeedForwardQuantum(FeedForwardBase):
    """
    Two‑layer feed‑forward network where each linear layer is a QLinear.
    """
    def __init__(self, embed_dim: int, ffn_dim: int, n_wires: int = 4, n_ops: int = 50):
        super().__init__(embed_dim, ffn_dim)
        self.qlinear1 = QLinear(embed_dim, ffn_dim, n_wires, n_ops)
        self.qlinear2 = QLinear(ffn_dim, embed_dim, n_wires, n_ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qlinear2(F.relu(self.qlinear1(x)))


# --------------------------------------------------------------------------- #
#  Hybrid transformer block – selects between classical and quantum parts
# --------------------------------------------------------------------------- #

class TransformerBlockQuantum(TransformerBlockBase):
    """
    Transformer block that can mix classical and quantum sub‑modules.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        n_qubits_transformer: int,
        n_qubits_ffn: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        if n_qubits_transformer > 0:
            self.attn = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout, n_qubits_transformer)
        else:
            self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)

        if n_qubits_ffn > 0:
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits_ffn)
        else:
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
#  Positional encoder – identical to classical version
# --------------------------------------------------------------------------- #

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
        return x + self.pe[:, :x.size(1)]


# --------------------------------------------------------------------------- #
#  Public classifier – API identical to the classical version
# --------------------------------------------------------------------------- #

class TextClassifier(nn.Module):
    """
    Transformer‑based text classifier that can operate in purely
    classical mode or with quantum blocks.  The flags `use_quantum`,
    `n_qubits_transformer` and `n_qubits_ffn` control the depth of
    quantum layers.
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
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)

        # Decide on block type
        if use_quantum and (n_qubits_transformer > 0 or n_qubits_ffn > 0):
            self.blocks = nn.ModuleList(
                [
                    TransformerBlockQuantum(
                        embed_dim,
                        num_heads,
                        ffn_dim,
                        n_qubits_transformer,
                        n_qubits_ffn,
                        dropout=dropout,
                    )
                    for _ in range(num_blocks)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                    for _ in range(num_blocks)
                ]
            )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x.mean(dim=1))
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
