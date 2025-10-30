"""
UnifiedAttentionTransformer - classical core with optional quantum augmentation.
Author: gpt-oss-20b
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional quantum dependencies
try:
    import torchquantum as tq
    import torchquantum.functional as tqf
except ImportError:
    tq = None
    tqf = None


# --------------------------------------------------------------------------- #
#  Classical self‑attention (API compatible)
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, rotation_params=None, entangle_params=None) -> torch.Tensor:
        # rotation_params and entangle_params are ignored for classical mode
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.embed_dim), dim=-1)
        return scores @ v


# --------------------------------------------------------------------------- #
#  Quantum self‑attention using TorchQuantum
# --------------------------------------------------------------------------- #
class QuantumSelfAttention(nn.Module):
    class _QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for w, gate in enumerate(self.params):
                gate(q_device, wires=[w])
            for w in range(self.n_wires - 1):
                tqf.cnot(q_device, wires=[w, w + 1])
            tqf.cnot(q_device, wires=[self.n_wires - 1, 0])
            return self.measure(q_device)

    def __init__(self, embed_dim: int, n_heads: int, n_wires: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_wires = n_wires
        self.q_layer = self._QLayer(n_wires)
        self.out_proj = nn.Linear(n_wires, embed_dim)

    def forward(self, x: torch.Tensor, rotation_params=None, entangle_params=None) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        head_dim = self.embed_dim // self.n_heads
        x_heads = x.reshape(batch, seq_len, self.n_heads, head_dim)
        outputs = []
        for head in range(self.n_heads):
            tokens = x_heads[:, :, head, :].reshape(batch, seq_len, head_dim)
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=batch)
            out = self.q_layer(tokens, qdev)
            outputs.append(out.reshape(batch, seq_len, 1))
        q_out = torch.cat(outputs, dim=-1)
        return self.out_proj(q_out)


# --------------------------------------------------------------------------- #
#  Classical multi‑head attention
# --------------------------------------------------------------------------- #
class MultiHeadAttentionClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, rotation_params=None, entangle_params=None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x)
        return attn_output


# --------------------------------------------------------------------------- #
#  Quantum feed‑forward
# --------------------------------------------------------------------------- #
class FeedForwardQuantum(nn.Module):
    class _QLayer(tq.QuantumModule):
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
            )
            self.params = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_qubits)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
            self.encoder(q_device, x)
            for w, gate in enumerate(self.params):
                gate(q_device, wires=[w])
            return self.measure(q_device)

    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int, dropout: float = 0.1):
        super().__init__()
        self.q_layer = self._QLayer(n_qubits)
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for token in x.unbind(dim=1):
            qdev = self.q_device.copy(bsz=token.size(0), device=token.device)
            outputs.append(self.q_layer(token, qdev))
        out = torch.stack(outputs, dim=1)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))


# --------------------------------------------------------------------------- #
#  Classical feed‑forward
# --------------------------------------------------------------------------- #
class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# --------------------------------------------------------------------------- #
#  Hybrid transformer block
# --------------------------------------------------------------------------- #
class HybridTransformerBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 mode: str = "classical",
                 n_qubits: int = 0,
                 dropout: float = 0.1):
        super().__init__()
        self.mode = mode
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        if mode == "quantum":
            self.attn = QuantumSelfAttention(embed_dim, num_heads, n_qubits)
            self.ffn = FeedForwardQuantum(embed_dim, ffn_dim, n_qubits, dropout)
        else:
            self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
            self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor,
                rotation_params=None,
                entangle_params=None) -> torch.Tensor:
        attn_out = self.attn(x, rotation_params, entangle_params)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
#  Positional encoding
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        self.register_buffer("pe", self._build_pe(embed_dim, max_len))

    @staticmethod
    def _build_pe(embed_dim: int, max_len: int) -> torch.Tensor:
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32)
                             * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
#  Unified class
# --------------------------------------------------------------------------- #
class UnifiedAttentionTransformer(nn.Module):
    """
    Hybrid transformer that can be instantiated in classical or quantum mode.
    The constructor accepts the same arguments as the original QTransformerTorch
    but adds ``mode`` and ``n_qubits`` for quantum augmentation.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int = 2,
                 dropout: float = 0.1,
                 mode: str = "classical",
                 n_qubits: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.ffn_dim = ffn_dim
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.mode = mode
        self.n_qubits = n_qubits

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)

        self.blocks = nn.ModuleList([
            HybridTransformerBlock(embed_dim, num_heads, ffn_dim,
                                   mode=mode, n_qubits=n_qubits,
                                   dropout=dropout)
            for _ in range(num_blocks)
        ])

        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, input_ids: torch.Tensor,
                rotation_params=None,
                entangle_params=None) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        x = self.pos_encoder(x)
        for block in self.blocks:
            x = block(x, rotation_params=rotation_params, entangle_params=entangle_params)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = ["UnifiedAttentionTransformer"]
