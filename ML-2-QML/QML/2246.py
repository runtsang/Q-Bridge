"""Unified QCNN‑Transformer (quantum‑enhanced transformer).

This module mirrors the classical implementation but replaces the
transformer blocks with quantum‑enhanced attention and feed‑forward
sub‑modules built with TorchQuantum.  The QCNN feature extractor
remains fully classical and is shared with the classical module.
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
# 1. Classical QCNN‑style feature extractor (unchanged)
# --------------------------------------------------------------------------- #
class _QCNNFeatureExtractor(nn.Module):
    """Same as in the classical module."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.pool3 = nn.Sequential(nn.Linear(4, 2), nn.Tanh())
        self.pool4 = nn.Sequential(nn.Linear(2, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.pool4(x))

# --------------------------------------------------------------------------- #
# 2. Quantum‑enhanced transformer components
# --------------------------------------------------------------------------- #
class _MultiHeadAttentionQuantum(nn.Module):
    """Quantum‑enhanced attention: each head is a variational circuit."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 n_qubits: int = 8,
                 q_device: Optional[tq.QuantumDevice] = None) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.n_qubits = n_qubits
        self.q_device = q_device or tq.QuantumDevice(n_wires=n_qubits)
        self.q_layer = self._build_layer()
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def _build_layer(self) -> tq.QuantumModule:
        class QLayer(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
                )
                self.param_layer = tq.RX(has_params=True, trainable=True)
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
                self.encoder(q_device, x)
                self.param_layer(q_device, wires=range(self.n_wires))
                return self.measure(q_device)
        return QLayer(self.n_qubits)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.size()
        return x.view(b, t, self.num_heads, self.d_k).transpose(1, 2)

    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, t, d = x.size()
        return x.transpose(1, 2).contiguous().view(b, t, h * d)

    def _quantum_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the quantum layer to each head vector."""
        batch_size, heads, seq_len, d_k = x.size()
        out = []
        for h in range(heads):
            head = x[:, h, :, :].view(batch_size * seq_len, d_k)
            qdev = self.q_device.copy(bsz=batch_size * seq_len, device=head.device)
            out.append(self.q_layer(head, qdev))
        out = torch.stack(out, dim=1)
        return out.view(batch_size, heads, seq_len, d_k)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, _, embed_dim = x.size()
        if embed_dim!= self.embed_dim:
            raise ValueError("Input embedding mismatch")
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        k = self.separate_heads(k)
        q = self.separate_heads(q)
        v = self.separate_heads(v)
        k = self._quantum_transform(k)
        q = self._quantum_transform(q)
        v = self._quantum_transform(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        out = torch.matmul(scores, v)
        out = self.merge_heads(out)
        return self.out_linear(out)

class _FeedForwardQuantum(nn.Module):
    """Feed‑forward implemented with a variational circuit."""
    def __init__(self,
                 embed_dim: int,
                 ffn_dim: int,
                 n_qubits: int = 8,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)
        self.n_qubits = n_qubits
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.q_layer = self._build_layer()
        self.linear1 = nn.Linear(n_qubits, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def _build_layer(self) -> tq.QuantumModule:
        class QLayer(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
                )
                self.param_layer = tq.RY(has_params=True, trainable=True)
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor, q_device: tq.QuantumDevice) -> torch.Tensor:
                self.encoder(q_device, x)
                self.param_layer(q_device, wires=range(self.n_wires))
                return self.measure(q_device)
        return QLayer(self.n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        out = []
        for i in range(seq_len):
            token = x[:, i, :].view(batch, self.n_qubits)
            qdev = self.q_device.copy(bsz=batch, device=token.device)
            out.append(self.q_layer(token, qdev))
        out = torch.stack(out, dim=1)  # (batch, seq_len, n_qubits)
        out = self.linear1(self.dropout(out))
        return self.linear2(F.relu(out))

class _TransformerBlockQuantum(nn.Module):
    """Transformer block that uses quantum attention and feed‑forward."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 n_qubits_transformer: int = 8,
                 n_qubits_ffn: int = 8,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = _MultiHeadAttentionQuantum(embed_dim, num_heads,
                                               dropout, n_qubits_transformer)
        self.ffn = _FeedForwardQuantum(embed_dim, ffn_dim,
                                       n_qubits_ffn, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding (identical to classical)."""
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

class UnifiedQCNNTransformer(nn.Module):
    """Quantum‑enhanced QCNN‑Transformer.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary.
    embed_dim : int
        Dimensionality of the transformer embeddings.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Width of the feed‑forward layers.
    num_classes : int
        Number of output classes.
    dropout : float, optional
        Drop‑out probability.
    n_qubits_transformer : int, optional
        Number of qubits used in each attention head.
    n_qubits_ffn : int, optional
        Number of qubits used in the feed‑forward variational circuit.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 num_classes: int,
                 dropout: float = 0.1,
                 n_qubits_transformer: int = 8,
                 n_qubits_ffn: int = 8) -> None:
        super().__init__()
        self.feature_extractor = _QCNNFeatureExtractor()
        self.transformer = nn.Sequential(
            *[ _TransformerBlockQuantum(embed_dim,
                                       num_heads,
                                       ffn_dim,
                                       n_qubits_transformer,
                                       n_qubits_ffn,
                                       dropout)
               for _ in range(num_blocks) ]
        )
        self.pos_embedding = PositionalEncoder(embed_dim)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) token indices
        features = self.feature_extractor(x.float())
        # Expand features to match sequence length for transformer
        seq = features.unsqueeze(1)  # (batch, 1, 1)
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformer(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

def QCNN() -> UnifiedQCNNTransformer:
    """Convenience factory matching the original QCNN API."""
    return UnifiedQCNNTransformer(
        vocab_size=10000,
        embed_dim=16,
        num_heads=4,
        num_blocks=2,
        ffn_dim=32,
        num_classes=2,
        dropout=0.1,
        n_qubits_transformer=8,
        n_qubits_ffn=8
    )

__all__ = [
    "QCNN",
    "UnifiedQCNNTransformer",
    "_QCNNFeatureExtractor",
    "_TransformerBlockQuantum",
    "PositionalEncoder",
]
