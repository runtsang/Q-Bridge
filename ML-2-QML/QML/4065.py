"""Hybrid transformer with quantum submodules and optional QCNN feature extractor."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from.QCNN import QCNN
from.FastBaseEstimator import FastEstimator
from.QTransformerTorch import PositionalEncoder

class QuantumAttention(nn.Module):
    """Placeholder quantum attention that mimics the interface of the QML version."""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.combine = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return self.combine(out)

class QuantumFeedForward(nn.Module):
    """Placeholder quantum feed‑forward block."""
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockQuantum(nn.Module):
    """Quantum transformer block that uses the placeholder quantum sub‑modules."""
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = QuantumAttention(embed_dim, num_heads, dropout)
        self.ffn = QuantumFeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class HybridTransformer(nn.Module):
    """Quantum hybrid transformer that replaces each classical block with a quantum one
    and optionally prepends a QCNN quantum feature extractor."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        use_qcnn: bool = False,
    ) -> None:
        super().__init__()
        self.use_qcnn = use_qcnn
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[
                TransformerBlockQuantum(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        if use_qcnn:
            self.qcnn = QCNN()
            self.qcnn_fc = nn.Linear(1, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_qcnn:
            flat = x.view(-1, 1).float()
            qcnn_out = self.qcnn(flat)
            qcnn_out = qcnn_out.view(x.size(0), x.size(1), -1)
            x = self.qcnn_fc(qcnn_out)
        else:
            x = self.token_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

    def evaluate(
        self,
        observables,
        parameter_sets,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> list[list[complex]]:
        """
        Evaluate the quantum hybrid model using FastEstimator.
        """
        estimator = FastEstimator(self)
        return estimator.evaluate(observables, parameter_sets,
                                  shots=shots, seed=seed)

__all__ = ["HybridTransformer"]
