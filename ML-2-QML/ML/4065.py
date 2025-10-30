"""Hybrid transformer with optional QCNN feature extractor and fast evaluation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from.QCNN import QCNN, QCNNModel
from.FastBaseEstimator import FastEstimator
from.QTransformerTorch import PositionalEncoder, TransformerBlockClassical

class HybridTransformer(nn.Module):
    """
    Classical transformer model that optionally prepends a QCNN feature extractor.
    Provides a fast evaluation routine via FastEstimator.
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
        use_qcnn: bool = False,
    ) -> None:
        super().__init__()
        self.use_qcnn = use_qcnn
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        if use_qcnn:
            # QCNN expects 8‑dimensional input; map token indices to a scalar feature
            self.qcnn = QCNN()
            self.qcnn_fc = nn.Linear(1, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. x is a tensor of token indices (batch, seq_len).
        """
        if self.use_qcnn:
            flat = x.view(-1, 1).float()
            qcnn_out = self.qcnn(flat)          # (batch*seq_len, 1)
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
    ) -> list[list[float]]:
        """
        Evaluate the model on a list of parameter sets using FastEstimator.
        """
        estimator = FastEstimator(self)
        return estimator.evaluate(observables, parameter_sets,
                                  shots=shots, seed=seed)

__all__ = ["HybridTransformer"]
