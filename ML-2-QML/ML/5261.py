"""Combined classical sampler, transformer, and classifier pipeline."""

from __future__ import annotations

import torch
import torch.nn as nn

# Seed modules
from SamplerQNN import SamplerQNN
from QTransformerTorch import TransformerBlockClassical, PositionalEncoder
from QuantumClassifierModel import build_classifier_circuit


class SamplerQNNGen145(nn.Module):
    """
    Classical pipeline that fuses a sampler network, a transformer encoder,
    and a variational classifier, mirroring the quantum architecture.
    """
    def __init__(
        self,
        seq_len: int = 10,
        embed_dim: int = 8,
        num_heads: int = 2,
        ffn_dim: int = 16,
        depth: int = 2,
    ):
        super().__init__()
        # Sampler network
        self.sampler = SamplerQNN()

        # Transformer encoder
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlockClassical(embed_dim, num_heads, ffn_dim) for _ in range(depth)]
        )

        # Linear projection from sampler output (2‑dim) to transformer embedding
        self.embed_proj = nn.Linear(2, embed_dim)

        # Classifier built via the same interface as the quantum helper
        self.classifier, _, _, _ = build_classifier_circuit(num_features=embed_dim, depth=depth)

        self.seq_len = seq_len
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, 2) representing raw 2‑D features
               for each timestep.
        Returns:
            logits: Tensor of shape (batch, 2)
        """
        batch, seq_len, _ = x.shape

        # Sample probabilities for each timestep
        samp_logits = []
        for i in range(seq_len):
            samp_logits.append(self.sampler(x[:, i, :]))
        samp_seq = torch.stack(samp_logits, dim=1)  # (batch, seq_len, 2)

        # Project to transformer embedding dimension
        embed = self.embed_proj(samp_seq)

        # Positional encoding
        embed = self.pos_encoder(embed)

        # Transformer encoder
        trans_out = self.transformer(embed)  # (batch, seq_len, embed_dim)

        # Pooling and classification
        pooled = trans_out.mean(dim=1)  # (batch, embed_dim)
        logits = self.classifier(pooled)  # (batch, 2)

        return logits
