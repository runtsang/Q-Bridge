from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from QTransformerTorch import TransformerBlockClassical, PositionalEncoder
from Autoencoder import Autoencoder, AutoencoderNet, AutoencoderConfig
from SelfAttention import SelfAttention

class HybridTransformerClassifier(nn.Module):
    """
    Classical transformer classifier that optionally incorporates a
    fully‑connected auto‑encoder and a classical self‑attention block.
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
        *,
        use_autoencoder: bool = False,
        autoencoder_config: AutoencoderConfig | None = None,
        use_self_attention: bool = False,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)

        self.transformers = nn.Sequential(
            *[
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )

        self.dropout = nn.Dropout(dropout)

        # Optional auto‑encoder
        if use_autoencoder:
            cfg = autoencoder_config or AutoencoderConfig(
                input_dim=embed_dim,
                latent_dim=max(4, embed_dim // 4),
                hidden_dims=(embed_dim, max(4, embed_dim // 2)),
                dropout=dropout,
            )
            self.autoencoder = Autoencoder(cfg)
        else:
            self.autoencoder = None

        # Optional classical self‑attention
        if use_self_attention:
            self.self_attention = SelfAttention()
            # Parameters for the classical self‑attention routine
            self.rotation_params = nn.Parameter(
                torch.randn(embed_dim, 4), requires_grad=True
            )
            self.entangle_params = nn.Parameter(
                torch.randn(embed_dim - 1), requires_grad=True
            )
        else:
            self.self_attention = None

        self.classifier = nn.Linear(
            embed_dim, num_classes if num_classes > 2 else 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token embedding
        tokens = self.token_embedding(x)  # (B, L, E)
        tokens = self.pos_embedding(tokens)

        # Optional auto‑encoder compression / decompression
        if self.autoencoder is not None:
            batch, seq_len, embed_dim = tokens.shape
            flat = tokens.view(-1, embed_dim)
            encoded = self.autoencoder.encode(flat)
            decoded = self.autoencoder.decode(encoded)
            tokens = decoded.view(batch, seq_len, embed_dim)

        # Transformer blocks
        x = self.transformers(tokens)

        # Optional classical self‑attention
        if self.self_attention is not None:
            # Convert to numpy for the legacy routine
            arr = x.detach().cpu().numpy()
            attn_out = self.self_attention.run(
                self.rotation_params.detach().cpu().numpy(),
                self.entangle_params.detach().cpu().numpy(),
                arr,
            )
            x = torch.from_numpy(attn_out).to(x.device)

        # Global pooling and classification
        pooled = x.mean(dim=1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)
