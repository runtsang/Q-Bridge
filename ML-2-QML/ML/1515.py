import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Classical building blocks (identical to the seed implementation)
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Base class for attention layers – keeps the same interface as the seed."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented with PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out, _ = self.attn(x, x, x, key_padding_mask=mask)
        return out


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward layers."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)


class FeedForwardClassical(FeedForwardBase):
    """Simple 2‑layer MLP."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# --------------------------------------------------------------------------- #
#  Lightweight variational quantum layer (pure‑torch implementation)
# --------------------------------------------------------------------------- #
class VariationalQuantumLayer(nn.Module):
    """
    A toy variational layer that mimics a quantum circuit.  It projects the input
    into a lower‑dimensional space, applies trainable RX rotations, and then
    expands back to the original embedding size.  The layer is wrapped as a
    torch.nn.Module so it can be inserted into a standard nn.Sequential pipeline.
    """
    def __init__(self, embed_dim: int, n_qubits: int = 2) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        # Encoder: linear projection into n_qubits
        self.encoder = nn.Linear(embed_dim, n_qubits)
        # Trainable rotation angles for each qubit
        self.angles = nn.Parameter(torch.randn(n_qubits))
        # Decoder: linear layer to return to original dimensionality
        self.decoder = nn.Linear(n_qubits, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (batch, seq_len, embed_dim)
        Output shape: (batch, seq_len, embed_dim)
        """
        batch, seq, _ = x.shape
        # Flatten batch & seq for matrix multiplication
        flat = x.reshape(batch * seq, -1)
        # Encode
        enc = self.encoder(flat)  # (B*L, n_qubits)
        # Apply RX rotations (simple cosine as expectation of Z after RX)
        rot = torch.cos(2 * self.angles.expand_as(enc))
        # Element‑wise multiplication to simulate rotation
        enc_rot = enc * rot
        # Decode back
        dec = self.decoder(enc_rot)
        return dec.reshape(batch, seq, self.embed_dim)


# --------------------------------------------------------------------------- #
#  Hybrid transformer block
# --------------------------------------------------------------------------- #
class TransformerBlockBase(nn.Module):
    """Base class that preserves LayerNorm and dropout semantics."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)


class TransformerBlockHybrid(nn.Module):
    """
    Hybrid block that combines the classical attention and feed‑forward
    sub‑modules with an optional variational quantum layer.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        quantum_depth: int = 0,  # 0 = no quantum sub‑module
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.quantum_depth = quantum_depth
        self.quantum_layer = (
            VariationalQuantumLayer(embed_dim, n_qubits=quantum_depth)
            if quantum_depth > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical attention
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))

        # Classical feed‑forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        # Optional quantum sub‑module
        if self.quantum_depth > 0:
            q_out = self.quantum_layer(x)
            x = x + self.dropout(q_out)  # residual connection

        return x


# --------------------------------------------------------------------------- #
#  Positional encoding (identical to seed)
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
#  Text classifier that can switch between classical and hybrid blocks
# --------------------------------------------------------------------------- #
class TextClassifier(nn.Module):
    """
    Transformer‑based text classifier supporting a hybrid quantum‑classical
    architecture.  The `quantum_depth` flag controls whether a variational
    quantum layer is inserted after each transformer block.
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
        quantum_depth: int = 0,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.blocks = nn.Sequential(
            *[
                TransformerBlockHybrid(
                    embed_dim,
                    num_heads,
                    ffn_dim,
                    dropout,
                    quantum_depth,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            embed_dim, num_classes if num_classes > 2 else 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)          # (B, L, E)
        x = self.pos_encoder(tokens)              # add positional info
        x = self.blocks(x)                        # transformer blocks
        x = x.mean(dim=1)                         # global average pooling
        x = self.dropout(x)
        return self.classifier(x)


__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "FeedForwardBase",
    "FeedForwardClassical",
    "VariationalQuantumLayer",
    "TransformerBlockBase",
    "TransformerBlockHybrid",
    "PositionalEncoder",
    "TextClassifier",
]
