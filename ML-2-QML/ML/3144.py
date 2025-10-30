import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from QCNN import QCNNModel

class MultiHeadAttentionBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output

class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionBase(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
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

class HybridTextClassifier(nn.Module):
    """
    Transformer‑based text classifier that can optionally append a QCNN‑style
    feature extractor.  The API is intentionally identical to the original
    ``TextClassifier`` from the seed, but the implementation is extended
    to support both classical and quantum back‑ends.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 8,
        num_heads: int = 2,
        num_blocks: int = 2,
        ffn_dim: int = 32,
        num_classes: int = 2,
        dropout: float = 0.1,
        use_qcnn: bool = True,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformers = nn.Sequential(
            *[
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.use_qcnn = use_qcnn
        if use_qcnn:
            self.qcnn = QCNNModel()
            classifier_input_dim = 1
        else:
            self.qcnn = None
            classifier_input_dim = embed_dim
        self.classifier = nn.Linear(
            classifier_input_dim,
            num_classes if num_classes > 2 else 1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input token ids of shape (batch, seq_len).

        Returns
        -------
        torch.Tensor
            Logits or probability scores of shape (batch, num_classes) or
            (batch, 1) for binary classification.
        """
        tokens = self.token_embedding(x)          # (batch, seq_len, embed_dim)
        x = self.pos_encoder(tokens)              # (batch, seq_len, embed_dim)
        x = self.transformers(x)                  # (batch, seq_len, embed_dim)
        x = x.mean(dim=1)                         # (batch, embed_dim)
        if self.use_qcnn:
            x = self.qcnn(x)                      # (batch, 1)
            x = x.squeeze(-1)                     # (batch,)
        x = self.dropout(x)
        return self.classifier(x)

__all__ = [
    "MultiHeadAttentionBase",
    "FeedForwardClassical",
    "TransformerBlockClassical",
    "PositionalEncoder",
    "HybridTextClassifier",
]
