import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional


# --------------------------------------------------------------------------- #
#  Classical kernel utilities (adapted from the original seed)
# --------------------------------------------------------------------------- #
class RBFKernel(nn.Module):
    """Radial‑basis function kernel with learnable gamma."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - y.unsqueeze(0)      # (m, n, d)
        sq_norm = (diff ** 2).sum(dim=2)            # (m, n)
        return torch.exp(-self.gamma * sq_norm)


class KernelEmbedding(nn.Module):
    """Embed tokens via an RBF kernel against a set of learnable reference vectors."""
    def __init__(self, embed_dim: int, num_refs: Optional[int] = None, gamma: float = 1.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_refs = num_refs or embed_dim
        # Reference vectors act as centers of the kernel map
        self.refs = nn.Parameter(torch.randn(self.num_refs, embed_dim))
        self.kernel = RBFKernel(gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim) – token embeddings.
        Returns:
            (batch, seq_len, num_refs) – kernel‑mapped features.
        """
        batch, seq_len, _ = x.size()
        flat = x.view(-1, self.embed_dim)
        # Compute kernel between each token embedding and all reference vectors
        k = self.kernel(flat, self.refs)            # (batch*seq_len, num_refs)
        return k.view(batch, seq_len, self.num_refs)


# --------------------------------------------------------------------------- #
#  Positional encoding
# --------------------------------------------------------------------------- #
class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
#  Classical transformer blocks
# --------------------------------------------------------------------------- #
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.size()
        q = self.q_linear(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)          # (batch, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                       # (batch, num_heads, seq_len, d_k)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


# --------------------------------------------------------------------------- #
#  Unified classifier API
# --------------------------------------------------------------------------- #
class QTransformerClassifier(nn.Module):
    """
    A transformer‑based text classifier that can operate in three modes:

    1. Pure classical (default).
    2. Classical with a kernel‑based token embedding.
    3. Quantum‑enhanced (see the QML module).

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embed_dim : int
        Dimensionality of token embeddings.
    num_heads : int
        Number of attention heads.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Hidden dimension in the feed‑forward sub‑network.
    num_classes : int
        Output classes (1 for binary).
    dropout : float, default 0.1
        Drop‑out probability.
    use_kernel : bool, default False
        If True, replace the token embedding with a kernel mapping.
    kernel_gamma : float, default 1.0
        Gamma parameter for the RBF kernel.
    kernel_refs : int or None, default None
        Number of reference vectors for the kernel embedding. Defaults
        to ``embed_dim``.
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
        use_kernel: bool = False,
        kernel_gamma: float = 1.0,
        kernel_refs: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.use_kernel = use_kernel
        if use_kernel:
            self.token_embedding = KernelEmbedding(embed_dim, kernel_refs, kernel_gamma)
        else:
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        self.pos_encoder = PositionalEncoder(embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        out_dim = 1 if num_classes <= 2 else num_classes
        self.classifier = nn.Linear(embed_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.LongTensor
            Input token indices of shape (batch, seq_len).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes) or (batch, 1).
        """
        if self.use_kernel:
            emb = self.token_embedding(x)          # (batch, seq_len, num_refs)
        else:
            emb = self.token_embedding(x)          # (batch, seq_len, embed_dim)

        emb = self.pos_encoder(emb)
        for block in self.blocks:
            emb = block(emb)

        pooled = emb.mean(dim=1)                    # global average pooling
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


__all__ = ["QTransformerClassifier"]
