import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionBase(nn.Module):
    """Base class for multi‑head attention."""
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
    """Standard multi‑head attention implemented with PyTorch's MultiheadAttention."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:  # type: ignore[override]
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output

class QuantumSimAttention(nn.Module):
    """Simple linear projection that mimics a quantum circuit."""
    def __init__(self, embed_dim: int, n_qubits: int = 8) -> None:
        super().__init__()
        self.project = nn.Linear(embed_dim, n_qubits, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x)

class MultiHeadAttentionQuantum(MultiHeadAttentionBase):
    """Quantum‑enabled projection of the linear layers used in the attention heads."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1, n_qubits: int = 8) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.n_qubits = n_qubits
        self.q_proj = QuantumSimAttention(embed_dim, n_qubits)
        self.combine = nn.Linear(n_qubits, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        # reshape for heads
        x = x.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)  # (B, H, S, Dk)
        outputs = []
        for head in x.unbind(dim=1):
            head_proj = self.q_proj(head)  # (B, S, n_qubits)
            outputs.append(head_proj)
        q_out = torch.stack(outputs, dim=1)  # (B, H, S, n_qubits)
        q_out = self.combine(q_out.reshape(batch_size, seq_len, -1))
        return q_out

class HybridAttention(nn.Module):
    """Learnable mix between classical and quantum attention."""
    def __init__(self, classical: nn.Module, quantum: nn.Module) -> None:
        super().__init__()
        self.classical = classical
        self.quantum = quantum
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        cls = self.classical(x, mask)
        qnt = self.quantum(x, mask)
        return self.gate * qnt + (1 - self.gate) * cls

class FeedForwardBase(nn.Module):
    """Base feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    """Standard two‑layer perceptron."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class FeedForwardQuantum(FeedForwardBase):
    """Quantum feed‑forward realized by a simple circuit."""
    def __init__(self, embed_dim: int, ffn_dim: int, n_qubits: int = 8, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.n_qubits = n_qubits
        self.linear1 = nn.Linear(embed_dim, n_qubits)
        self.linear2 = nn.Linear(n_qubits, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simulate quantum effect with a linear layer
        out = self.linear1(x)
        out = self.linear2(self.dropout(F.relu(out)))
        return out

class HybridFeedForward(nn.Module):
    """Learnable mix between classical and quantum feed‑forward."""
    def __init__(self, classical: nn.Module, quantum: nn.Module) -> None:
        super().__init__()
        self.classical = classical
        self.quantum = quantum
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls = self.classical(x)
        qnt = self.quantum(x)
        return self.gate * qnt + (1 - self.gate) * cls

class TransformerBlockBase(nn.Module):
    """Base transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class TransformerBlockClassical(TransformerBlockBase):
    """Purely classical transformer block."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class TransformerBlockQuantum(TransformerBlockBase):
    """Hybrid block mixing classical and quantum sub‑modules."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = HybridAttention(
            MultiHeadAttentionClassical(embed_dim, num_heads, dropout),
            MultiHeadAttentionQuantum(embed_dim, num_heads, dropout)
        )
        self.ffn = HybridFeedForward(
            FeedForwardClassical(embed_dim, ffn_dim, dropout),
            FeedForwardQuantum(embed_dim, ffn_dim, n_qubits=8, dropout=dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

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
        return x + self.pe[:, : x.size(1)]

class TextClassifier(nn.Module):
    """Transformer‑based text classifier with optional quantum sub‑modules."""
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
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoder(embed_dim)
        block_cls = TransformerBlockQuantum if use_quantum else TransformerBlockClassical
        self.transformers = nn.Sequential(
            *[block_cls(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(x)
        x = self.pos_embedding(tokens)
        x = self.transformers(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionQuantum",
    "HybridAttention",
    "FeedForwardBase",
    "FeedForwardClassical",
    "FeedForwardQuantum",
    "HybridFeedForward",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "TextClassifier",
]
