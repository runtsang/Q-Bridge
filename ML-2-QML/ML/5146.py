import math
import itertools
from typing import Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx

# Classical transformer components
class MultiHeadAttentionBase(nn.Module):
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
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output

class MultiHeadAttentionQuantum(MultiHeadAttentionClassical):
    """Alias for API compatibility; quantum implementation resides in QML module."""
    pass

class FeedForwardBase(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class FeedForwardClassical(FeedForwardBase):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class FeedForwardQuantum(FeedForwardClassical):
    """Alias for API compatibility."""
    pass

class TransformerBlockBase(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class TransformerBlockClassical(TransformerBlockBase):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        attn_module: Optional[nn.Module] = None,
        ffn_module: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = attn_module if attn_module is not None else MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = ffn_module if ffn_module is not None else FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class TransformerBlockQuantum(TransformerBlockClassical):
    """Alias for API compatibility."""
    pass

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
        return x + self.pe[:, :x.size(1)]

# Classical kernel utilities
class KernalAnsatz(nn.Module):
    """RBF kernel ansatz."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wraps KernalAnsatz."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# Graph utilities based on fidelity
def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(states: Sequence[torch.Tensor], threshold: float,
                       *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# Quanvolution utilities
class QuanvolutionFilter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

# Hybrid transformer combining all
class HybridTransformer(nn.Module):
    """
    A versatile transformer that can process text or image data, optionally using
    quantum attention, quantum feed‑forward layers, a quantum kernel‑based graph mask, and a quanvolution filter.
    """
    def __init__(
        self,
        input_type: str = "text",
        vocab_size: int = 30522,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_blocks: int = 12,
        ffn_dim: int = 3072,
        num_classes: int = 2,
        dropout: float = 0.1,
        use_quantum_attention: bool = False,
        use_quantum_ffn: bool = False,
        use_graph_attention: bool = False,
        use_quanvolution: bool = False,
        kernel_gamma: float = 1.0,
        graph_threshold: float = 0.8,
        graph_secondary: float | None = None,
    ) -> None:
        super().__init__()
        self.input_type = input_type
        self.use_graph_attention = use_graph_attention
        self.kernel_gamma = kernel_gamma
        self.graph_threshold = graph_threshold
        self.graph_secondary = graph_secondary

        if input_type == "text":
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)
            self.pos_embedding = PositionalEncoder(embed_dim)

            # Quantum modules for attention and feed‑forward
            attn_module = MultiHeadAttentionQuantum(embed_dim, num_heads, dropout) if use_quantum_attention else None
            ffn_module = FeedForwardQuantum(embed_dim, ffn_dim, dropout) if use_quantum_ffn else None

            # Build transformer blocks
            self.transformers = nn.Sequential(*[
                TransformerBlockClassical(embed_dim, num_heads, ffn_dim, dropout,
                                         attn_module=attn_module,
                                         ffn_module=ffn_module)
                for _ in range(num_blocks)
            ])

            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(embed_dim, num_classes if num_classes > 2 else 1)
        elif input_type == "image":
            self.qfilter = QuanvolutionFilter()
            self.linear_head = nn.Linear(4 * 14 * 14, num_classes if num_classes > 2 else 1)
        else:
            raise ValueError(f"Unsupported input_type {input_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_type == "text":
            tokens = self.token_embedding(x)
            x = self.pos_embedding(tokens)

            # Optional graph‑based attention mask
            mask: Optional[torch.Tensor] = None
            if self.use_graph_attention:
                emb = tokens.detach().cpu().numpy()
                mask_np = kernel_matrix(emb, emb, self.kernel_gamma)
                mask_np = (mask_np >= self.graph_threshold).astype(np.float32)
                mask = torch.tensor(mask_np, device=x.device).bool()

            x = self.transformers(x)
            x = self.dropout(x.mean(dim=1))
            return self.classifier(x)
        else:  # image
            features = self.qfilter(x)
            logits = self.linear_head(features)
            return logits

__all__ = [
    "MultiHeadAttentionBase",
    "MultiHeadAttentionClassical",
    "MultiHeadAttentionQuantum",
    "FeedForwardBase",
    "FeedForwardClassical",
    "FeedForwardQuantum",
    "TransformerBlockBase",
    "TransformerBlockClassical",
    "TransformerBlockQuantum",
    "PositionalEncoder",
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "state_fidelity",
    "fidelity_adjacency",
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "HybridTransformer",
]
