import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable

# --------------------------------------------------------------------------- #
# Classical stand‑in for the fully‑connected quantum layer (FCL)
# --------------------------------------------------------------------------- #
class FCL(nn.Module):
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

# --------------------------------------------------------------------------- #
# Classical Quanvolution filter (inspired by reference 3)
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

# --------------------------------------------------------------------------- #
# Classical self‑attention (Multi‑head) inspired by reference 4
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int = 4, num_heads: int = 1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, seq_len, embed_dim]
        attn_output, _ = self.attn(x, x, x)
        return attn_output

# --------------------------------------------------------------------------- #
# Hybrid model combining the components
# --------------------------------------------------------------------------- #
class HybridNATModel(nn.Module):
    """
    Classical hybrid architecture that mirrors the quantum‑inspired
    Quantum‑NAT construction.  The flow is:
        1. 2×2 quanvolution filter
        2. Multi‑head self‑attention over the patch embeddings
        3. Linear classification head
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.attention = ClassicalSelfAttention(embed_dim=4, num_heads=1)
        self.classifier = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Quanvolution
        features = self.qfilter(x)                     # [B, 4*14*14]
        # 2. Reshape for self‑attention: seq_len=14*14, embed_dim=4
        seq = features.view(x.size(0), 14 * 14, 4)      # [B, seq_len, embed_dim]
        attn_out = self.attention(seq)                 # [B, seq_len, embed_dim]
        # 3. Flatten and classify
        flat = attn_out.reshape(x.size(0), -1)          # [B, 4*14*14]
        logits = self.classifier(flat)                 # [B, num_classes]
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridNATModel"]
