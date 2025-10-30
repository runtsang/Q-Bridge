import torch
from torch import nn
import numpy as np

class HybridEstimatorQNN(nn.Module):
    """Classical hybrid estimator that first applies a self‑attention block
    to the raw inputs and then feeds the attended representation into a
    small fully‑connected regression head."""
    def __init__(self, input_dim: int, embed_dim: int = 4, hidden_sizes: list[int] | None = None):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [8, 4]
        # Linear projection from input to embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)
        # Self‑attention layer (single‑head, learnable query/key/value matrices)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        # Regression head
        layers = []
        last_dim = embed_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_dim, size))
            layers.append(nn.Tanh())
            last_dim = size
        layers.append(nn.Linear(last_dim, 1))
        self.regression = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        x_proj = self.input_proj(x)                      # (batch, embed_dim)
        # attention expects (batch, seq_len, embed_dim); treat each sample as a single token
        attn_output, _ = self.attention(x_proj.unsqueeze(1), x_proj.unsqueeze(1), x_proj.unsqueeze(1))
        attn_output = attn_output.squeeze(1)             # (batch, embed_dim)
        return self.regression(attn_output)

def EstimatorQNN() -> nn.Module:
    """Convenience factory mirroring the original EstimatorQNN API."""
    return HybridEstimatorQNN(input_dim=2)

__all__ = ["EstimatorQNN", "HybridEstimatorQNN"]
