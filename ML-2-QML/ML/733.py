import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNGen409(nn.Module):
    """
    A deeper classical sampler neural network with residual connections, batch
    normalization and dropout. It mirrors the original two‑layer architecture
    but adds expressive power while keeping the interface identical.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        # First block
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # Second block with a residual shortcut
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # Output layer
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.block1(x)
        # Residual connection
        out = self.block2(out) + out
        logits = self.output(out)
        return F.softmax(logits, dim=-1)

__all__ = ["SamplerQNNGen409"]
