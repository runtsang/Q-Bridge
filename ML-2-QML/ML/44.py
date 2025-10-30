import torch
from torch import nn
import torch.nn.functional as F

class _EstimatorQNN(nn.Module):
    """Deep residual MLP for regression.

    Architecture:
        Input (2) → Linear → BatchNorm → ReLU
        4 residual blocks, each:
            Linear → BatchNorm → ReLU
            Linear → BatchNorm
            Add skip connection → ReLU
        Dropout(0.2)
        Linear → Output (1)
    """
    def __init__(self, hidden_dim: int = 32, num_residual: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_residual):
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            )
            self.residual_blocks.append(block)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.input_layer(x)
        for block in self.residual_blocks:
            residual = out
            out = block(out)
            out += residual
            out = F.relu(out)
        out = self.dropout(out)
        return self.output_layer(out)

def EstimatorQNN():
    """Return a fresh instance of the residual MLP."""
    return _EstimatorQNN()

__all__ = ["EstimatorQNN"]
