import torch
from torch import nn
import torch.nn.functional as F

class EstimatorQNNMod(nn.Module):
    """A deep residual regressor with dropout and L2 regularisation.

    The architecture mirrors the original two‑layer network but
    incorporates skip connections and dropout to mitigate overfitting
    on small datasets. The forward method accepts a tensor of shape
    (batch, 2) and returns a scalar prediction.
    """
    def __init__(self, dropout: float = 0.1, l2_reg: float = 1e-4) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.l2_reg = l2_reg
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
        )
        self.res_block = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
        )
        self.output = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        residual = self.res_block(x)
        x = x + residual  # residual connection
        x = self.dropout(x)
        return self.output(x)

    def l2_loss(self) -> torch.Tensor:
        """Return L2 penalty on all parameters."""
        return sum(p.pow(2).sum() for p in self.parameters()) * self.l2_reg

__all__ = ["EstimatorQNNMod"]
