"""Enhanced classical feed‑forward regressor with residual blocks and optional variance output.

The original EstimatorQNN was a tiny two‑layer network.  
This upgrade introduces:

* Residual connections for deeper architectures.
* Dropout & batch‑normalisation for regularisation.
* An optional variance head for Bayesian‑style uncertainty estimates.

The model is fully PyTorch‑compatible and can be trained with any standard optimiser.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Simple residual block with two linear layers and ReLU activations."""
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.linear1(x))
        out = self.linear2(out)
        return F.relu(out + residual)


class EstimatorQNN(nn.Module):
    """
    Deep residual regression network.

    Parameters
    ----------
    input_dim : int, default 2
        Dimension of the input features.
    hidden_dim : int, default 64
        Width of hidden layers.
    output_dim : int, default 1
        Dimension of the regression output.
    dropout : float, default 0.1
        Dropout probability.
    use_variance : bool, default False
        If True, a second head predicts a positive variance via softplus.
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        output_dim: int = 1,
        dropout: float = 0.1,
        use_variance: bool = False,
    ) -> None:
        super().__init__()
        self.use_variance = use_variance
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        if use_variance:
            self.var_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.use_variance:
            var = F.softplus(self.var_head(x))
            return out, var
        return out


__all__ = ["EstimatorQNN"]
