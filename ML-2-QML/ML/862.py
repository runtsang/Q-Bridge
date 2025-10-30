"""Enhanced classical estimator with flexible architecture and evaluation metrics."""

import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple

class EstimatorQNN(nn.Module):
    """
    A flexible feed‑forward regressor that supports arbitrary hidden layer sizes,
    dropout, and optional GPU acceleration.  The network is built from a list of
    hidden layer dimensions, e.g. [8, 4] produces the same topology as the seed.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: List[int] = None,
        dropout: float = 0.0,
        device: str = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [8, 4]
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        self.device = torch.device(device or "cpu")
        self.to(self.device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs.to(self.device))

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return predictions without gradient tracking."""
        with torch.no_grad():
            return self.forward(inputs)

    def evaluate(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Compute RMSE and R² on a batch of data.
        Returns (rmse, r2).
        """
        preds = self.predict(inputs)
        rmse = torch.sqrt(F.mse_loss(preds, targets))
        ss_res = torch.sum((targets - preds) ** 2)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
        r2 = 1 - ss_res / ss_tot
        return rmse.item(), r2.item()

__all__ = ["EstimatorQNN"]
