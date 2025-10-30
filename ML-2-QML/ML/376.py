"""Enhanced regression neural network with dropout, batchnorm, and a training helper.

The original EstimatorQNN returned a tiny 3‑layer network.  This version
expands the depth, adds regularisation, and provides a small training
helper that can be used in downstream experiments.  It is fully
compatible with PyTorch's autograd and can be plugged into any
training loop.

The class is named EstimatorQNN to match the QML counterpart, enabling
a one‑to‑one mapping between classical and quantum experiments.
"""

import torch
from torch import nn
from torch.nn import functional as F

class EstimatorQNN(nn.Module):
    """A richer fully‑connected regressor.

    Architecture:
        - Input (dim 2) → Linear(8) → BatchNorm1d → Dropout(0.1) → Tanh
        - Linear(8) → BatchNorm1d → Dropout(0.1) → Tanh
        - Linear(4) → BatchNorm1d → Dropout(0.1) → Tanh
        - Linear(1)
    """

    def __init__(self, input_dim: int = 2, hidden_dims: list[int] | tuple[int] = (8, 8, 4)) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(p=0.1))
            layers.append(nn.Tanh())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning a scalar prediction."""
        return self.net(x)

    def train_on_batch(self, x: torch.Tensor, y: torch.Tensor,
                       optimiser: torch.optim.Optimizer,
                       loss_fn: nn.Module = nn.MSELoss()) -> float:
        """Convenience training step on a single batch."""
        self.train()
        optimiser.zero_grad()
        pred = self(x)
        loss = loss_fn(pred.squeeze(-1), y)
        loss.backward()
        optimiser.step()
        return loss.item()

__all__ = ["EstimatorQNN"]
