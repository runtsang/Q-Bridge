import torch
from torch import nn
import torch.nn.functional as F

class EstimatorQNN(nn.Module):
    """
    A modern fully‑connected regression network.

    Architecture:
        - Input layer: 2 neurons
        - Hidden layer 1: 16 neurons, ReLU, BatchNorm, 0.1 Dropout
        - Hidden layer 2: 8 neurons, ReLU, BatchNorm, 0.1 Dropout
        - Output layer: 1 neuron (linear)

    The added regularisation stabilises training on noisy data and
    improves generalisation compared to the original 2‑layer toy model.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(8, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper that returns the network output without gradients.
        """
        with torch.no_grad():
            return self.forward(x)

__all__ = ["EstimatorQNN"]
