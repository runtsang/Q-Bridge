import torch
from torch import nn
from torch.nn import functional as F

def EstimatorQNN():
    """Return a deep neural network with advanced regularisation."""
    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(2, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 1),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.block(inputs)

    return EstimatorNN()

__all__ = ["EstimatorQNN"]
