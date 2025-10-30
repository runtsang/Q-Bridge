import torch
from torch import nn

def EstimatorQNN():
    """Return a robust fully‑connected regression network with batch‑norm and dropout."""
    class EstimatorNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 8),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(8, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(16, 8),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(8, 1),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.net(inputs)

    return EstimatorNN()


__all__ = ["EstimatorQNN"]
