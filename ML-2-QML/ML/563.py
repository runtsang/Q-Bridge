import torch
from torch import nn

class EstimatorQNN(nn.Module):
    """
    A robust regression network that combines a feature extractor,
    a residual block, and dropout for regularization.
    """
    def __init__(self) -> None:
        super().__init__()
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(2, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        # Residual transformation
        self.residual = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        # Output head
        self.output = nn.Linear(16, 1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual addition and dropout.
        """
        features = self.feature_extractor(x)
        res = self.residual(features)
        out = features + res
        out = self.dropout(out)
        return self.output(out)

__all__ = ["EstimatorQNN"]
