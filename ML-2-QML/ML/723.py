import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedSamplerQNN(nn.Module):
    """
    A deeper, regularized MLP for probability distribution sampling.
    Supports configurable hidden layers, dropout, and batch normalization.
    """
    def __init__(self, input_dim=2, hidden_dims=(64, 32), output_dim=2, dropout=0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a probability distribution over the output classes."""
        return F.softmax(self.net(x), dim=-1)

__all__ = ["AdvancedSamplerQNN"]
