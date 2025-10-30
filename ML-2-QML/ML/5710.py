import torch
from torch import nn

class EstimatorQNN(nn.Module):
    """
    Robust regression network that extends the original seed.
    Architecture:
        - Input layer: 2 -> 32
        - BatchNorm1d
        - ReLU
        - Dropout(0.1)
        - Hidden layer: 32 -> 16
        - BatchNorm1d
        - ReLU
        - Dropout(0.1)
        - Output layer: 16 -> 1
    """
    def __init__(self, input_dim=2, hidden_dims=[32, 16], dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x)

__all__ = ["EstimatorQNN"]
