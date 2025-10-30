import torch
from torch import nn
import numpy as np

class HybridFCL(nn.Module):
    """
    Classical fully connected layer with dropout and batch normalization,
    extending the original simple linear layer.
    """
    def __init__(self, n_features: int = 1, n_hidden: int = 16, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(n_features, n_hidden)
        self.bn = nn.BatchNorm1d(n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor):
        """
        x: Tensor of shape (batch, n_features)
        Returns: Tensor of shape (batch, n_hidden)
        """
        x = self.linear(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

    def run(self, thetas):
        """
        thetas: Iterable[float] of length n_features
        Returns: np.ndarray of shape (1,)
        """
        x = torch.tensor(thetas, dtype=torch.float32).unsqueeze(0)  # (1, n_features)
        output = self.forward(x)
        return output.mean(dim=1, keepdim=True).numpy()

__all__ = ["HybridFCL"]
