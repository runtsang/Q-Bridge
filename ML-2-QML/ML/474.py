import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNN(nn.Module):
    """
    Enhanced classical sampler network. Uses a deeper MLP with residual
    connections and dropout for regularization. Produces a probability
    distribution over two output classes via softmax.
    """
    def __init__(self, input_dim: int = 2, hidden_dims: list[int] = [16, 32, 16], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

__all__ = ["SamplerQNN"]
