import torch
import torch.nn as nn
import torch.nn.functional as F

class SamplerQNNGen(nn.Module):
    """
    Extended sampler network with a flexible depth and dropout regularisation.
    The network accepts a 2‑dimensional input and outputs a probability
    distribution over two classes.  Hidden layers can be customised via
    ``hidden_dims``.
    """
    def __init__(self, input_dim: int = 2, hidden_dims: list[int] | None = None, output_dim: int = 2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [8, 4]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.1))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

__all__ = ["SamplerQNNGen"]
