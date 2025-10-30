import torch
import torch.nn as nn
import numpy as np
from typing import Iterable

class FCL(nn.Module):
    """Enhanced fully‑connected layer with dropout, batch‑norm and
    parameter‑override for variational optimisation."""
    def __init__(self, in_features: int, out_features: int = 1,
                 dropout: float = 0.0, use_batchnorm: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.bn = nn.BatchNorm1d(out_features) if use_batchnorm else nn.Identity()
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor, params: torch.Tensor | None = None) -> torch.Tensor:
        if params is not None:
            weight, bias = torch.unbind(torch.split(params,
                                                    [self.linear.weight.numel(),
                                                     self.linear.bias.numel()],
                                                    dim=0))
            self.linear.weight.data.copy_(weight.view_as(self.linear.weight))
            self.linear.bias.data.copy_(bias)
        out = self.linear(x)
        out = self.dropout(out)
        out = self.bn(out)
        out = self.activation(out)
        return out.mean(dim=0)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Compatibility wrapper that accepts a flat list of parameters."""
        params = torch.tensor(list(thetas), dtype=torch.float32)
        dummy_input = torch.ones(1, self.linear.in_features)
        return self.forward(dummy_input, params).detach().numpy()

__all__ = ["FCL"]
