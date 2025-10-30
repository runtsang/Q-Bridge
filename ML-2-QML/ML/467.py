import math
import torch
from torch import nn

class EstimatorQNN(nn.Module):
    """Extended feed‑forward regressor with dropout and batch normalization."""
    def __init__(self, input_dim: int = 2, hidden_dims: list[int] = [32, 16], output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

__all__ = ["EstimatorQNN"]
