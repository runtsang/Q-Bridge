import math
import torch
from torch import nn

class EstimatorQNN(nn.Module):
    """
    A deeper feed‑forward regression network with batch normalization and dropout.
    Compatible with the original EstimatorQNN interface.
    """
    def __init__(self, input_dim: int = 2, hidden_dims: list[int] = [16, 32, 16], output_dim: int = 1) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.BatchNorm1d(hdim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=0.2))
            prev_dim = hdim
        self.feature_extractor = nn.Sequential(*layers)
        self.regressor = nn.Linear(prev_dim, output_dim)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        return self.regressor(x)

__all__ = ["EstimatorQNN"]
