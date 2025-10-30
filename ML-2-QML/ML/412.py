import torch
from torch import nn


class EstimatorQNNModel(nn.Module):
    """
    Fully‑connected regression network that extends the original seed with
    batch‑normalisation, dropout and a flexible hidden‑layer configuration.
    The network accepts a 2‑dimensional input and predicts a single continuous
    target value.
    """
    def __init__(self, input_dim: int = 2, hidden_dims: tuple[int,...] = (64, 32, 16)):
        super().__init__()
        layers = []
        prev = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev, hd))
            layers.append(nn.BatchNorm1d(hd))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=0.1))
            prev = hd
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def EstimatorQNN() -> nn.Module:
    """
    Return an instance of EstimatorQNNModel.  The function signature mirrors
    the original seed so that downstream code can remain unchanged.
    """
    return EstimatorQNNModel()


__all__ = ["EstimatorQNN", "EstimatorQNNModel"]
