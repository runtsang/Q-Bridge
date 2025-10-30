import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class FraudLayerParameters:
    """
    Parameters for a single fully‑connected layer.

    Attributes
    ----------
    weight : torch.Tensor
        2×2 weight matrix.
    bias : torch.Tensor
        2‑element bias vector.
    activation : str
        Name of an activation in torch.nn (e.g. "tanh", "relu").
    dropout : float
        Dropout probability (0–1).
    residual : bool
        If True, add a residual connection (output = linear+input).
    """
    weight: torch.Tensor
    bias: torch.Tensor
    activation: str
    dropout: float
    residual: bool


class ResidualBlock(nn.Module):
    """Linear layer with activation, dropout and an optional residual skip."""
    def __init__(self, linear: nn.Linear, activation: nn.Module, dropout: nn.Module):
        super().__init__()
        self.linear = linear
        self.activation = activation
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.activation(out)
        out = self.dropout(out)
        return out + x


class FraudDetectionModel(nn.Module):
    """
    Extensible fraud‑detection classifier.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first (input) layer.
    layers : Iterable[FraudLayerParameters]
        Parameters for subsequent hidden layers.
    """
    def __init__(self, input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        modules: List[nn.Module] = [self._layer_from_params(input_params, clip=False)]
        modules.extend(self._layer_from_params(p, clip=True) for p in layers)
        modules.append(nn.Linear(2, 1))
        self.model = nn.Sequential(*modules)

    def _layer_from_params(self, params: FraudLayerParameters, *, clip: bool) -> nn.Module:
        # Clone the provided weight/bias to avoid unwanted side effects
        linear = nn.Linear(2, 2)
        with torch.no_grad():
            linear.weight.copy_(params.weight.clone())
            linear.bias.copy_(params.bias.clone())
        if clip:
            linear.weight.clamp_(-5.0, 5.0)
            linear.bias.clamp_(-5.0, 5.0)

        activation = getattr(nn, params.activation.capitalize())()
        dropout = nn.Dropout(params.dropout)

        if params.residual:
            return ResidualBlock(linear, activation, dropout)
        else:
            return nn.Sequential(linear, activation, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
