import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Tuple

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

class FraudDetector(nn.Module):
    """
    Enhanced classical fraud detection model.
    Builds a sequential network from a list of FraudLayerParameters.
    Supports optional dropout and batch normalization.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        hidden_params: Iterable[FraudLayerParameters],
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        clip: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []

        # Input layer (no clipping)
        layers.append(_layer_from_params(input_params, clip=False))

        # Hidden layers
        for params in hidden_params:
            layers.append(_layer_from_params(params, clip=clip))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(2))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(2, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the sequential model.
        """
        return self.model(inputs)

    @staticmethod
    def build_fraud_detection_model(
        input_params: FraudLayerParameters,
        hidden_params: Iterable[FraudLayerParameters],
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        clip: bool = True,
    ) -> "FraudDetector":
        """
        Convenience method to instantiate the FraudDetector.
        """
        return FraudDetector(
            input_params=input_params,
            hidden_params=hidden_params,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            clip=clip,
        )

__all__ = ["FraudLayerParameters", "FraudDetector"]
