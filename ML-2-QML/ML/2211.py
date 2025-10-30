import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Sequence

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

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

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class Conv(nn.Module):
    """
    Classical convolutional filter with a fraud‑detection style fully‑connected head.
    The convolution is followed by a thresholded sigmoid and a two‑feature
    representation that is fed into a sequential network mirroring the
    photonic fraud‑detection architecture.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        fraud_input_params: FraudLayerParameters | None = None,
        fraud_layer_params: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # Default fraud‑detection parameters if none supplied
        if fraud_input_params is None:
            fraud_input_params = FraudLayerParameters(
                bs_theta=0.0,
                bs_phi=0.0,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(0.0, 0.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
        if fraud_layer_params is None:
            fraud_layer_params = []

        self.fraud_net = build_fraud_detection_program(
            fraud_input_params, fraud_layer_params
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Output of the fraud‑detection head of shape (batch, 1).
        """
        conv_out = self.conv(x)
        conv_out = torch.sigmoid(conv_out - self.threshold)
        # Reduce to two features: mean over spatial dims duplicated
        mean = conv_out.mean(dim=[2, 3])  # shape (batch, 1)
        features = torch.cat([mean, mean], dim=1)  # shape (batch, 2)
        return self.fraud_net(features)

__all__ = ["Conv", "FraudLayerParameters", "build_fraud_detection_program"]
