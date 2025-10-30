import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

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

class HybridConvFraudLayer(nn.Module):
    """
    Classical hybrid layer combining a 2‑D convolution with a fraud‑detection‑style
    fully‑connected sequence.  The convolution produces a scalar feature that is
    repeated to match the 2‑D input expected by the fraud network.
    """

    def __init__(
        self,
        conv_kernel_size: int = 2,
        conv_threshold: float = 0.0,
        fraud_input_params: FraudLayerParameters | None = None,
        fraud_layers_params: Iterable[FraudLayerParameters] | None = None,
        clip: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=conv_kernel_size, bias=True)
        self.conv_threshold = conv_threshold
        self.fraud_network = build_fraud_detection_program(
            fraud_input_params or FraudLayerParameters(
                bs_theta=0.0,
                bs_phi=0.0,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(1.0, 1.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            ),
            fraud_layers_params or [],
        )
        self.clip = clip

    def run(self, data: np.ndarray) -> float:
        """
        Run the hybrid pipeline.

        Parameters
        ----------
        data
            2‑D array matching the convolution kernel size.

        Returns
        -------
        float
            Final scalar output after the fraud network.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.conv.kernel_size, self.conv.kernel_size)
        conv_logits = self.conv(tensor)
        conv_out = torch.sigmoid(conv_logits - self.conv_threshold).mean()
        conv_scalar = conv_out.item()
        # Fraud network expects a 2‑D vector; we duplicate the scalar.
        fraud_input = torch.tensor([conv_scalar, conv_scalar], dtype=torch.float32)
        fraud_out = self.fraud_network(fraud_input)
        return fraud_out.item()

    def forward(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(self.run(data), dtype=torch.float32)

__all__ = ["HybridConvFraudLayer"]
