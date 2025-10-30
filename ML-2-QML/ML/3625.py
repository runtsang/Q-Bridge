from __future__ import annotations

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
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
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

class HybridConvFraud(nn.Module):
    """Hybrid classical model combining a 2‑D convolution filter with a fraud‑detection fully‑connected network.
    The convolution acts as a feature extractor that mirrors the quantum quanvolution in the QML counterpart.  The
    extracted scalar is concatenated with a second dummy feature so that the fraud‑detection sub‑network receives a
    2‑dimensional input, exactly as in the photonic reference.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the square convolution kernel.  Default is 2.
    fraud_params : Iterable[FraudLayerParameters], optional
        Sequence of parameters defining the fraud‑detection layers.  The first element is used as the input
        layer (no clipping); subsequent layers are clipped to keep the weights bounded.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        fraud_params: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.threshold = 0.0
        params_list = list(fraud_params) if fraud_params is not None else []
        self.fraud_seq = (
            build_fraud_detection_program(params_list[0], params_list[1:])
            if params_list else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Fraud‑risk score of shape (batch, 1).
        """
        conv_logits = self.conv(x)
        conv_act = torch.sigmoid(conv_logits - self.threshold)
        conv_feat = conv_act.mean(dim=[2, 3])  # scalar per sample
        features = torch.stack([conv_feat.squeeze(-1), conv_feat.squeeze(-1)], dim=-1)
        return self.fraud_seq(features)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "HybridConvFraud"]
