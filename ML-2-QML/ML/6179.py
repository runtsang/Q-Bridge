import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Callable, Optional, Sequence

@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected layer in the classical model."""
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
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class FraudHybridDetector(nn.Module):
    """
    Classical network that optionally delegates feature extraction to a quantum encoder.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layer_params: Iterable[FraudLayerParameters],
        encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.network = build_fraud_detection_program(input_params, layer_params)

    def set_encoder(self, encoder: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Attach a quantum encoder that maps raw inputs to a feature vector."""
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder is not None:
            x = self.encoder(x)
        return self.network(x)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudHybridDetector"]
