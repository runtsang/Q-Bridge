import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Tuple

@dataclass
class HybridFraudParameters:
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

class HybridFraudClassifier(nn.Module):
    """Classical neural network that emulates a photonic fraud‑detection circuit.

    The first layer is unbounded to preserve the raw input transformation; subsequent
    layers are clipped to emulate the limited dynamic range of physical photonic
    components. A final linear head outputs a binary fraud score.
    """

    def __init__(self, input_params: HybridFraudParameters, layers: Iterable[HybridFraudParameters]) -> None:
        super().__init__()
        modules: list[nn.Module] = [self._layer_from_params(input_params, clip=False)]
        modules.extend(self._layer_from_params(l, clip=True) for l in layers)
        modules.append(nn.Linear(2, 1))
        self.model = nn.Sequential(*modules)

    def _layer_from_params(self, params: HybridFraudParameters, clip: bool) -> nn.Module:
        weight = torch.tensor(
            [[params.bs_theta, params.bs_phi],
             [params.squeeze_r[0], params.squeeze_r[1]]],
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
                out = self.activation(self.linear(inputs))
                out = out * self.scale + self.shift
                return out

        return Layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

__all__ = ["HybridFraudParameters", "HybridFraudClassifier"]
