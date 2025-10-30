import torch
import torch.nn as nn
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
    attention_heads: int = 1

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

class FraudDetectionModel(nn.Module):
    """
    Classical fraud detection model with a multi‑head attention block.
    """
    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        self.input_layer = _layer_from_params(input_params, clip=False)
        self.attention = nn.MultiheadAttention(embed_dim=2, num_heads=input_params.attention_heads, batch_first=True)
        self.hidden_layers = nn.ModuleList(_layer_from_params(p, clip=True) for p in layers)
        self.output_layer = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        attn_out, _ = self.attention(out.unsqueeze(1), out.unsqueeze(1), out.unsqueeze(1))
        out = attn_out.squeeze(1)
        for layer in self.hidden_layers:
            out = layer(out)
        out = self.output_layer(out)
        return out

__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
