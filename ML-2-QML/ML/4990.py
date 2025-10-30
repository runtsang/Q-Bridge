import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Sequence

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 patch extractor with a small Conv2d layer."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class Kernel(nn.Module):
    """Classical RBF kernel used as a quantum‑inspired similarity evaluator."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

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
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs
    return Layer()

def build_fraud_detection_program(input_params: FraudLayerParameters,
                                 layers: Iterable[FraudLayerParameters]) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class HybridSamplerQNN(nn.Module):
    """
    Classical hybrid sampler/classifier that fuses a quanvolution extractor,
    fraud‑style fully‑connected layers, and a quantum‑inspired RBF kernel.
    """

    def __init__(self,
                 mode: str = "sampler",
                 fraud_params: Sequence[FraudLayerParameters] | None = None,
                 gamma: float = 1.0) -> None:
        """
        Parameters
        ----------
        mode:
            Either ``"sampler"`` (output 2 softmax logits) or ``"classifier"``
            (output 10 class log‑softmax logits).
        fraud_params:
            Iterable of :class:`FraudLayerParameters`. The first element
            defines the input layer (``clip=False``), the rest are hidden
            layers (``clip=True``).  If omitted, a minimal default is used.
        gamma:
            RBF kernel width.
        """
        super().__init__()
        self.mode = mode
        self.feature_extractor = QuanvolutionFilter()
        if fraud_params is None:
            dummy = FraudLayerParameters(0.5, 0.3, (0.1, 0.2),
                                         (0.0, 0.0), (0.0, 0.0),
                                         (1.0, 1.0), (0.0, 0.0),
                                         (0.0, 0.0))
            fraud_params = (dummy,)
        self.fraud_model = build_fraud_detection_program(fraud_params[0], fraud_params[1:])
        self.kernel = Kernel(gamma)
        out_features = 4 * 14 * 14
        if mode == "sampler":
            self.output_layer = nn.Linear(out_features, 2)
        else:
            self.output_layer = nn.Linear(out_features, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        The data flows through the quanvolution filter, then the fraud‑style
        sequential network, and finally an output head.  When ``mode`` is
        ``"sampler"``, a softmax over 2 logits is returned; otherwise a
        log‑softmax over 10 logits is produced.
        """
        feats = self.feature_extractor(x)
        fraud_out = self.fraud_model(feats)
        logits = self.output_layer(fraud_out)
        if self.mode == "sampler":
            return F.softmax(logits, dim=-1)
        return F.log_softmax(logits, dim=-1)
