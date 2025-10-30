import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Tuple, Sequence

class ConvFilter(nn.Module):
    """2‑D convolutional filter that emulates the quantum quanvolution."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect a batch of 1‑channel images of shape (B, H, W)
        x = x.view(-1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        # Return a scalar per sample
        return activations.mean(dim=(1, 2, 3))

@dataclass
class FraudParams:
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

def _layer_from_params(params: FraudParams, *, clip: bool) -> nn.Module:
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
    input_params: FraudParams,
    layers: Iterable[FraudParams],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class ConvFraudHybrid(nn.Module):
    """
    Hybrid classical model that first applies a 2‑D convolutional filter and then
    passes the result through a fraud‑detection style fully‑connected network.
    """
    def __init__(
        self,
        conv_kernel: int = 2,
        conv_threshold: float = 0.0,
        input_params: FraudParams | None = None,
        hidden_layers: Iterable[FraudParams] | None = None,
    ):
        super().__init__()
        self.conv = ConvFilter(kernel_size=conv_kernel, threshold=conv_threshold)

        if input_params is None:
            # Provide a sane default
            input_params = FraudParams(
                bs_theta=0.5,
                bs_phi=0.5,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(0.0, 0.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
        if hidden_layers is None:
            hidden_layers = []

        self.fraud_net = build_fraud_detection_program(input_params, hidden_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ConvFilter returns a scalar per sample
        conv_out = self.conv(x)  # shape (B,)
        # Duplicate the scalar to a 2‑dimensional feature for the fraud net
        features = conv_out.unsqueeze(1).repeat(1, 2)
        return self.fraud_net(features)

__all__ = ["ConvFraudHybrid", "ConvFilter", "FraudParams", "build_fraud_detection_program"]
