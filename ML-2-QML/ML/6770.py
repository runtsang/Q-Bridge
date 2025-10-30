import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Classical layer parameters – identical to the original ML seed
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
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

def build_fraud_detection_model(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# Classical convolutional filter – drop‑in replacement for quanvolution
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """2‑D convolutional filter that emulates a quantum quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

# --------------------------------------------------------------------------- #
# High‑level hybrid model
# --------------------------------------------------------------------------- #
class FraudDetectionHybrid(nn.Module):
    """
    A hybrid fraud‑detection model that first applies a classical convolution
    (simulating a quanvolution layer) and then runs the photonic fully‑connected
    network with the convolution result as an additional feature.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Sequence[FraudLayerParameters],
        kernel_size: int = 2,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size, threshold)
        self.model = build_fraud_detection_model(input_params, layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            2‑D input array of shape (kernel_size, kernel_size)

        Returns
        -------
        torch.Tensor
            Model prediction (scalar per batch element)
        """
        # Classical convolution output
        conv_out = self.conv.run(x.cpu().numpy())
        # Simple secondary feature: mean of the raw input
        mean_val = x.mean().item()
        # Build a 2‑D feature vector for the photonic network
        features = torch.tensor([conv_out, mean_val], dtype=torch.float32).unsqueeze(0)
        return self.model(features)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_model",
    "ConvFilter",
    "FraudDetectionHybrid",
]
