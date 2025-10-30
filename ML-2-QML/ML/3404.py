import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable

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

class ConvFraudHybrid(nn.Module):
    """
    Hybrid model that first applies a 2‑D convolution (classical or quantum) and
    then passes the scalar result through a fraud‑detection‑style feed‑forward
    network.  The classical convolution is a single Conv2d layer; the quantum
    version is handled by ConvFraudHybridQML.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        fraud_input: FraudLayerParameters | None = None,
        fraud_layers: Iterable[FraudLayerParameters] | None = None,
        use_quantum_conv: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.use_quantum_conv = use_quantum_conv

        if use_quantum_conv:
            # The quantum convolution is provided by ConvFraudHybridQML.  We keep a
            # placeholder so that the forward method can detect the mismatch.
            self.conv = None
        else:
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        if fraud_input is None:
            fraud_input = FraudLayerParameters(
                bs_theta=0.0,
                bs_phi=0.0,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(1.0, 1.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
        if fraud_layers is None:
            fraud_layers = []

        self.fraud_network = build_fraud_detection_program(fraud_input, fraud_layers)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        data : torch.Tensor
            Input image of shape (B, 1, H, W) or (H, W).

        Returns
        -------
        torch.Tensor
            Fraud‑risk score of shape (B, 1).
        """
        if self.use_quantum_conv:
            raise RuntimeError(
                "Quantum convolution is not implemented in the classical module. "
                "Use ConvFraudHybridQML for quantum inference."
            )

        x = data
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)

        conv_out = self.conv(x)  # (B,1,k,k)
        conv_out = conv_out.mean(dim=(2, 3))  # (B,1)
        conv_out = conv_out.view(-1, 1)

        fraud_input = conv_out.repeat(1, 2)  # (B,2)
        out = self.fraud_network(fraud_input)
        return out

__all__ = ["ConvFraudHybrid", "FraudLayerParameters", "build_fraud_detection_program"]
