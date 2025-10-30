import numpy as np
import torch
from torch import nn
from typing import Iterable, List, Optional, Tuple

# ------------------------------------------------------------------
# Auxiliary building blocks – simplified versions of the seed code
# ------------------------------------------------------------------

class ConvFilter(nn.Module):
    """2‑D convolutional filter with a sigmoid threshold."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        tensor = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()


class FraudLayerParameters:
    """Parameter container mirroring the photonic layer description."""
    def __init__(self,
                 bs_theta: float,
                 bs_phi: float,
                 phases: Tuple[float, float],
                 squeeze_r: Tuple[float, float],
                 squeeze_phi: Tuple[float, float],
                 displacement_r: Tuple[float, float],
                 displacement_phi: Tuple[float, float],
                 kerr: Tuple[float, float]) -> None:
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]],
                          dtype=torch.float32)
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


class FraudDetectionSeq(nn.Sequential):
    """Sequential container that reproduces the fraud‑detection stack."""
    def __init__(self,
                 input_params: FraudLayerParameters,
                 layers: List[FraudLayerParameters]) -> None:
        modules = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
        modules.append(nn.Linear(2, 1))
        super().__init__(*modules)


# ------------------------------------------------------------------
# Main hybrid class
# ------------------------------------------------------------------

class FCL(nn.Module):
    """
    Hybrid fully‑connected layer that stitches together:
      * a linear mapping,
      * a convolutional filter,
      * a fraud‑detection style sequence,
      * a shallow classifier.
    """

    def __init__(self,
                 n_features: int = 1,
                 conv_kernel: int = 2,
                 fraud_params: Optional[List[FraudLayerParameters]] = None,
                 classifier_depth: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.conv = ConvFilter(kernel_size=conv_kernel)
        # Default fraud parameters (zeroed) keep the module functional
        default_input = FraudLayerParameters(0.0, 0.0, (0.0, 0.0),
                                             (0.0, 0.0), (0.0, 0.0),
                                             (0.0, 0.0), (0.0, 0.0),
                                             (0.0, 0.0))
        self.fraud_seq = FraudDetectionSeq(default_input, fraud_params or [])
        # Classifier head mirroring the quantum ansatz
        classifier_layers = []
        in_dim = 1
        for _ in range(classifier_depth):
            classifier_layers.append(nn.Linear(in_dim, 1))
            classifier_layers.append(nn.ReLU())
        classifier_layers.append(nn.Linear(1, 2))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self,
                data: torch.Tensor,
                thetas: Iterable[float]) -> torch.Tensor:
        """
        Forward pass that updates the linear layer from ``thetas`` and
        propagates the data through all sub‑modules.

        Parameters
        ----------
        data : torch.Tensor
            Input tensor of shape (batch, n_features).
        thetas : Iterable[float]
            Flat list of parameters for the linear layer (weights + bias).

        Returns
        -------
        torch.Tensor
            Logits from the classifier head.
        """
        # Update linear weights from thetas
        weight_len = self.linear.weight.numel()
        bias_len = self.linear.bias.numel()
        theta_list = list(thetas)
        if len(theta_list)!= weight_len + bias_len:
            raise ValueError(f"Expected {weight_len + bias_len} parameters, got {len(theta_list)}")
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor(theta_list[:weight_len]).view_as(self.linear.weight))
            self.linear.bias.copy_(torch.tensor(theta_list[weight_len:]).view_as(self.linear.bias))

        x = self.linear(data)
        # Convolution expects a 2‑D kernel – reshape the scalar output
        conv_input = x.view(-1, 1, self.conv.kernel_size, self.conv.kernel_size)
        conv_out = self.conv(conv_input)
        # Fraud detection sequence expects 2‑D input; duplicate the scalar
        fraud_in = torch.cat([conv_out, conv_out], dim=1)
        fraud_out = self.fraud_seq(fraud_in)
        logits = self.classifier(fraud_out)
        return logits

    def run(self,
            data: Iterable[float],
            thetas: Iterable[float]) -> np.ndarray:
        """
        Convenience wrapper that accepts plain Python iterables.
        """
        tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        logits = self.forward(tensor, thetas)
        return logits.detach().numpy()


__all__ = ["FCL"]
