from dataclasses import dataclass
from typing import Iterable
import torch
from torch import nn

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

class FraudDetectionHybrid(nn.Module):
    """
    Classical fraud‑detection network.
    Re‑implements the photonic‑style layers from the seed but with a
    fully‑connected PyTorch implementation.  The final linear layer
    produces a sigmoid probability for binary classification.
    """
    def __init__(self, input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        seq = [self._build_layer(input_params, clip=False)]
        seq.extend(self._build_layer(lay, clip=True) for lay in layers)
        seq.append(nn.Linear(2, 1))
        self.feature_extractor = nn.Sequential(*seq)

    def _build_layer(self, params: FraudLayerParameters, clip: bool) -> nn.Module:
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
            def __init__(self):
                super().__init__()
                self.linear = linear
                self.activation = activation
                self.register_buffer("scale", scale)
                self.register_buffer("shift", shift)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.activation(self.linear(x))
                out = out * self.scale + self.shift
                return out
        return Layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Input tensor of shape (batch, 2) or (2,).  The model
            expects two features per sample.
        Returns
        -------
        torch.Tensor
            Sigmoid probability tensor of shape (batch, 1).
        """
        out = self.feature_extractor(x)
        return torch.sigmoid(out)

__all__ = ["FraudDetectionHybrid", "FraudLayerParameters"]
