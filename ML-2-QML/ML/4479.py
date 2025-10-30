import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
from typing import Iterable, List, Optional

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

class HybridFCL(nn.Module):
    """
    Classical hybrid layer that combines:
    • a convolutional pre‑processor (from Conv.py),
    • a fully‑connected block (from FCL.py),
    • optional fraud‑detection style clipping layers (from FraudDetection.py).
    The API mirrors the seed examples with a ``run`` method that accepts a
    parameter vector and returns a NumPy array.
    """
    def __init__(self,
                 n_features: int = 1,
                 conv_kernel: int = 2,
                 fraud_params: Optional[List[FraudLayerParameters]] = None,
                 clip: bool = True):
        super().__init__()
        modules: List[nn.Module] = []

        # Convolutional filter
        if conv_kernel > 1:
            modules.append(nn.Conv2d(1, 1, kernel_size=conv_kernel, bias=True))
            modules.append(nn.Sigmoid())

        # Fully‑connected block
        modules.append(nn.Linear(n_features, 1))
        modules.append(nn.Tanh())

        # Fraud‑detection style layers
        if fraud_params:
            for params in fraud_params:
                weight = torch.tensor([[params.bs_theta, params.bs_phi],
                                      [params.squeeze_r[0], params.squeeze_r[1]]],
                                      dtype=torch.float32)
                bias = torch.tensor(params.phases, dtype=torch.float32)
                if clip:
                    weight = weight.clamp(-5.0, 5.0)
                    bias = bias.clamp(-5.0, 5.0)
                linear = nn.Linear(2, 2, bias=True)
                with torch.no_grad():
                    linear.weight.copy_(weight)
                    linear.bias.copy_(bias)
                activation = nn.Tanh()
                scale = torch.tensor(params.displacement_r, dtype=torch.float32)
                shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

                class FraudLayer(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.linear = linear
                        self.activation = activation
                        self.register_buffer("scale", scale)
                        self.register_buffer("shift", shift)

                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        out = self.activation(self.linear(x))
                        return out * self.scale + self.shift

                modules.append(FraudLayer())

        modules.append(nn.Linear(2, 1))
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def run(self, thetas: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Accept a vector of parameters and return the network output as a
        NumPy array, matching the API of the original FCL.py.
        """
        if isinstance(thetas, np.ndarray):
            thetas = torch.as_tensor(thetas, dtype=torch.float32)
        out = self.forward(thetas)
        return out.detach().cpu().numpy()

__all__ = ["HybridFCL"]
