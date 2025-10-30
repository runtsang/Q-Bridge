import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

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

class HybridFilter(nn.Module):
    """Classical convolutional filter with fraud‑detection inspired scaling."""

    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 scaling: FraudLayerParameters = None,
                 clip: bool = True) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        if scaling is None:
            self.scale = torch.tensor(1.0, dtype=torch.float32)
            self.shift = torch.tensor(0.0, dtype=torch.float32)
        else:
            # use first component of the photonic displacement params
            self.scale = torch.tensor(scaling.displacement_r[0], dtype=torch.float32)
            self.shift = torch.tensor(scaling.displacement_phi[0], dtype=torch.float32)
            if clip:
                self.scale = self.scale.clamp(-5.0, 5.0)
                self.shift = self.shift.clamp(-5.0, 5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a scalar per batch item after convolution, thresholding, and scaling."""
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        activations = activations * self.scale + self.shift
        return activations.mean(dim=(2, 3))

def Conv() -> HybridFilter:
    """Factory mirroring the original Conv interface."""
    return HybridFilter()

__all__ = ["HybridFilter", "Conv", "FraudLayerParameters"]
