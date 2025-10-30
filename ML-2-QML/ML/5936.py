import torch
from torch import nn
import torch.nn.functional as F

class ConvGen340(nn.Module):
    """Hybrid classical convolutional backbone with optional quantum filter.

    This class merges the classical convolutional backbone from QuantumNAT
    with a 2‑D convolutional filter inspired by Conv.py.  It can be used
    as a drop‑in replacement for the original Conv.py while optionally
    delegating the filter operation to a quantum backend when available.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 stride: int = 1,
                 threshold: float = 0.0,
                 quantum: bool = False,
                 device: str = "cpu"):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.threshold = threshold
        self.quantum = quantum

        # Classical feature extractor (from QuantumNAT)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 2‑D convolutional filter (from Conv.py)
        self.filter = nn.Conv2d(1, 1, kernel_size=self.kernel_size,
                                bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        feat = self.features(x)
        # Reduce channels to a single map
        feat_mean = feat.mean(dim=1, keepdim=True)
        # Apply the 2‑D filter
        logits = self.filter(feat_mean)
        activations = torch.sigmoid(logits - self.threshold)
        # Return a scalar per example
        return activations.mean(dim=[1, 2, 3])

    def run(self, data: torch.Tensor) -> float:
        """Convenience wrapper that accepts a 2‑D tensor and returns a scalar."""
        with torch.no_grad():
            return self.forward(data).item()

__all__ = ["ConvGen340"]
