import torch
from torch import nn
import numpy as np

class HybridFullyConnectedLayer(nn.Module):
    """
    Classical counterpart to the hybrid fully‑connected layer.

    Features:
      * Linear mapping + Tanh activation (original FCL).
      * Optional scaling (scale * x + shift) inspired by fraud‑detection.
      * Optional softmax output mimicking the SamplerQNN behaviour.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int = 1,
                 scale: float = 1.0,
                 shift: float = 0.0,
                 sampler: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.tanh = nn.Tanh()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor(shift, dtype=torch.float32))
        self.sampler = sampler
        if sampler:
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.tanh(out)
        out = out * self.scale + self.shift
        if self.sampler:
            out = self.softmax(out)
        return out

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Dummy stub for quantum expectation.  In a true hybrid setting,
        this would invoke the quantum module.
        """
        return np.asarray([float(t) for t in thetas], dtype=np.float32)

__all__ = ["HybridFullyConnectedLayer"]
