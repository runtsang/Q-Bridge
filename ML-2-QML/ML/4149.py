import torch
from torch import nn
import torch.nn.functional as F
from typing import Iterable

def _clip_tensor(tensor: torch.Tensor, bound: float) -> torch.Tensor:
    """Clamp tensor values to the range [-bound, bound]."""
    return torch.clamp(tensor, -bound, bound)

class HybridFCL(nn.Module):
    """
    Classical hybrid block that combines:
    * A 2‑D convolution acting like a quanvolution filter.
    * A fraud‑detection style fully‑connected layer with clipping and scaling.
    * A quantum expectation factor supplied by an external quantum circuit.
    """
    def __init__(self, quantum_circuit, in_channels: int = 1, out_features: int = 10, clip_max: float = 5.0):
        super().__init__()
        self.quantum_circuit = quantum_circuit
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=2, stride=2)
        self.fc = nn.Linear(4 * 14 * 14, out_features)
        self.activation = nn.Tanh()
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))
        self.clip_max = clip_max

    def forward(self, x: torch.Tensor, thetas: Iterable[float]) -> torch.Tensor:
        # Convolutional feature extraction
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)

        # Fraud‑style linear layer with clipping
        weight = _clip_tensor(self.fc.weight, self.clip_max)
        bias = _clip_tensor(self.fc.bias, self.clip_max)
        out = F.linear(feat, weight, bias)
        out = self.activation(out)

        # Quantum expectation factor
        q_exp = torch.tensor(self.quantum_circuit.run(thetas), dtype=torch.float32)
        out = out * self.scale * q_exp + self.shift
        return out
