import torch
import torch.nn as nn

class QuanvolutionFilter(nn.Module):
    """
    A 2×2 stride‑2 convolution that emulates the behaviour of a
    quanvolutional filter.  The output is flattened so it can be fed
    into a linear head.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class HybridFullyConnectedLayer(nn.Module):
    """
    Classical front‑end that prepares a feature vector for the quantum
    expectation head.  It mirrors the pattern of the fully‑connected
    quantum layer from reference FCL.py but replaces the circuit with
    a linear projection of quanvolutional features.
    """
    def __init__(self, in_features: int = 4 * 14 * 14, out_features: int = 1):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a tensor of shape (batch, out_features) that will be used
        as the rotation angles for the quantum circuit.
        """
        features = self.qfilter(x)
        angles = self.linear(features)
        return angles
