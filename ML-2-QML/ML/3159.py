from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable, Sequence

class HybridFCL_QCNN(nn.Module):
    """
    Classical hybrid network that merges a fully‑connected layer with a
    convolution‑style block inspired by the QCNN seed.
    """

    def __init__(self, n_features: int = 1, conv_in: int = 8) -> None:
        super().__init__()
        # Linear mapping (FCL part)
        self.linear = nn.Linear(n_features, conv_in, bias=False)

        # Convolution‑style block (QCNN part)
        self.feature_map = nn.Sequential(nn.Linear(conv_in, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.linear(inputs)
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Mimics the original FCL API: ``run`` accepts a list of parameters
        that are fed into the linear map.  The output is the scalar produced
        by the final sigmoid layer.
        """
        theta_tensor = torch.as_tensor(list(thetas), dtype=torch.float32)
        if theta_tensor.ndim == 0:
            theta_tensor = theta_tensor.unsqueeze(0)
        theta_tensor = theta_tensor.reshape(1, -1)
        with torch.no_grad():
            out = self.forward(theta_tensor)
        return out.squeeze().cpu().numpy()

__all__ = ["HybridFCL_QCNN"]
