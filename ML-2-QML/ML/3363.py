"""Classical implementation of the hybrid QCNet with a fully‑connected quantum layer simulated classically.

The module contains:
* `FCL` – a stand‑in for the quantum fully‑connected layer from the reference pair.
* `HybridFunction` – a differentiable sigmoid head.
* `Hybrid` – a hybrid head that can switch between a linear layer and the simulated quantum layer.
* `HybridQCNet` – a CNN backbone followed by the hybrid head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable
import numpy as np


class FCL:
    """Simulated fully connected quantum layer used as a classical stand‑in."""
    def __init__(self, n_features: int = 1):
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        outputs, = ctx.saved_tensors
        return grad_output * outputs * (1 - outputs), None


class Hybrid(nn.Module):
    """Hybrid head that can use either a linear layer or the simulated quantum FCL."""
    def __init__(self, n_features: int, use_fcl: bool = True, shift: float = 0.0):
        super().__init__()
        self.use_fcl = use_fcl
        self.shift = shift
        if use_fcl:
            self.fcl = FCL(n_features)
        else:
            self.linear = nn.Linear(n_features, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.use_fcl:
            # Treat each element of the input as a parameter for the simulated quantum layer.
            params = inputs.detach().cpu().numpy().flatten()
            expectation = self.fcl.run(params)
            # Broadcast back to batch dimension.
            return torch.tensor(expectation, device=inputs.device).unsqueeze(0)
        else:
            logits = self.linear(inputs)
            return HybridFunction.apply(logits, self.shift)


class HybridQCNet(nn.Module):
    """Classic CNN followed by a hybrid head."""
    def __init__(self, use_fcl: bool = True, shift: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(1, use_fcl=use_fcl, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x).view(-1, 1)
        logits = self.hybrid(x)
        probs = torch.cat((logits, 1 - logits), dim=-1)
        return probs
