"""Extended classical deep learning classifier with a hybrid quantum‑inspired head.

This module defines a richer convolutional architecture and a learnable
Hybrid head that mimics quantum expectation values.  The network
supports batch‑wise activation shifting and optional residual connections.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridFunction(torch.autograd.Function):
    """Sigmoid with a learnable shift applied to the linear output.

    The shift can be fixed or a learnable parameter.  The backward pass
    uses the analytical derivative of the sigmoid.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        out = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (out,) = ctx.saved_tensors
        return grad_output * out * (1 - out), None


class Hybrid(nn.Module):
    """Dense head that replaces the quantum circuit in the original model.

    The linear layer can be followed by a learnable shift and the
    sigmoid activation defined above.  It supports batched inputs.
    """
    def __init__(self, in_features: int, shift: float | torch.Tensor = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        if isinstance(shift, (int, float)):
            self.shift = nn.Parameter(torch.full((1,), shift))
        else:
            self.shift = nn.Parameter(shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.linear(inputs)
        return HybridFunction.apply(logits, self.shift)


class HybridQCNet(nn.Module):
    """Convolutional net with a hybrid quantum‑inspired head.

    Adds batch‑norm after each conv, residual connections between
    the two conv stages, and optional dropout on the fully‑connected
    layers.  The final output is a two‑class probability vector.
    """
    def __init__(self, use_residual: bool = True, dropout_fc: float = 0.5):
        super().__init__()
        self.use_residual = use_residual
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(15)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=dropout_fc)

        # Flattened size for 32x32 input after the conv layers
        self._flattened = 15 * 6 * 6  # 540
        self.fc1 = nn.Linear(self._flattened, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.hybrid = Hybrid(self.fc3.out_features)

    def _conv_block(self, x: torch.Tensor, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> torch.Tensor:
        out = F.relu(bn(conv(x)))
        out = self.pool(out)
        out = self.drop1(out)
        return out

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self._conv_block(inputs, self.conv1, self.bn1)
        out_res = self._conv_block(out, self.conv2, self.bn2)
        if self.use_residual:
            out_res = out_res + out  # simple residual addition
        out = out_res
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out = self.drop2(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        probs = self.hybrid(out)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["HybridFunction", "Hybrid", "HybridQCNet"]
