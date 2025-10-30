"""Classical PyTorch implementation of a hybrid quantum-classical binary classifier.

The model mirrors the structure of the original hybrid network but replaces the
quantum head with a lightweight differentiable sigmoid layer that can be swapped
with a quantum head during experimentation.  The architecture is extended with
batch‑normalisation, residual connections and temperature‑scaled logits
to improve calibration and convergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFunction(torch.autograd.Function):
    """Temperature‑scaled sigmoid activation used as a stand‑in for a quantum
    expectation value.  The temperature parameter allows the function to
    mimic the smoothness of a QPU measurement.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, temperature: float) -> torch.Tensor:
        ctx.save_for_backward(inputs)
        ctx.temperature = temperature
        return torch.sigmoid(inputs / temperature)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        inputs, = ctx.saved_tensors
        temp = ctx.temperature
        sigmoid = torch.sigmoid(inputs / temp)
        grad = grad_output * sigmoid * (1 - sigmoid) / temp
        return grad, None

class HybridLayer(nn.Module):
    """Simple linear head followed by the temperature‑scaled sigmoid."""
    def __init__(self, in_features: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return HybridFunction.apply(logits, self.temperature)

class HybridQCNet(nn.Module):
    """CNN backbone with residual blocks and a hybrid classification head."""
    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        # Residual block
        self.res_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.res_bn   = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=0.3)

        # Fully‑connected head
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.hybrid = HybridLayer(128, temperature=temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Residual connection
        res = self.res_bn(self.res_conv(x))
        x = F.relu(x + res)
        x = self.pool(x)
        x = self.dropout(x)

        # Flatten
        x = torch.flatten(x, 1)

        # FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Hybrid head
        probs = self.hybrid(x)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridFunction", "HybridLayer", "HybridQCNet"]
