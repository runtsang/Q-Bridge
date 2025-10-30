import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ShiftedSigmoidFunction(torch.autograd.Function):
    """
    Differentiable sigmoid with a learnable shift to emulate a quantum
    expectation value.  The shift is a trainable scalar.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: torch.Tensor):
        ctx.save_for_backward(inputs, shift)
        return torch.sigmoid(inputs + shift)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, shift = ctx.saved_tensors
        sigmoid = torch.sigmoid(inputs + shift)
        grad_inputs = grad_output * sigmoid * (1 - sigmoid)
        grad_shift = torch.zeros_like(shift)
        return grad_inputs, grad_shift

class HybridQuantumBinaryClassifier(nn.Module):
    """
    Classical CNN backbone with a hybrid head that can operate in two modes:
    1. Pure classical sigmoid (default).
    2. Lightweight surrogate using ShiftedSigmoidFunction.
    The class exposes a learnable shift parameter that can be tuned during
    training, providing a bridge to the quantum head.
    """
    def __init__(self, use_surrogate: bool = True, shift_init: float = 0.0):
        super().__init__()
        self.use_surrogate = use_surrogate
        self.shift = nn.Parameter(torch.tensor(shift_init, dtype=torch.float32))
        # Backbone
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.3)
        # Fully connected
        self.fc1 = nn.Linear(64 * 8 * 8, 256)  # assumes 32x32 input
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x).squeeze(-1)
        if self.use_surrogate:
            probs = ShiftedSigmoidFunction.apply(logits, self.shift)
        else:
            probs = torch.sigmoid(logits + self.shift)
        return torch.stack([probs, 1 - probs], dim=-1)

__all__ = ["HybridQuantumBinaryClassifier"]
