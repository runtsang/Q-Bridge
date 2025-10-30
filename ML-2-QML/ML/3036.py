"""Classical hybrid binary classifier mirroring the quantum architecture.

This module implements the same network topology as the quantum
counterpart but replaces the quantum expectation head with a
differentiable sigmoid layer.  It supports a tunable shift for the
activation, enabling direct comparison with the quantum version.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation with optional shift."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        return grad_output * outputs * (1 - outputs), None

class HybridBinaryClassifier(nn.Module):
    """Classical hybrid binary classifier."""
    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )
        # Fully connected head producing four intermediate features
        self.fc = nn.Sequential(
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(84, 4),
            nn.BatchNorm1d(4),
        )
        # Final linear layer to logit
        self.logit = nn.Linear(4, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        logits = self.logit(x).squeeze(-1)
        probs = HybridFunction.apply(logits, self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridBinaryClassifier", "HybridFunction"]
