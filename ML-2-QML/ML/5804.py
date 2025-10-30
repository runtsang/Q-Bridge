import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalHybridFunction(torch.autograd.Function):
    """Differentiable surrogate for a quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        # Simple sigmoid with a shift to mimic quantum expectation
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class ClassicalHybrid(nn.Module):
    """Classical dense head that emulates a quantum expectation layer."""
    def __init__(self, in_features: int, hidden: int = 32, shift: float = 0.0) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden)
        self.linear2 = nn.Linear(hidden, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(inputs))
        logits = self.linear2(x)
        return ClassicalHybridFunction.apply(logits, self.shift)

class QCNet(nn.Module):
    """CNN backbone with a classical hybrid head for binary classification."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.classical_head = ClassicalHybrid(1, hidden=32, shift=0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.classical_head(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["ClassicalHybridFunction", "ClassicalHybrid", "QCNet"]
