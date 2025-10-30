import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class FeatureScaler(nn.Module):
    """Per‑feature affine scaling that can be learned or fixed."""
    def __init__(self, in_features: int, learnable: bool = True):
        super().__init__()
        if learnable:
            self.weight = nn.Parameter(torch.ones(in_features))
            self.bias = nn.Parameter(torch.zeros(in_features))
        else:
            self.register_buffer("weight", torch.ones(in_features))
            self.register_buffer("bias", torch.zeros(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight + self.bias

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid that emulates a quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    """Classical replacement for the quantum head."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.linear(inputs)
        return HybridFunction.apply(logits, self.shift)

class HybridQuantumClassifier(nn.Module):
    """CNN backbone + configurable dropout schedule + MLP head."""
    def __init__(self,
                 dropout_rates: List[float] = None,
                 n_hidden: int = 120,
                 shift: float = 0.0,
                 learn_scaling: bool = True):
        super().__init__()
        # Convolutional backbone (identical to seed)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Feature scaling before MLP
        self.scaler = FeatureScaler(55815, learnable=learn_scaling)

        # MLP head
        self.fc1 = nn.Linear(55815, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 84)
        self.fc3 = nn.Linear(84, 1)

        # Drop‑out schedule
        if dropout_rates is None:
            dropout_rates = [0.0, 0.0, 0.0]
        self.dropout1 = nn.Dropout(dropout_rates[0])
        self.dropout2 = nn.Dropout(dropout_rates[1])
        self.dropout3 = nn.Dropout(dropout_rates[2])

        # Hybrid head
        self.hybrid = Hybrid(1, shift=shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = self.scaler(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.hybrid(x)
        return torch.cat((x, 1 - x), dim=-1)

__all__ = ["HybridQuantumClassifier", "FeatureScaler", "HybridFunction", "Hybrid"]
