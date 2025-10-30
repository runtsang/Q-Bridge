"""QuantumHybridBinaryClassifier – classical branch of the extended hybrid model.

Features:
* Multi‑branch dense network with residual connections.
* Gated attention that blends classical logits with a quantum‑derived score.
* Configurable loss weighting between binary cross‑entropy and auxiliary classification loss.

The public API matches the original `QCNet` interface so users can swap the class
without any code changes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class _ResidualDenseBlock(nn.Module):
    """Simple residual dense block."""
    def __init__(self, features: int):
        super().__init__()
        self.linear = nn.Linear(features, features)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(x)

class _GatedAttention(nn.Module):
    """Gate that blends classical logits and quantum score."""
    def forward(self, logits: torch.Tensor, quantum_score: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(logits)
        return gate * quantum_score + (1 - gate) * logits

class ClassicalHybridFunction(torch.autograd.Function):
    """Differentiable sigmoid‑approximated quantum expectation."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.save_for_backward(inputs)
        expectation_z = torch.sin(inputs + shift)
        return expectation_z
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        inputs, = ctx.saved_tensors
        grad_inputs = grad_output * torch.cos(inputs + ctx.shift)
        return grad_inputs, None

class ClassicalHybrid(nn.Module):
    """Wrapper that applies the classical hybrid function."""
    def __init__(self, shift: float = 0.0) -> None:
        super().__init__()
        self.shift = shift
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return ClassicalHybridFunction.apply(inputs, self.shift)

class QuantumHybridBinaryClassifier(nn.Module):
    """CNN followed by a multi‑branch dense head with a gated attention hybrid."""
    def __init__(self) -> None:
        super().__init__()
        # Convolutional feature extractor (identical to the original)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Auxiliary dense branch
        self.branch_fc = nn.Sequential(
            nn.Linear(84, 50),
            nn.ReLU(inplace=True),
            _ResidualDenseBlock(50),
            nn.Linear(50, 1)
        )

        # Quantum‑inspired hybrid head
        self.hybrid = ClassicalHybrid(shift=0.0)
        self.attention = _GatedAttention()

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
        logits = self.fc3(x)                       # (batch,1)

        branch_out = self.branch_fc(x)             # (batch,1)
        quantum_score = self.hybrid(branch_out)    # (batch,1)

        fused = self.attention(logits, quantum_score)  # (batch,1)
        probabilities = torch.cat((fused, 1 - fused), dim=-1)
        return probabilities

__all__ = ["QuantumHybridBinaryClassifier"]
