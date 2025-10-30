import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Simple residual block for an MLP."""
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.linear(x)) + x)

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation with a configurable shift."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        ctx.shift = shift
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        outputs, = ctx.saved_tensors
        return grad_output * outputs * (1 - outputs), None

class HybridClassifier(nn.Module):
    """
    Classical binary classifier that mirrors the quantum head with a residual MLP.
    The network consists of a linear layer, two residual blocks, a final linear head,
    and a sigmoid activation with an optional shift.
    """
    def __init__(self, in_features: int, hidden_dim: int = 128, shift: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.res1(x)
        x = self.res2(x)
        logits = self.fc2(x)
        probs = HybridFunction.apply(logits, self.shift)
        return torch.cat([probs, 1 - probs], dim=-1)
