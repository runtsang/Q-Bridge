import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid with a shift."""
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

class FourierFeatureLayer(nn.Module):
    """Map inputs to sinusoidal features."""
    def __init__(self, in_features: int, B: int = 10):
        super().__init__()
        self.B = B
        self.register_buffer('weights', torch.randn(in_features, B))

    def forward(self, x: torch.Tensor):
        proj = x @ self.weights
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class Hybrid(nn.Module):
    """Classical surrogate for the quantum expectation head with shift and scale."""
    def __init__(self, in_features: int = 1, out_features: int = 1):
        super().__init__()
        self.feature = FourierFeatureLayer(in_features, B=10)
        # 2*B features after sin/cos
        self.linear = nn.Linear(in_features * 20, out_features)
        self.shift = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor):
        x = self.feature(x)
        logits = self.linear(x)
        probs = torch.sigmoid(logits + self.shift)
        return probs * self.scale

class QCNet(nn.Module):
    """CNN followed by a classical surrogate quantum head."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(in_features=1)

    def forward(self, inputs: torch.Tensor):
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
        probs = self.hybrid(x)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridFunction", "Hybrid", "QCNet"]
