import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 filter that emulates the quantum patch encoder."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

class ClassicalApproxFunction(torch.autograd.Function):
    """Analytic surrogate for a quantum expectation value."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        ctx.inputs = inputs
        ctx.shift = shift
        return torch.cos(inputs + shift)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -grad_output * torch.sin(ctx.inputs + ctx.shift), None

class HybridHead(nn.Module):
    """Classical head that replaces the quantum circuit."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return ClassicalApproxFunction.apply(logits, self.shift)

class HybridNet(nn.Module):
    """Full hybrid network with quanvolution, classical CNN, and classical head."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.conv1 = nn.Conv2d(4, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(60, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.head = HybridHead(self.fc3.out_features, shift=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assume input shape (N, 1, 28, 28)
        features = self.qfilter(x)
        features = features.view(x.size(0), 4, 14, 14)
        x = F.relu(self.conv1(features))
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
        probs = self.head(x)
        return torch.cat((probs, 1 - probs), dim=-1)
