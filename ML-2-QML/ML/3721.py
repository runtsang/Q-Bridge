import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid head that mimics the quantum expectation."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        outputs, = ctx.saved_tensors
        return grad_output * outputs * (1 - outputs), None


class Hybrid(nn.Module):
    """Linear layer followed by a differentiable sigmoid."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)


class ResidualBlock(nn.Module):
    """Simple residual block used inside the CNN."""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x))) + x


class QCNet(nn.Module):
    """
    Residual‑augmented CNN followed by a hybrid sigmoid head.  The head can
    be switched between a pure classical linear layer and the quantum
    expectation head by toggling ``use_quantum``.
    """
    def __init__(self, use_quantum: bool = True) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.resblock = ResidualBlock(64, 64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        if self.use_quantum:
            self.hybrid = Hybrid(1, shift=0.0)
        else:
            self.linear_head = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.resblock(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.use_quantum:
            out = self.hybrid(x)
        else:
            out = torch.sigmoid(x)
        return torch.cat((out, 1 - out), dim=-1)


__all__ = ["HybridFunction", "Hybrid", "ResidualBlock", "QCNet"]
