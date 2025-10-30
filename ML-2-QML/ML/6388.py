import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSigmoidFunction(torch.autograd.Function):
    """Differentiable sigmoid activation that mimics the quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        out = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (out,) = ctx.saved_tensors
        return grad_output * out * (1 - out), None

class HybridHead(nn.Module):
    """Classical head that replaces the quantum circuit."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        probs = HybridSigmoidFunction.apply(logits, self.shift)
        return probs

class HybridQuanvolutionNet(nn.Module):
    """Classical network that mirrors the quanvolution architecture and ends with a hybrid sigmoid head."""
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        # Classical 2×2 convolution mimicking the quanvolution filter
        self.conv = nn.Conv2d(3, 4, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(4)
        self.flatten = nn.Flatten()
        # Feature extractor
        self.fc1 = nn.Linear(4 * 14 * 14, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.head = HybridHead(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv(x))
        x = self.bn(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.head(x)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridQuanvolutionNet"]
