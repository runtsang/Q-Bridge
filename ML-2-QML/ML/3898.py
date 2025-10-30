import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalHybridFunction(torch.autograd.Function):
    """Classical surrogate for a quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, bias: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + bias)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class ClassicalHybrid(nn.Module):
    """Dense head that replaces a quantum circuit."""
    def __init__(self, in_features: int, bias: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.bias = bias

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return ClassicalHybridFunction.apply(self.linear(logits), self.bias)

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 convolution that mimics a quantum patch."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class HybridQuanvolutionQCNet(nn.Module):
    """Classical hybrid network combining conv layers, a quanvolution filter,
    fully‑connected layers, and a classical surrogate quantum head."""
    def __init__(self):
        super().__init__()
        # Quanvolution filter
        self.quanv = QuanvolutionFilter()

        # Convolutional backbone
        self.conv1 = nn.Conv2d(4, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully connected head
        self.fc1 = nn.Linear(15 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Hybrid head
        self.hybrid = ClassicalHybrid(self.fc3.out_features, bias=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quanv(x)                      # shape: (B, 4, H/2, W/2)
        x = F.relu(self.conv1(x))
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
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["ClassicalHybridFunction", "ClassicalHybrid",
           "QuanvolutionFilter", "HybridQuanvolutionQCNet"]
