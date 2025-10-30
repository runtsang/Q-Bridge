import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class HybridFunction(Function):
    # Differentiable sigmoid activation mimicking quantum expectation
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float = 0.0) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    # Classical dense head using the HybridFunction
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.linear(inputs)
        return HybridFunction.apply(logits, self.shift)

class QuanvolutionFilter(nn.Module):
    # Classical convolutional filter inspired by the quanvolution example
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class QuanvolutionHybridClassifier(nn.Module):
    # Hybrid classical-quantum classifier with a classical quanvolution filter and a quantum‑like head
    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(4, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.qfilter = QuanvolutionFilter(in_channels=in_channels, out_channels=4)
        self.fc1 = nn.Linear(60, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(self.fc3.out_features, shift=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        prob = self.hybrid(x)
        return torch.cat((prob, 1 - prob), dim=-1)

__all__ = ["QuanvolutionHybridClassifier"]
