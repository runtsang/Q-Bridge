import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation with optional shift."""
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
    """Dense head that applies a linear layer followed by a sigmoid."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

class QuanvolutionFilter(nn.Module):
    """Classical patch‑wise filter with a 2×2 conv kernel."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class QuanvolutionHybridClassifier(nn.Module):
    """Classical classifier with optional hybrid dense head."""
    def __init__(self, in_channels: int = 1, num_classes: int = 10, shift: float = 0.0):
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels)
        self.linear = nn.Linear(4 * 14 * 14, 128)
        self.hybrid = Hybrid(128, shift=shift)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        probs = self.hybrid(logits)
        logits = self.classifier(probs)
        return F.log_softmax(logits, dim=-1)
