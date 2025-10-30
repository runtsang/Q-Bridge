import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalHybridFunction(torch.autograd.Function):
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

class ClassicalHybrid(nn.Module):
    """Dense head that mimics quantum expectation using sigmoid."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return ClassicalHybridFunction.apply(self.linear(logits), self.shift)

class QuanvolutionFilter(nn.Module):
    """Classical convolutional filter inspired by quanvolution."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class HybridQuanvolutionClassifier(nn.Module):
    """Hybrid classifier combining quanvolution filter with classical hybrid head."""
    def __init__(self, in_channels: int = 1, num_classes: int = 2, shift: float = 0.0):
        super().__init__()
        self.qfilter = QuanvolutionFilter(in_channels=in_channels)
        self.hybrid = ClassicalHybrid(in_features=4 * 14 * 14, shift=shift)
        self.num_classes = num_classes
        if num_classes > 2:
            self.classifier = nn.Linear(1, num_classes)
        else:
            self.classifier = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        hybrid_out = self.hybrid(features)
        if self.num_classes > 2:
            logits = self.classifier(hybrid_out)
            return F.log_softmax(logits, dim=-1)
        else:
            probs = torch.cat((hybrid_out, 1 - hybrid_out), dim=-1)
            return probs

__all__ = ["ClassicalHybridFunction", "ClassicalHybrid",
           "QuanvolutionFilter", "HybridQuanvolutionClassifier"]
