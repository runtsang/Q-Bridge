import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation with optional shift, mimicking a quantum expectation head."""
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
    """Linear head with an optional shift, used as a classical surrogate for quantum expectation."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)


class QuanvolutionFilterClassical(nn.Module):
    """Classical 2×2 convolution mimicking a patch‑wise quantum kernel."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)


class QuantumNATGen223(nn.Module):
    """Hybrid classical architecture that combines convolutional feature extraction,
    a patch‑wise quanvolution filter, and a lightweight MLP head with a sigmoid surrogate for quantum expectation."""
    def __init__(self):
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Classical quanvolution
        self.qfilter = QuanvolutionFilterClassical()
        # MLP head
        self.fc = nn.Sequential(
            nn.Linear(4 * 3 * 3, 128),  # 4 channels, 3x3 patches after pooling
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        # Hybrid sigmoid surrogate
        self.hybrid = Hybrid(4, shift=0.0)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.qfilter(x)
        x = self.fc(x)
        x = self.hybrid(x)
        return self.norm(x)


__all__ = ["HybridFunction", "Hybrid", "QuanvolutionFilterClassical", "QuantumNATGen223"]
