import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFunction(torch.autograd.Function):
    """Sigmoid with optional shift, emulating a quantum expectation."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        return grad_output * outputs * (1 - outputs), None

class Hybrid(nn.Module):
    """Linear head followed by HybridFunction."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 stride‑2 convolution that mimics a quantum patch encoder."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class QuantumNATEnhanced(nn.Module):
    """Hybrid classical network combining CNN, quanvolution, and sigmoid head."""
    def __init__(self) -> None:
        super().__init__()

        # Classical backbone
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(p=0.3)

        # Quanvolution component
        self.quanv = QuanvolutionFilter()

        # Fully‑connected head
        self.fc1 = nn.Linear(16 * 7 * 7 + 4 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)

        # Hybrid sigmoid head
        self.hybrid = Hybrid(4, shift=0.0)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_orig = x.clone()  # retain original for quanvolution

        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        cnn_feat = torch.flatten(x, 1)

        # Quanvolution feature extraction
        quanv_feat = self.quanv(x_orig)

        # Concatenate features
        feat = torch.cat([cnn_feat, quanv_feat], dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(feat))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Hybrid sigmoid head
        probs = self.hybrid(x)
        return self.norm(probs)

__all__ = ["HybridFunction", "Hybrid", "QuanvolutionFilter", "QuantumNATEnhanced"]
