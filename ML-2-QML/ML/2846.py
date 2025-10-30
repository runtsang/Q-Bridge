import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalFilter(nn.Module):
    """Fast 2x2 convolution that reduces spatial resolution."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

class QuanvolutionHybridClassifier(nn.Module):
    """Hybrid classifier that combines a classical conv, a quantum kernel, and a linear head.
    Supports both classification and regression."""
    def __init__(self, quantum_filter: nn.Module, num_classes: int = 10, regression: bool = False):
        super().__init__()
        self.classical = ClassicalFilter()
        self.quantum = quantum_filter
        self.regression = regression
        # Quantum filter must expose an output_dim or we assume the same size as input
        out_dim = getattr(quantum_filter, "output_dim", None)
        if out_dim is None:
            out_dim = 4 * 14 * 14  # default for 28×28 MNIST patches
        self.head = nn.Linear(out_dim, 1 if regression else num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.classical(x)
        qfeatures = self.quantum(features)
        logits = self.head(qfeatures)
        if self.regression:
            return logits.squeeze(-1)
        return F.log_softmax(logits, dim=-1)

def create_hybrid_classifier(quantum_filter: nn.Module, num_classes: int = 10, regression: bool = False) -> nn.Module:
    """Convenience factory that returns a fully‑connected hybrid model."""
    return QuanvolutionHybridClassifier(quantum_filter, num_classes, regression)

__all__ = ["ClassicalFilter", "QuanvolutionHybridClassifier", "create_hybrid_classifier"]
