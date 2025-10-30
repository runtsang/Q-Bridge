import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Classical 2‑D convolutional filter that emulates the patch‑wise quantum kernel used in the quantum version."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class QuanvolutionHybrid(nn.Module):
    """Hybrid classical quanvolution network.

    Parameters
    ----------
    n_classes : int
        Number of output classes.
    hidden_dim : int
        Size of the hidden fully‑connected layer.
    """
    def __init__(self,
                 n_classes: int = 10,
                 hidden_dim: int = 128) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.fc = nn.Linear(4 * 14 * 14, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        hidden = self.fc(features)
        logits = self.classifier(hidden)
        return F.log_softmax(logits, dim=-1)

# Backwards compatibility with the original module name
QuanvolutionClassifier = QuanvolutionHybrid

__all__ = ["QuanvolutionFilter", "QuanvolutionHybrid", "QuanvolutionClassifier"]
