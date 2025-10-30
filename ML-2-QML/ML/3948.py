import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionHybridFilter(nn.Module):
    """Classical convolutional filter that downsamples MNIST images to a 4‑channel feature map."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class QuanvolutionHybridClassifier(nn.Module):
    """Two‑stage classical model: a convolutional filter followed by a fully‑connected head."""
    def __init__(self) -> None:
        super().__init__()
        self.filter = QuanvolutionHybridFilter()
        self.fc = nn.Sequential(
            nn.Linear(4 * 14 * 14, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.filter(x)          # (bsz, 4, 14, 14)
        flattened = features.view(x.size(0), -1)
        logits = self.fc(flattened)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybridFilter", "QuanvolutionHybridClassifier"]
