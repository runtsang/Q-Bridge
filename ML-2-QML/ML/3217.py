"""Hybrid classical model combining convolutional feature extraction with classification and regression heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Standard 2x2 convolutional filter with stride 2."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class QuanvolutionHybridModel(nn.Module):
    """Classical hybrid model with classification and regression heads."""
    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.classifier = nn.Linear(4 * 14 * 14, n_classes)
        self.regressor = nn.Linear(4 * 14 * 14, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.qfilter(x)
        logits = self.classifier(features)
        regression = self.regressor(features)
        return F.log_softmax(logits, dim=-1), regression

__all__ = ["QuanvolutionFilter", "QuanvolutionHybridModel"]
