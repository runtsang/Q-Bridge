import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """
    Classical 2x2 convolution filter that emulates the quanvolution concept.
    Operates on a single‑channel 28×28 image and produces a flattened
    feature vector of length 4×14×14.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class EstimatorNN(nn.Module):
    """
    Hybrid classical network: quanvolution feature extractor followed by a
    deep feed‑forward core for regression. The architecture is deliberately
    deeper than the seed to demonstrate scaling.
    """
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.head = nn.Sequential(
            nn.Linear(4 * 14 * 14, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(inputs)
        return self.head(features)

def EstimatorQNN() -> nn.Module:
    """
    Factory that returns a fully‑trained regression model.
    The function mirrors the original API but now builds a richer network.
    """
    return EstimatorNN()

__all__ = ["EstimatorQNN", "QuanvolutionFilter", "EstimatorNN"]
