"""HybridBinaryClassifier – classical CNN with residual and Bayesian head.

The module extends the original QCNet by adding a residual block to the
convolutional backbone and a Bayesian output layer that samples from a
Bernoulli distribution.  The architecture is fully PyTorch‑compatible
and can be trained with standard optimizers.

Classes
-------
ResidualBlock : nn.Module
    A simple 3×3 convolutional residual block.
BayesianOutput : nn.Module
    Produces a Bernoulli distribution over the predicted class.
HybridBinaryClassifier : nn.Module
    Full classifier that plugs the residual block and Bayesian head
    into the original feature extractor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli


class ResidualBlock(nn.Module):
    """3×3 convolutional residual block with optional down‑sampling."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels!= out_channels or stride!= 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return self.relu(out + self.shortcut(x))


class BayesianOutput(nn.Module):
    """Bayesian head that returns a Bernoulli distribution."""
    def __init__(self, in_features: int):
        super().__init__()
        self.logit = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> Bernoulli:
        logits = self.logit(x)
        probs = torch.sigmoid(logits)
        return Bernoulli(probs)


class HybridBinaryClassifier(nn.Module):
    """CNN backbone with a residual block and a Bayesian output."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            ResidualBlock(6, 15, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )

        # Determine the flattened feature size
        dummy = torch.zeros(1, 3, 32, 32)
        feat_size = self.features(dummy).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(feat_size, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 1),
        )
        self.output_head = BayesianOutput(1)

    def forward(self, x: torch.Tensor) -> Bernoulli:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return self.output_head(x)


__all__ = ["ResidualBlock", "BayesianOutput", "HybridBinaryClassifier"]
