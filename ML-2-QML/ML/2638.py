"""
Classical implementation of the hybrid quanvolution binary classifier.
It combines a CNN backbone with a classical analogue of the quanvolution
filter and a dense head producing binary probabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalQuanvolutionFilter(nn.Module):
    """
    Classical 2×2 convolutional filter that mimics the patch‑wise
    operation of the quantum quanvolution.  It reduces a single‑channel
    image to 4 feature maps, then flattens the result.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Expect input shape (N, 1, H, W)
        features = self.conv(x)
        return features.view(x.size(0), -1)


class HybridQuanvolutionBinaryClassifier(nn.Module):
    """
    Classical hybrid network:
        * Convolutional backbone (two conv layers + pooling + dropout)
        * Classical quanvolution branch
        * Concatenation of both feature vectors
        * Dense head producing a single logit
        * Sigmoid activation to obtain class probabilities
    """
    def __init__(self) -> None:
        super().__init__()
        # Backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        # Quanvolution branch
        self.quanvolution = ClassicalQuanvolutionFilter()
        # Final classifier
        self.classifier = nn.Linear(84 + 4 * 14 * 14, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Backbone
        xb = F.relu(self.conv1(x))
        xb = self.pool(xb)
        xb = self.drop1(xb)
        xb = F.relu(self.conv2(xb))
        xb = self.pool(xb)
        xb = self.drop1(xb)
        xb = torch.flatten(xb, 1)
        xb = F.relu(self.fc1(xb))
        xb = self.drop2(xb)
        xb = F.relu(self.fc2(xb))
        # Quanvolution branch
        # Convert RGB to grayscale for the classical filter
        qx = x.mean(dim=1, keepdim=True)
        qb = self.quanvolution(qx)
        # Concatenate and classify
        combined = torch.cat((xb, qb), dim=1)
        logits = self.classifier(combined)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["HybridQuanvolutionBinaryClassifier", "ClassicalQuanvolutionFilter"]
