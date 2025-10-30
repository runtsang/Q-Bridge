"""HybridAdvancedClassifier
========================

Model architecture for a binary classification task that
improves upon the original classical‑only and hybrid models.
The design introduces:
*   a residual‑dense block that leverages both conv‑and‑dense
    layers for better feature learning.
*   **Bayesian** (Monte‑Carlo dropout) head for  calibration
    and confidence‑estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualDenseBlock(nn.Module):
    """Residual block with dense connections across layers."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn3 = nn.BatchNorm2d(out_channels)
        if in_channels!= out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Identity()
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
        residual = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += residual
        out = F.relu(out)
        out = self.dropout(out)
        return out

class HybridAdvancedClassifier(nn.Module):
    """
    Classical neural‑network head with Bayesian dropout for calibration.
    The architecture mirrors the original QCNet but replaces the quantum
    expectation layer with a residual‑dense block followed by a
    Monte‑Carlo dropout head.
    """
    def __init__(self, dropout_rate: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.rd_block = ResidualDenseBlock(6, 15)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mc_samples: int = 1) -> torch.Tensor:
        """
        Forward pass with optional Monte‑Carlo dropout sampling.
        Args:
            x: input image tensor of shape (B,3,H,W)
            mc_samples: number of stochastic forward passes for Bayesian inference
        Returns:
            Tensor of shape (B,2) containing probability estimates for the two
            classes. If mc_samples > 1, the returned tensor is the mean of
            the predictions across the samples.
        """
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.rd_block(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if mc_samples > 1:
            preds = []
            self.train()
            for _ in range(mc_samples):
                y = self.dropout(x)
                prob = torch.sigmoid(y)
                preds.append(prob)
            probs = torch.stack(preds, dim=0).mean(0)
            self.eval()
        else:
            probs = torch.sigmoid(self.dropout(x))
        return torch.cat((probs, 1 - probs), dim=-1)

    def get_mc_predictions(self, x: torch.Tensor, mc_samples: int = 10) -> torch.Tensor:
        """
        Convenience wrapper for generating predictions with Monte‑Carlo dropout.
        """
        return self.forward(x, mc_samples=mc_samples)

__all__ = ["HybridAdvancedClassifier", "ResidualDenseBlock"]
