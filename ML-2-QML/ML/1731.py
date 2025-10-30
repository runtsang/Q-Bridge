import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ParameterShuffle(nn.Module):
    """Randomly permutes the feature dimension for data augmentation."""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or torch.rand(1).item() > self.p:
            return x
        batch_size, *dims = x.shape
        flat = x.view(batch_size, -1)
        n = flat.size(1)
        perm = torch.randperm(n, device=flat.device)
        return flat[:, perm].view(batch_size, *dims)

class ResidualBlock(nn.Module):
    """Two‑layer residual block with batch‑norm and dropout."""
    def __init__(self, in_features, out_features, dropout=0.1, activation=F.relu):
        super().__init__()
        self.activation = activation
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1   = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2   = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        return self.activation(out + residual)

class ClassicalHybridClassifier(nn.Module):
    """Classical CNN with residual blocks and a quantum‑like sigmoid head."""
    def __init__(self, num_classes=2, dropout=0.25):
        super().__init__()
        # Feature extractor
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=2)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.shuffle = ParameterShuffle(p=0.3)

        # Fully connected part
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.res1 = ResidualBlock(256, 256, dropout)
        self.fc2 = nn.Linear(256, 128)
        self.res2 = ResidualBlock(128, 128, dropout)

        # Quantum‑like head
        self.head = nn.Linear(128, 1)
        self.shift = 0.0

    def forward(self, x):
        # Convolutional backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.shuffle(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected with residuals
        x = F.relu(self.fc1(x))
        x = self.res1(x)
        x = F.relu(self.fc2(x))
        x = self.res2(x)

        logits = self.head(x)
        probs = torch.sigmoid(logits + self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["ParameterShuffle", "ResidualBlock", "ClassicalHybridClassifier"]
