import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

class FraudLayer(nn.Module):
    """Custom linear layer that applies photonic‑inspired scaling and shifting."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer("scale", torch.ones(out_features))
        self.register_buffer("shift", torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        return out * self.scale + self.shift

class FraudDetectionHybrid(nn.Module):
    """
    Classical fraud detection model.
    Combines a lightweight CNN with custom linear layers featuring
    scale/shift buffers inspired by photonic displacement.
    """
    def __init__(self, num_channels: int = 1, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            FraudLayer(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            FraudLayer(64, 32),
            nn.ReLU(inplace=True),
            FraudLayer(32, num_classes),
        )
        self.prob = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        logits = self.fc(x)
        return logits

class FraudDataset(Dataset):
    """
    Synthetic fraud dataset using superposition data generation.
    Labels are binarised from continuous outputs.
    """
    def __init__(self, samples: int = 10000, num_features: int = 2):
        x, y = generate_superposition_data(num_features, samples)
        labels = (y > 0).astype(np.int64)
        self.data = torch.tensor(x, dtype=torch.float32).unsqueeze(1)  # shape (N,1,2)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for fraud detection.
    Returns input matrix (samples, num_features) and continuous target.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

__all__ = ["FraudDetectionHybrid", "FraudDataset", "generate_superposition_data"]
