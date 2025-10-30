import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumHybridBinaryClassifier(nn.Module):
    """
    Classical binary classifier that mirrors the hybrid quantum model.
    Contains a CNN backbone, an MLP head, and a dropout‑based calibration
    layer to produce calibrated probabilities.
    """
    def __init__(self, dropout_rate: float = 0.5) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop_conv = nn.Dropout2d(p=0.2)

        # Fully‑connected backbone
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Calibration head (small MLP with dropout)
        self.calibration = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(1, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop_conv(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop_conv(x)

        # Flatten and fully‑connected layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Calibration head
        logits = self.calibration(x)
        probs = F.softmax(logits, dim=-1)
        return probs

__all__ = ["QuantumHybridBinaryClassifier"]
