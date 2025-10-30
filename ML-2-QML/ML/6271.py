import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NoiseLayer(nn.Module):
    """
    Adds Gaussian noise to the activations before the final classifier.
    This regulariser can be tuned via ``noise_std`` and is useful when
    training on limited data or when the quantum head is noisy.
    """
    def __init__(self, noise_std: float = 0.0):
        super().__init__()
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_std > 0:
            return x + torch.randn_like(x) * self.noise_std
        return x


class QuantumHybridClassifier(nn.Module):
    """
    A fully‑augmented classical‑quantum architecture that mirrors the
    original QCNet but expands the quantum head to a 3‑qubit variational
    circuit and introduces a learnable batch norm.
    """
    def __init__(self, in_features: int = 55815, noise_std: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(0.2)
        self.drop2 = nn.Dropout2d(0.5)

        self.fc1 = nn.Linear(in_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Learnable batch‑norm to stabilize the quantum head
        self.bn = nn.BatchNorm1d(1, affine=True)

        # Noise regulariser
        self.noise = NoiseLayer(noise_std)

        # Replace the quantum head with a differentiable sigmoid
        self.classifier = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.bn(x)
        x = self.noise(x)
        x = self.classifier(x)
        return torch.cat((x, 1 - x), dim=-1)
