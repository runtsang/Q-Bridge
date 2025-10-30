import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

class HybridQuantumClassifier(nn.Module):
    """
    Classical PyTorch implementation of the hybrid classifier.
    It mirrors the architecture of the original model but replaces
    the quantum head with a small MLP and a configurable dropout
    schedule that can be tuned for different training regimes.
    """

    def __init__(
        self,
        conv_dropout: Sequence[float] = (0.2, 0.5),
        mlp_dropout: Sequence[float] = (0.3, 0.4),
        mlp_hidden: int = 120,
        mlp_second_hidden: int = 84,
    ) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=conv_dropout[0])
        self.drop2 = nn.Dropout2d(p=conv_dropout[1])

        # Dummy forward to compute flattened feature size
        dummy = torch.zeros(1, 3, 32, 32)
        dummy = self._forward_conv(dummy)
        flat_features = dummy.size(1)

        # MLP head before the final sigmoid
        self.fc1 = nn.Linear(flat_features, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, mlp_second_hidden)
        self.fc3 = nn.Linear(mlp_second_hidden, 1)

        self.dropout1 = nn.Dropout(p=mlp_dropout[0])
        self.dropout2 = nn.Dropout(p=mlp_dropout[1])

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop2(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_conv(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        logits = self.fc3(x).squeeze(-1)
        probs = torch.sigmoid(logits)
        return torch.stack([probs, 1 - probs], dim=-1)
