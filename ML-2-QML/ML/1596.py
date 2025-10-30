import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridFunction(nn.Module):
    """
    Classical head that maps logits to probabilities via sigmoid.
    A learnable shift and optional dropout increase expressivity.
    """
    def __init__(self, shift: float = 0.0, dropout: float | None = None):
        super().__init__()
        self.shift = nn.Parameter(torch.tensor(shift))
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            logits = self.dropout(logits)
        probs = torch.sigmoid(logits + self.shift)
        return probs

class HybridBinaryClassifier(nn.Module):
    """
    Fully‑classical binary classifier mirroring the original hybrid architecture.
    It comprises a small CNN, a dense head, and the HybridFunction.
    """
    def __init__(self, dropout: float | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.head = HybridFunction(shift=0.0, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.head(x)
        return torch.cat([probs, 1 - probs], dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(x)
