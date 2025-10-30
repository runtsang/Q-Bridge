"""Classical implementation of the hybrid classifier with a feature selector and a classical head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FeatureSelector(nn.Module):
    """Selects the top‑k activations from a dense layer based on learned importance scores."""
    def __init__(self, in_features: int, k: int):
        super().__init__()
        self.k = k
        self.score = nn.Parameter(torch.randn(in_features))

    def forward(self, x: torch.Tensor):
        probs = F.softmax(self.score, dim=0)
        _, idx = torch.topk(probs, self.k)
        return torch.index_select(x, 1, idx)

class ClassicalHead(nn.Module):
    """Simple classical head that maps selected features to a binary logit."""
    def __init__(self, in_features: int, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class QuantumHybridClassifier(nn.Module):
    """CNN followed by a feature selector and a classical head that mimics a quantum expectation."""
    def __init__(self, k: int = 10, hidden: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.selector = FeatureSelector(in_features=84, k=k)
        self.head = ClassicalHead(in_features=k, hidden=hidden)

    def forward(self, inputs: torch.Tensor):
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.selector(x)
        logits = self.head(x)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["QuantumHybridClassifier"]
