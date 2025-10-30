import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable

class FullyConnectedLayer(nn.Module):
    """Classical approximation of the quantum FCL."""
    def __init__(self, n_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        return torch.tanh(self.linear(values)).mean(dim=0, keepdim=True)

class ClassicalHybridHead(nn.Module):
    """Sigmoid head that can replace a quantum expectation layer."""
    def __init__(self, shift: float = 0.0):
        super().__init__()
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return torch.sigmoid(logits + self.shift)

class UnifiedHybridClassifier(nn.Module):
    """
    Classical hybrid classifier that uses a CNN backbone followed by
    either a classical sigmoid head or a quantum expectation head.
    The quantum variant is defined in the QML module.
    """
    def __init__(self, mode: str = "classical", **kwargs):
        super().__init__()
        self.mode = mode
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        if mode == "classical":
            self.head = ClassicalHybridHead(shift=kwargs.get("shift", 0.0))
        else:
            # Placeholder; quantum head is provided in qml module
            self.head = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
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
        x = self.fc3(x)
        if self.head is None:
            raise RuntimeError(
                "Quantum head not available in classical module. "
                "Import the QML version for quantum execution."
            )
        probs = self.head(x).unsqueeze(-1)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["FullyConnectedLayer", "ClassicalHybridHead", "UnifiedHybridClassifier"]
