import torch
from torch import nn
import torch.nn.functional as F

class HybridLayer(nn.Module):
    """Dense layer with a sigmoid activation that emulates a quantum expectation."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return torch.sigmoid(logits + self.shift)


class QCNNGen169(nn.Module):
    """
    Classical convolution‑inspired network that optionally delegates the final
    classification/regression head to a quantum module.  The architecture
    mirrors the original QCNN but adds a flexible head that can be swapped
    between a dense sigmoid head and a quantum expectation head.
    """
    def __init__(self,
                 in_features: int = 8,
                 num_classes: int = 2,
                 use_quantum_head: bool = False) -> None:
        super().__init__()
        self.use_quantum_head = use_quantum_head
        self.feature_map = nn.Sequential(
            nn.Linear(in_features, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh()
        )
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

        if num_classes == 1:
            self.head = nn.Linear(4, 1)
        else:
            self.head = nn.Linear(4, num_classes)

        self.hybrid = HybridLayer(1, shift=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        logits = self.head(x)

        if self.use_quantum_head:
            # The quantum head is implemented in the QML module.
            # Here we simply return the logits; the caller will
            # pass them to a quantum estimator if desired.
            return logits
        else:
            return self.hybrid(logits)


class RegressionDataset(torch.utils.data.Dataset):
    """
    Simple regression dataset that mirrors the quantum superposition data.
    """
    def __init__(self, samples: int, num_features: int) -> None:
        self.features, self.labels = self._generate(samples, num_features)

    @staticmethod
    def _generate(samples: int, num_features: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {"states": self.features[idx], "target": self.labels[idx]}


class QCNet(nn.Module):
    """
    Hybrid CNN + quantum expectation head for binary classification.
    Mirrors the QCNet from reference 3 but uses the HybridLayer defined above.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = HybridLayer(1, shift=0.0)

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
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = [
    "QCNNGen169",
    "RegressionDataset",
    "QCNet",
    "HybridLayer",
]
