import torch
import torch.nn as nn
import torch.nn.functional as F


class QCNNModel(nn.Module):
    """Classical fully‑connected network that mimics QCNN convolution and pooling."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


class HybridCNNQML(nn.Module):
    """
    Classical hybrid model that combines a lightweight CNN backbone with a QCNN‑style
    fully‑connected block and a final binary classification head.
    """
    def __init__(self) -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Dimensionality reduction to match QCNNModel input (8)
        self.fc_reduce = nn.Linear(8, 8)

        # QCNN‑style fully‑connected block
        self.qcnn = QCNNModel()

        # Final binary classification head
        self.head = nn.Linear(1, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)          # shape: (batch, 8)
        x = self.fc_reduce(x)            # shape: (batch, 8)
        x = self.qcnn(x)                 # shape: (batch, 1)
        logits = self.head(x)            # shape: (batch, 2)
        return logits


__all__ = ["QCNNModel", "HybridCNNQML"]
