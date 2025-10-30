import torch
import torch.nn as nn
import torch.nn.functional as F
from quantum_module import QCNet as QuantumQCNet

class QCNet(nn.Module):
    """Hybrid CNN + quantum classifier for binary classification."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Convolutional backbone
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.flatten = nn.Flatten()
        self.classical_head = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        # Quantum head
        self.quantum = QuantumQCNet(num_qubits=4, device="default.qubit")
        # Final classifier
        self.classifier = nn.Linear(1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.classical_head(x)
        q = self.quantum(x)
        out = self.classifier(q)
        return out
