import torch
import torch.nn as nn
import torch.nn.functional as F
from QCNN import QCNNModel
from SamplerQNN import SamplerQNN

class HybridQCNNSamplerNet(nn.Module):
    """Hybrid CNN + QCNN + Sampler architecture for binary classification."""
    def __init__(self):
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Fully connected stack
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)  # 8‑dim feature for QCNN
        # Classical QCNN emulation
        self.qcnn = QCNNModel()
        # Linear mapping to 2‑dim sampler input
        self.to_sampler = nn.Linear(1, 2)
        # Sampler network
        self.sampler = SamplerQNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional feature extractor
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # (batch, 8)
        # Classical QCNN head
        qcnn_out = self.qcnn(x)  # (batch, 1)
        # Map to 2‑dim sampler input
        sampler_in = self.to_sampler(qcnn_out)  # (batch, 2)
        # Sampler output
        probs = self.sampler(sampler_in)
        return probs

__all__ = ["HybridQCNNSamplerNet"]
