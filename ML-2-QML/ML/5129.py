import torch
from torch import nn
import torch.nn.functional as F

class HybridQCNN(nn.Module):
    """Hybrid classical model combining a CNN feature extractor, a QCNN-inspired classifier,
    and a softmax sampler.

    The architecture mirrors the quantum QCNN circuit (QCNNModel) while adding a
    convolutional backbone (QFCModel features) and a probabilistic output
    (SamplerQNN).  This design allows end‑to‑end training on classical hardware
    while preserving the depth‑wise structure of the quantum counterpart.
    """
    def __init__(self) -> None:
        super().__init__()
        # Convolutional backbone (from QuantumNAT)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # QCNN-inspired classifier
        self.classifier = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 2),
        )
        # Softmax sampler
        self.sampler = nn.Softmax(dim=-1)
        self.norm = nn.BatchNorm1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        f = self.features(x)
        f = f.view(f.size(0), -1)
        # QCNN classifier
        h = self.classifier(f)
        h = self.norm(h)
        # Probabilistic output
        out = self.sampler(h)
        return out

__all__ = ["HybridQCNN"]
