import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridNATEstimator(nn.Module):
    """Hybrid classical‑quantum regression model.

    Combines a CNN feature extractor, a quantum regression layer,
    and a fully‑connected head.
    """

    def __init__(self, quantum_layer: nn.Module):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Expect 16‑dim feature vector from encoder
        self.quantum_layer = quantum_layer
        self.head = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )
        self.norm = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.cnn(x)
        # Flatten and reduce to 16‑dim vector
        flattened = features.view(bsz, -1)
        pooled = flattened.reshape(bsz, 16, -1).mean(dim=2)
        q_out = self.quantum_layer(pooled)
        out = self.head(q_out)
        return self.norm(out)
