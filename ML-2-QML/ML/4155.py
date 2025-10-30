"""QCNNHybrid: classical convolutional network inspired by QCNN architecture.

The network mirrors the layers of the quantum QCNN but replaces the
quantum expectation head with a lightweight classical head that
provides a differentiable sigmoid output.  The architecture is
parameterised by `depth` so that the number of fully‑connected layers
can be tuned, and by `shift` which is added before the sigmoid to
control the activation range.  This design allows a direct
comparison with the quantum counterpart while remaining fully
classical and fast on CPUs/GPUs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class QCNNHybrid(nn.Module):
    """Classical QCNN‑style network.

    Parameters
    ----------
    depth : int
        Number of fully‑connected layers after the convolutional blocks.
    shift : float, default 0.0
        Bias added before the sigmoid head to emulate the quantum shift.
    """

    def __init__(self, depth: int = 2, shift: float = 0.0) -> None:
        super().__init__()
        # Convolutional block
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Feature extractor flattening
        dummy = torch.zeros(1, 3, 224, 224)  # placeholder to compute flatten size
        with torch.no_grad():
            x = self._extract_features(dummy)
        flat_dim = x.shape[1]

        # Fully‑connected head
        layers = [nn.Linear(flat_dim, 120), nn.ReLU(), nn.Dropout(p=0.5)]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(120, 120), nn.ReLU(), nn.Dropout(p=0.5)])
        layers.append(nn.Linear(120, 1))
        self.fc = nn.Sequential(*layers)

        self.shift = shift

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._extract_features(x)
        logits = self.fc(x).squeeze(-1)
        probs = torch.sigmoid(logits + self.shift)
        return torch.stack([probs, 1 - probs], dim=-1)

def QCNN() -> QCNNHybrid:
    """Factory returning a default QCNNHybrid instance."""
    return QCNNHybrid()
