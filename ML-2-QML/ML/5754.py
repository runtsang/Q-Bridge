"""Classical neural net with an optional quantum fusion layer.

The backbone is a lightweight ResNet‑style feature extractor.  The head can be a
purely classical linear layer or a quantum fusion head that forwards the same
feature vector to a quantum backend.  The two heads are blended with a
user‑controlled weight, allowing gradual experimentation with quantum
contributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# --------------------------------------------------------------------------- #
# Backbone
# --------------------------------------------------------------------------- #
class ResidualBlock(nn.Module):
    """Two‑layer residual block."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class FeatureExtractor(nn.Module):
    """ResNet‑style backbone that outputs a dense feature vector."""
    def __init__(self, in_channels: int = 3, out_features: int = 84):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block1 = ResidualBlock(32)
        self.block2 = ResidualBlock(32)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.fc(x)

# --------------------------------------------------------------------------- #
# Heads
# --------------------------------------------------------------------------- #
class ClassicalHead(nn.Module):
    """Linear head that produces a single logit."""
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class QuantumFusionHead(nn.Module):
    """Wrapper that forwards features to a quantum backend."""
    def __init__(self, quantum_module: Optional[object] = None):
        super().__init__()
        self.quantum_module = quantum_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantum_module is None:
            raise RuntimeError("Quantum module not provided.")
        # The quantum module should expose a forward method that accepts a 1‑D tensor
        # of parameters and returns a scalar logit.
        return self.quantum_module(x)

# --------------------------------------------------------------------------- #
# Hybrid classifier
# --------------------------------------------------------------------------- #
class QuantumHybridBinaryClassifier(nn.Module):
    """End‑to‑end hybrid classifier with optional quantum fusion."""
    def __init__(self,
                 in_channels: int = 3,
                 feature_dim: int = 84,
                 use_quantum: bool = False,
                 quantum_module: Optional[object] = None,
                 fusion_weight: float = 0.5):
        super().__init__()
        self.backbone = FeatureExtractor(in_channels, feature_dim)
        self.classical_head = ClassicalHead(feature_dim)
        self.use_quantum = use_quantum
        self.quantum_head = QuantumFusionHead(quantum_module) if use_quantum else None
        self.fusion_weight = fusion_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classical_head(features)
        probs = torch.sigmoid(logits)

        if self.use_quantum and self.quantum_head is not None:
            q_logits = self.quantum_head(features)
            q_probs = torch.sigmoid(q_logits)
            probs = self.fusion_weight * probs + (1.0 - self.fusion_weight) * q_probs

        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["ResidualBlock",
           "FeatureExtractor",
           "ClassicalHead",
           "QuantumFusionHead",
           "QuantumHybridBinaryClassifier"]
