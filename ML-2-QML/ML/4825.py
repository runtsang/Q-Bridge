"""Hybrid fraud detection model combining CNN feature extraction with a quantum layer placeholder.

The class `FraudDetectionHybridClassifier` is a purely classical PyTorch module that extracts
features from an image, projects them to a 4‑dimensional vector and forwards these values
to an external quantum routine.  The quantum routine must be supplied by the user
(e.g. the `QuantumFraudLayer` defined in the QML module).  This design keeps the
classical training pipeline independent of any quantum SDK while still allowing
plug‑and‑play of a quantum backend.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional

class FraudDetectionHybridClassifier(nn.Module):
    """
    Hybrid fraud detection classifier.

    Args:
        quantum_forward: Optional callable that accepts a batch of 4‑dimensional
            tensors and returns processed tensors.  If None, the identity
            function is used.
    """

    def __init__(self, quantum_forward: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        super().__init__()
        self.quantum_forward = quantum_forward or (lambda x: x)

        # Convolutional feature extractor (from QuantumNAT)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Projection to 4‑dimensional feature vector
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(4, 2)  # binary fraud / legit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid model.

        Parameters
        ----------
        x: torch.Tensor
            Input image tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, 2).
        """
        batch = x.shape[0]
        feat = self.features(x)
        feat = feat.view(batch, -1)
        feat4 = self.fc(feat)
        # Pass through quantum routine
        processed = self.quantum_forward(feat4)
        logits = self.classifier(processed)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the 4‑dimensional feature vector before the quantum layer.
        """
        batch = x.shape[0]
        feat = self.features(x).view(batch, -1)
        return self.fc(feat)

__all__ = ["FraudDetectionHybridClassifier"]
