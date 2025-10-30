"""QuantumNAT Enhanced – classical component with hybrid encoder and classifier head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import QuantumNAT_QML  # Import the quantum block module

__all__ = ["QuantumNAT_Enhanced"]

class QuantumNAT_Enhanced(nn.Module):
    """
    Hybrid classical-quantum neural network that extends the original Quantum-NAT.
    Features:
      * 2D convolutional feature extractor.
      * Linear encoder mapping flattened features to a 16-dim vector.
      * Quantum variational block producing a 4-dim latent.
      * Linear classifier mapping latent to logits.
    """
    def __init__(self, n_classes: int = 4, n_qubits: int = 4, n_layers: int = 3):
        super().__init__()
        # Feature extractor identical to the seed
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Encoder to reduce dimensionality before quantum block
        self.encoder = nn.Linear(16 * 7 * 7, 16)
        # Quantum block
        self.quantum = QuantumNAT_QML.QuantumNAT_Enhanced(
            n_qubits=n_qubits, n_layers=n_layers
        )
        # Classifier head
        self.classifier = nn.Linear(4, n_classes)
        self.norm = nn.BatchNorm1d(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        encoded = self.encoder(flattened)
        latent = self.quantum(encoded)  # 4-dim latent
        logits = self.classifier(latent)
        return self.norm(logits)
