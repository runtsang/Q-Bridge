import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNatEnhanced(nn.Module):
    """
    Classical CNN to 8‑class classifier.
    Builds on the original Quantum‑NAT architecture by adding
    additional convolutional layers, dropout, and a two‑stage
    fully‑connected head that first projects to a 4‑dimensional
    latent space and then to 8 logits.
    """
    def __init__(self) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        # Flattened size after 3 poolings on 28x28 input
        self._feature_dim = 64 * 3 * 3  # 28 -> 14 -> 7 -> 3

        # Fully‑connected head
        self.fc = nn.Sequential(
            nn.Linear(self._feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, 4),  # latent 4‑dim
        )
        # Final classifier
        self.classifier = nn.Linear(4, 8)

        # Normalisation for the final logits
        self.norm = nn.BatchNorm1d(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feats = self.features(x)
        flattened = feats.view(bsz, -1)
        latent = self.fc(flattened)
        logits = self.classifier(latent)
        return self.norm(logits)

__all__ = ["QuantumNatEnhanced"]
