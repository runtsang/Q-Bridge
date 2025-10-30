import torch
import torch.nn as nn
import torch.nn.functional as F
from.quantum_encoder import QuantumEncoder

class QuantumHybridNAT(nn.Module):
    """Hybrid classical‑quantum model combining a CNN backbone with a quantum variational kernel.

    The model first extracts classical features via a lightweight CNN, then applies a
    quantum kernel on 2×2 patches of the input image.  The two embeddings are
    concatenated and passed through a final fully‑connected head.  The architecture
    is inspired by the Quantum‑NAT and Quanvolution examples and is designed to be
    trainable with standard gradient‑based optimizers.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Classical convolutional backbone (similar to QFCModel)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 28x28 -> 14x14
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 14x14 -> 7x7
        )
        self.classical_proj = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
        )

        # Quantum kernel on image patches
        self.quantum_encoder = QuantumEncoder()

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(32 + 4 * 14 * 14, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Classical pathway
        cls_feat = self.features(x)
        cls_feat = cls_feat.view(bsz, -1)
        cls_feat = self.classical_proj(cls_feat)

        # Quantum pathway
        q_feat = self.quantum_encoder(x)  # shape (bsz, 4*14*14)

        # Concatenate
        combined = torch.cat([cls_feat, q_feat], dim=1)

        # Classification head
        logits = self.classifier(combined)
        return logits

__all__ = ["QuantumHybridNAT"]
