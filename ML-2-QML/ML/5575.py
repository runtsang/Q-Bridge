import torch
import torch.nn as nn
import torch.nn.functional as F
# Import the quantum hybrid layer from the QML module
from.QCNN__gen482_qml import HybridQCNN as QuantumHybridQCNN

class HybridQCNN(nn.Module):
    """Hybrid classical‑quantum network combining a CNN feature extractor with a variational QCNN.
    The classical backbone follows the QFCModel design, producing a 4‑dimensional embedding
    that is fed into the quantum layer returned by :class:`QCNN__gen482_qml.HybridQCNN`.
    The final linear head maps the quantum output to class logits.
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Classical feature extractor (inspired by QFCModel)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully‑connected projection to 4 features
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        # Quantum variational QCNN layer
        self.quantum = QuantumHybridQCNN()
        # Final classifier
        self.classifier = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        proj = self.fc(flat)
        # Quantum layer expects input shape (batch, 4)
        qout = self.quantum(proj)
        logits = self.classifier(qout)
        return logits

__all__ = ["HybridQCNN"]
