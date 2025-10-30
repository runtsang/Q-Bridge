import torch
import torch.nn as nn

class UnifiedQCNN(nn.Module):
    """
    Classical backbone that produces the 8‑dimensional vector required
    by the quantum QCNN.  It mirrors the convolutional part of the
    original QCNN and the Quantum‑NAT CNN, providing a scalable
    pre‑processing pipeline.
    """
    def __init__(self) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Projection to 8‑dimensional quantum feature vector
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )
        # Final classifier matching the QCNN output
        self.head = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.head(x)

def create_unified_qcnn() -> UnifiedQCNN:
    """Factory returning the configured :class:`UnifiedQCNN`."""
    return UnifiedQCNN()

__all__ = ["UnifiedQCNN", "create_unified_qcnn"]
