import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATHybrid(nn.Module):
    """
    Classical hybrid model: CNN feature extractor -> linear projection -> MLP head.
    The projection mimics a 4‑qubit quantum measurement output.
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        # Feature extractor: CNN
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Flatten and projection to 4 features (simulate quantum output)
        self.projection = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 4),
        )
        # MLP head for classification
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(4),
            nn.Linear(4, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.feature_extractor(x)
        flattened = features.view(bsz, -1)
        q_like = self.projection(flattened)  # shape (bsz, 4)
        logits = self.classifier(q_like)
        return logits
