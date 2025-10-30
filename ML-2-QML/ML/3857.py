import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSamplerQNN(nn.Module):
    """
    Classical hybrid sampler network that extracts image features with a lightweight CNN
    (inspired by Quantum‑NAT) and projects them to a 4‑dimensional probability vector via a
    fully connected head followed by a softmax. This mirrors the structure of the original
    SamplerQNN but augments it with convolutional pre‑processing.
    """
    def __init__(self) -> None:
        super().__init__()
        # Feature extractor (adapted from QFCModel.features)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Fully connected projection to 4 logits
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)
        )
        # Normalisation before softmax
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. Extract features.
        2. Flatten and project with fully connected layers.
        3. Apply batch norm and softmax to obtain probability distribution.
        """
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        logits = self.fc(flattened)
        logits = self.norm(logits)
        probs = F.softmax(logits, dim=-1)
        return probs

__all__ = ["HybridSamplerQNN"]
