import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQuantumNAT(nn.Module):
    """
    Classical hybrid model inspired by QuantumNAT.
    Features: CNN backbone (as in QuantumNAT.py ML seed),
              fully connected head producing 4 features,
              custom sigmoid-based activation for each class.
    The output is a 4‑class probability distribution.
    """

    def __init__(self, shift: float = 0.0):
        super().__init__()
        # CNN backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        logits = self.fc(flattened)
        logits = self.norm(logits)
        # apply shift and sigmoid to each logit
        probs = torch.sigmoid(logits + self.shift)
        # normalize to sum to 1
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs

__all__ = ["HybridQuantumNAT"]
