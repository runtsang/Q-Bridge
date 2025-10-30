import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridBinaryClassifier(nn.Module):
    """Classical CNN + 4‑dimensional projection + sigmoid head for binary classification."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )
        # 4‑dimensional projection mimicking Quantum‑NAT
        self.fc_proj = nn.Sequential(
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 4),
        )
        self.norm = nn.BatchNorm1d(4)
        self.classifier = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc_proj(x)
        x = self.norm(x)
        logits = self.classifier(x)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridBinaryClassifier"]
