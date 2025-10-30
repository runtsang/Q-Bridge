import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATExtended(nn.Module):
    """
    Classical CNN + fully‑connected architecture with residual connections and dropout.
    Builds upon the original QFCModel by adding depth and regularisation while
    preserving the 4‑class output structure.
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.2, use_residual: bool = True):
        super().__init__()
        # Feature extractor: two conv blocks with batch norm and ReLU
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Optional residual projection
        if use_residual:
            self.res_proj = nn.Conv2d(16, 16, kernel_size=1)
        else:
            self.res_proj = None

        # Fully‑connected head with two hidden layers
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if self.res_proj is not None:
            residual = self.res_proj(x)
            x = x + residual
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.norm(x)

__all__ = ["QuantumNATExtended"]
