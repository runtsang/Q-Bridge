import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATHybrid(nn.Module):
    """
    Classical CNN‑to‑MLP pipeline that mirrors the original Quantum‑NAT architecture
    but adds depth, a learnable feature‑mixing layer, and an optional
    multi‑class classifier head.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default: 1 for MNIST‑style images).
    num_classes : int
        Number of output classes (default: 4).
    dropout : float
        Dropout probability applied after the feature‑mixing layer.
    """
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 4,
                 dropout: float = 0.3) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Feature mixing
        self.mix = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),          # map to 4 “quantum” features
            nn.BatchNorm1d(4),
            nn.Dropout(dropout)
        )
        # Classical classifier head
        self.classifier = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, num_classes)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, height, width).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes).
        """
        feat = self.features(x)
        flat = feat.view(feat.size(0), -1)
        mixed = self.mix(flat)
        logits = self.classifier(mixed)
        return logits

__all__ = ["QuantumNATHybrid"]
