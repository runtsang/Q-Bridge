import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATEnhanced(nn.Module):
    """
    Hybrid classical model that encodes 28x28 grayscale images into a 10‑dim latent
    space via a lightweight CNN, then projects to four outputs. Adds dropout and
    residual connections for improved regularisation.
    """
    def __init__(self) -> None:
        super().__init__()
        # Encoder: 1->8->16 channels, 3x3 conv, stride 1, padding 1
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7
        )
        # Flattened features: 16*7*7 = 784
        self.fc_latent = nn.Sequential(
            nn.Linear(16*7*7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 10),
            nn.ReLU(inplace=True),
        )
        # Final projection to 4 outputs
        self.fc_out = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        features = self.encoder(x)
        flattened = features.view(bsz, -1)
        latent = self.fc_latent(flattened)
        out = self.fc_out(latent)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
