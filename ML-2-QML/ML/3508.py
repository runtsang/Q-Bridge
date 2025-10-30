import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATGen128(nn.Module):
    """
    Hybrid CNN that extracts a 128‑dimensional embedding from a single‑channel
    image and projects it to four class logits.  The embedding matches the
    7‑qubit amplitude‑encoding used in the quantum companion.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 4,
                 embed_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                     # 14×14
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                     # 7×7
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),        # 1×1
        )
        self.embed_dim = embed_dim
        self.fc = nn.Sequential(
            nn.Linear(32, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_classes)
        )
        self.bn = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (batch, num_classes) after batch‑norm
            embed : (batch, embed_dim) ready for quantum encoding
        """
        bsz = x.shape[0]
        feat = self.features(x)                # (bsz, 32, 1, 1)
        flat = feat.view(bsz, -1)              # (bsz, 32)
        embed = self.fc[0](flat)               # (bsz, embed_dim)
        logits = self.fc[2](embed)             # (bsz, num_classes)
        return self.bn(logits), embed

__all__ = ["QuantumNATGen128"]
