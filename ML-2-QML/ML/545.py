import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumNATEnhanced(nn.Module):
    """
    Hybrid classical‑quantum inspired model that extends the original QFCModel.
    The classical branch uses a deeper CNN encoder, and the output is projected
    to four features.  The class is intentionally kept fully classical so that
    it can be used as a drop‑in replacement in existing pipelines.
    """
    def __init__(self) -> None:
        super().__init__()
        # Encoder: 3 conv layers + batchnorm + pool
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        # Projection to 4 features
        self.projector = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Normalized 4‑dimensional output of shape (B, 4).
        """
        bsz = x.shape[0]
        features = self.encoder(x)
        flattened = features.view(bsz, -1)
        out = self.projector(flattened)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
