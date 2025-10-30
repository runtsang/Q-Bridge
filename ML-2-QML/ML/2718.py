import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantumHybridNAT(nn.Module):
    """
    Classical hybrid model that combines a convolutional backbone,
    a quantum‑kernel emulation, and a fully‑connected head.
    """

    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        # Convolutional backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Quantum‑kernel emulation: a linear layer that mimics the
        # feature mapping performed by the quantum circuit in the QML version.
        self.kernel = nn.Linear(16 * 7 * 7, 64)
        # Fully‑connected head
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
        self.norm = nn.BatchNorm1d(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (batch_size, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Normalized logits of shape (batch_size, n_classes).
        """
        bsz = x.shape[0]
        features = self.backbone(x)          # (bsz, 16, 7, 7)
        flattened = features.view(bsz, -1)   # (bsz, 16*7*7)
        kernel_out = self.kernel(flattened)  # (bsz, 64)
        logits = self.head(kernel_out)       # (bsz, n_classes)
        return self.norm(logits)

__all__ = ["QuantumHybridNAT"]
