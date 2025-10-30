import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridQuantumNAT(nn.Module):
    """
    Classical convolutional backbone with a quantum-inspired random layer and a final fully connected head.
    The architecture combines ideas from the original QFCModel (multi‑scale CNN + FC) and the
    QuanvolutionFilter (patch‑wise processing).  It replaces the single 2‑layer CNN with
    a deeper 3‑layer CNN, adds a random feature mapping that mimics a quantum kernel,
    and preserves the 4‑dimensional output with batch‑norm for stability.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 4, seed: int | None = None) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Classical convolutional backbone
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Quantum‑inspired random feature mapping
        # Random linear layer followed by a non‑linear activation
        self.random_proj = nn.Linear(32 * 3 * 3, 128, bias=False)
        nn.init.orthogonal_(self.random_proj.weight)
        self.random_activation = nn.SiLU()

        # Final classification head
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, channels, height, width).

        Returns
        -------
        torch.Tensor
            Normalised logits of shape (batch, num_classes).
        """
        bsz = x.size(0)
        feat = self.features(x)          # (bsz, 32, 3, 3)
        flat = feat.view(bsz, -1)        # (bsz, 32*3*3)
        random_feat = self.random_activation(self.random_proj(flat))
        logits = self.head(random_feat)
        return self.norm(logits)

__all__ = ["HybridQuantumNAT"]
