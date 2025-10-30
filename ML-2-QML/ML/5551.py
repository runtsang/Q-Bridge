import torch
import torch.nn as nn
import torch.nn.functional as F

class FraudDetectionHybridML(nn.Module):
    """
    Classical fraud‑detection front‑end.

    * Convolutional feature extractor (QuantumNAT style).
    * Fully‑connected projection to 4 latent features.
    * Sampler network producing 2‑class probabilities
      (SamplertQNN inspired).
    """
    def __init__(self) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Latent projection
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

        # Sampler (softmax over 2 outputs)
        self.sampler = nn.Sequential(
            nn.Linear(4, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Probability distribution over 2 classes.
        """
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        logits = self.fc(flat)
        logits = self.norm(logits)
        probs = F.softmax(self.sampler(logits), dim=-1)
        return probs

__all__ = ["FraudDetectionHybridML"]
