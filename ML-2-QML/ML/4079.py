import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSamplerQNN(nn.Module):
    """
    Hybrid sampler network combining convolutional feature extraction
    (from QuantumNAT) with a lightweight sampler head (from SamplerQNN).
    The network maps a single‑channel image to a 2‑class probability
    distribution.  All layers are fully differentiable and ready for
    joint training with a quantum backend if desired.
    """
    def __init__(self) -> None:
        super().__init__()
        # Feature extractor: 1→8→16→4 channels, 2 pooling stages
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        # Flatten and project to 2 logits
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 7 * 7, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        self.norm = nn.BatchNorm1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28) – e.g. MNIST.
        Returns
        -------
        torch.Tensor
            Softmax probabilities of shape (batch, 2).
        """
        x = self.features(x)
        logits = self.fc(x)
        return F.softmax(self.norm(logits), dim=-1)

__all__ = ["HybridSamplerQNN"]
