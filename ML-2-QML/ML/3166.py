import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSamplerQNN(nn.Module):
    """
    Hybrid classical sampler network.
    Combines a lightweight CNN encoder (inspired by Quantum‑NAT's QFCModel)
    with a fully connected head that outputs 4 logits.  The logits can be
    interpreted as parameters for the quantum sampler defined in the
    corresponding QML module.  The network is fully differentiable and
    can be trained end‑to‑end with any standard PyTorch loss.
    """
    def __init__(self) -> None:
        super().__init__()
        # Feature extractor: two conv layers + ReLU + MaxPool
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),  # 4 logits for 4 output classes
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid sampler.
        Args:
            x: Input image tensor of shape (batch, 1, 28, 28) or similar.
        Returns:
            Tensor of shape (batch, 4) containing logits that can be
            passed to a softmax or to the quantum sampler.
        """
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        logits = self.fc(flattened)
        return self.norm(logits)

__all__ = ["HybridSamplerQNN"]
