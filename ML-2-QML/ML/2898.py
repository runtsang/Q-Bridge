import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSamplerQNN(nn.Module):
    """
    Classical hybrid model that emulates a quantum sampler using a
    convolution‑style feature extractor followed by fully connected layers.
    The architecture mirrors the QCNNModel while keeping the sampler
    interface identical to the original SamplerQNN class.
    """

    def __init__(self) -> None:
        super().__init__()
        # Feature mapping (2 → 4)
        self.feature = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh()
        )
        # QCNN‑style layers
        self.conv1 = nn.Sequential(nn.Linear(4, 8), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(8, 6), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(6, 4), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(4, 3), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(3, 3), nn.Tanh())
        self.head = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass identical to the original SamplerQNN but enriched
        with additional convolution‑like transformations.
        """
        x = self.feature(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

def HybridSamplerQNN_factory() -> HybridSamplerQNN:
    """
    Factory function to keep backward compatibility with the original
    SamplerQNN interface.
    """
    return HybridSamplerQNN()

__all__ = ["HybridSamplerQNN", "HybridSamplerQNN_factory"]
