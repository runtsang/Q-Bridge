import torch
import torch.nn as nn
import torch.nn.functional as F
from.quantum_kernel import QuantumKernel

class QuanvolutionHybridFilter(nn.Module):
    """Hybrid filter that applies a 2×2 classical convolution followed by a quantum kernel on each patch."""
    def __init__(self) -> None:
        super().__init__()
        # Classical convolution to reduce spatial resolution
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # Quantum kernel that maps each 4‑dim patch to a 4‑dim measurement
        self.qkernel = QuantumKernel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Flattened quantum feature vector of shape (batch, 196*4).
        """
        # Apply classical convolution
        conv_out = self.conv(x)  # shape: (batch, 4, 14, 14)
        # Reshape into patches: (batch, 196, 4)
        batch, ch, h, w = conv_out.shape
        patches = conv_out.permute(0, 2, 3, 1).reshape(batch, -1, 4)
        # Apply quantum kernel to each patch
        q_out = self.qkernel(patches)  # shape: (batch, 196, 4)
        # Flatten all patches into a single feature vector
        return q_out.reshape(batch, -1)

class QuanvolutionHybridClassifier(nn.Module):
    """Classifier that uses the hybrid filter followed by a linear head."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionHybridFilter()
        self.linear = nn.Linear(196 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybridFilter", "QuanvolutionHybridClassifier"]
