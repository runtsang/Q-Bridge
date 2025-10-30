import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridQuanvolutionClassifier(nn.Module):
    """
    Classical hybrid model that emulates a quanvolutional architecture.
    Combines a 2×2 convolution, a fully‑connected layer (mimicking a quantum
    FCL), and a linear classifier.  The kernel_matrix routine is a
    lightweight RBF implementation inspired by the quantum kernel examples.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10, gamma: float = 1.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=2, stride=2)
        self.fcl = nn.Linear(4 * 14 * 14, 128)
        self.classifier = nn.Linear(128, num_classes)
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)                     # 4×14×14
        x = x.view(x.size(0), -1)            # flatten
        x = torch.tanh(self.fcl(x))          # FCL‑style non‑linearity
        logits = self.classifier(x)
        return F.log_softmax(logits, dim=-1)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """
        Compute an RBF kernel matrix between two batches of feature vectors.
        """
        diff = a[:, None, :] - b[None, :, :]
        dist_sq = (diff ** 2).sum(-1)
        K = torch.exp(-self.gamma * dist_sq)
        return K.detach().cpu().numpy()

__all__ = ["HybridQuanvolutionClassifier"]
