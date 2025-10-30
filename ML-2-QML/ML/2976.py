import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridQCNet(nn.Module):
    """
    Classical hybrid neural network for binary classification.

    Architecture:
        * Convolutional backbone mirroring the original quantum model.
        * Feature vector is compared against a set of learnable prototypes via an
          RBF kernel.  The resulting kernel vector is passed through a linear
          head to produce the final logits.

    The design merges the expressive power of a CNN with the flexibility
    of kernel methods, enabling the model to capture both local visual
    patterns and global non‑linear relationships.
    """
    def __init__(self,
                 num_prototypes: int = 10,
                 gamma: float = 1.0,
                 shift: float = 0.0):
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 84)  # feature dimension for kernel comparison

        # Kernel parameters
        self.num_prototypes = num_prototypes
        self.gamma = gamma
        # Learnable prototype vectors (same dimensionality as fc3 output)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, 84))
        # Linear head after kernel evaluation
        self.kernel_head = nn.Linear(num_prototypes, 1)
        self.shift = shift

    def rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the RBF kernel between two batches of vectors.
        """
        # x: (batch, dim), y: (num, dim)
        diff = x.unsqueeze(1) - y.unsqueeze(0)          # (batch, num, dim)
        sq_norm = torch.sum(diff * diff, dim=-1)        # (batch, num)
        return torch.exp(-self.gamma * sq_norm)         # (batch, num)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Convolutional feature extraction
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                                 # (batch, 84)

        # Compute kernel vector against prototypes
        k = self.rbf_kernel(x, self.prototypes)          # (batch, num_prototypes)

        # Linear transformation to logits
        logits = self.kernel_head(k).squeeze(-1) + self.shift  # (batch,)
        probs = torch.sigmoid(logits)                    # (batch,)
        return torch.stack([probs, 1 - probs], dim=1)

__all__ = ["HybridQCNet"]
