"""Unified classical quanvolution module with optional RBF kernel enhancement.

This module extends the original quanvolution implementation by adding
a radial‑basis‑function (RBF) kernel applied to each 2×2 patch.
The kernel is parameterised by a learnable gamma, which allows the
model to adapt the similarity measure during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RBFKernel(nn.Module):
    """Learnable RBF kernel for patch similarity."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class QuanvolutionFilter(nn.Module):
    """Classical 2×2 patch convolution followed by an optional RBF transform.

    The filter first extracts non‑overlapping 2×2 patches from a 28×28 image
    using a single‑channel 2×2 convolution.  The resulting 14×14 patch
    representation can be optionally passed through an RBF kernel to
    emphasise local similarity.
    """

    def __init__(self, use_kernel: bool = False, gamma: float = 1.0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)
        self.use_kernel = use_kernel
        if use_kernel:
            self.kernel = RBFKernel(gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        patches = self.conv(x)          # (B, 4, 14, 14)
        patches = patches.view(x.size(0), 4, -1)  # (B, 4, 14*14)
        if self.use_kernel:
            # apply kernel pairwise between each patch and itself
            # resulting feature dimension expands to 4*14*14
            B, C, N = patches.shape
            patches = patches.permute(0, 2, 1)  # (B, N, C)
            feats = torch.zeros(B, N, C, device=x.device)
            for i in range(N):
                feats[:, i, :] = self.kernel(patches[:, i, :], patches[:, i, :])
            patches = feats.view(B, -1)  # (B, 4*14*14)
        else:
            patches = patches.reshape(x.size(0), -1)
        return patches


class QuanvolutionClassifier(nn.Module):
    """Simple linear classifier on the quanvolution feature map."""

    def __init__(self, use_kernel: bool = False, gamma: float = 1.0) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter(use_kernel, gamma)
        self.fc = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        logits = self.fc(features)
        return F.log_softmax(logits, dim=-1)


class HybridQCNet(nn.Module):
    """CNN followed by a dense head – a classical analogue to the quantum QCNet."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["RBFKernel", "QuanvolutionFilter", "QuanvolutionClassifier", "HybridQCNet"]
