import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvFilter(nn.Module):
    """2×2 convolutional filter implemented with a 1×1 Conv2d and a sigmoid threshold."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv(x) - self.threshold)


class QCNNModel(nn.Module):
    """
    Fully‑connected network that mimics the structure of the quantum QCNN.
    Architecture is adapted to process 4‑dimensional feature vectors
    produced by 2×2 patches.
    """
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(4, 8), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(8, 8), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(8, 6), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(6, 4), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(4, 2), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(2, 2), nn.Tanh())
        self.head = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


class HybridQCNNConvNet(nn.Module):
    """
    Classical hybrid network that integrates a 2‑D convolutional filter,
    a QCNN‑inspired fully‑connected backbone, and a final sigmoid head.
    """
    def __init__(self, kernel_size: int = 2, patch_size: int = 8) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.conv_filter = ConvFilter(kernel_size=kernel_size, threshold=0.0)
        self.qcnn = QCNNModel()
        self.fc = nn.Linear(1, 1)

    def _extract_patches(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Extract overlapping 2×2 patches from a batch of images.
        """
        B, C, H, W = imgs.shape
        patches = imgs.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        # shape: (B, C, H-k+1, W-k+1, k, k)
        patches = patches.contiguous().view(B, C, -1, self.kernel_size, self.kernel_size)
        patches = patches.mean(1)  # collapse channel dimension
        return patches  # (B, num_patches, k, k)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        patches = self._extract_patches(imgs)  # (B, N, k, k)
        B, N, _, _ = patches.shape
        patches = patches.view(B * N, -1)  # (B*N, 4)
        qcnn_out = self.qcnn(patches).view(B, N, 1)  # (B, N, 1)
        out = qcnn_out.mean(1)  # (B, 1)
        prob = torch.sigmoid(self.fc(out))
        return torch.cat([prob, 1 - prob], dim=-1)


__all__ = ["HybridQCNNConvNet"]
