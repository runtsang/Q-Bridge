import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvFilter(nn.Module):
    """Classical 2‑D filter mimicking a quantum quanvolution layer."""
    def __init__(self, kernel_size=2, threshold=0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data):
        t = torch.as_tensor(data, dtype=torch.float32)
        t = t.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(t)
        act = torch.sigmoid(logits - self.threshold)
        return act.mean().item()

class RBFKernel(nn.Module):
    """Radial basis function kernel with learnable centre."""
    def __init__(self, dim, gamma=1.0):
        super().__init__()
        self.gamma = gamma
        self.centroid = nn.Parameter(torch.randn(dim))

    def forward(self, x):
        diff = x - self.centroid
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class QuantumNATModel(nn.Module):
    """Hybrid classical model that combines CNN, RBF kernel, and a convolutional filter."""
    def __init__(self, gamma=1.0, conv_kernel=2, conv_threshold=0.0):
        super().__init__()
        self.conv_filter = ConvFilter(kernel_size=conv_kernel, threshold=conv_threshold)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        feature_dim = 16 * 7 * 7
        self.kernel = RBFKernel(feature_dim, gamma)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x):
        bsz = x.shape[0]
        # Compute convolutional filter features per patch
        patches = x.unfold(2, self.conv_filter.kernel_size, 1).unfold(3, self.conv_filter.kernel_size, 1)
        patches = patches.contiguous().view(bsz, -1, self.conv_filter.kernel_size, self.conv_filter.kernel_size)
        conv_val = patches.mean(dim=(1,2,3)).unsqueeze(-1)          # (bsz,1)
        # CNN features
        feat = self.cnn(x).view(bsz, -1)
        # Kernel similarity
        k = self.kernel(feat).squeeze(-1)
        # Concatenate
        combined = torch.cat([feat, k.unsqueeze(-1), conv_val], dim=1)
        out = self.fc(combined)
        return self.norm(out)

__all__ = ["QuantumNATModel"]
