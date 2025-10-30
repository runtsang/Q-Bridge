"""Enhanced classical quanvolution for MNIST with patch attention and training utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class PatchAttention(nn.Module):
    """Learnable attention weights applied to each 2×2 patch feature vector."""
    def __init__(self, num_patches: int, patch_dim: int = 4):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(1, num_patches, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weights

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 quanvolution filter with optional patch attention."""
    def __init__(self, use_attention: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = PatchAttention(num_patches=14 * 14)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)  # (batch, 4, 14, 14)
        features = features.view(x.size(0), 4, -1)  # (batch, 4, 196)
        if self.use_attention:
            features = self.attention(features)
        return features.view(x.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    """Hybrid neural network using the quanvolutional filter followed by a linear head."""
    def __init__(self, use_attention: bool = True):
        super().__init__()
        self.qfilter = QuanvolutionFilter(use_attention=use_attention)
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

def train_quanvolution(
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    use_attention: bool = True,
) -> Tuple[float, float]:
    """Quick training loop on MNIST that returns the final training and test accuracy."""
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(root='.', train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.MNIST(root='.', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QuanvolutionClassifier(use_attention=use_attention).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    def evaluate(loader: DataLoader) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=-1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        return correct / total

    train_acc = evaluate(train_loader)
    test_acc = evaluate(test_loader)
    return train_acc, test_acc

__all__ = ["PatchAttention", "QuanvolutionFilter", "QuanvolutionClassifier", "train_quanvolution"]
