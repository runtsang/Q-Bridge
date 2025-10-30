import torch
import torch.nn as nn
import torch.nn.functional as F

class RBFKernel(nn.Module):
    """Radial basis function kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class HybridQuantumNAT(nn.Module):
    """
    Classical hybrid model: CNN backbone → optional RBF kernel augmentation → linear classifier.
    """
    def __init__(self, use_kernel: bool = False, gamma: float = 1.0, num_classes: int = 4):
        super().__init__()
        self.use_kernel = use_kernel
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.norm = nn.BatchNorm1d(64)

        if use_kernel:
            self.kernel = RBFKernel(gamma)
            # Learnable prototype vectors for kernel similarity
            self.prototype = nn.Parameter(torch.randn(10, 64), requires_grad=True)
            self.classifier = nn.Linear(10, num_classes)
        else:
            self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        features = self.conv(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        out = self.norm(out)

        if self.use_kernel:
            # Compute RBF kernel between batch features and prototypes
            k = torch.exp(-self.kernel.gamma * torch.sum((out.unsqueeze(1) - self.prototype.unsqueeze(0)) ** 2, dim=2))
            logits = self.classifier(k)
        else:
            logits = self.classifier(out)
        return logits

__all__ = ["HybridQuantumNAT"]
