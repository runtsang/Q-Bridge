import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionKernelFilter(nn.Module):
    """Classical quanvolutional filter using convolution and RBF kernel with learned prototypes."""
    def __init__(self, gamma: float = 1.0, num_prototypes: int = 10) -> None:
        super().__init__()
        self.gamma = gamma
        self.num_prototypes = num_prototypes
        # Random prototypes to compute kernel features
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, 4))
        # Convolution to extract 2x2 patches
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Extract 2x2 patches via convolution
        patches = self.conv(x)  # shape (batch, 4, 14, 14)
        # Reshape to (batch*14*14, 4)
        patches = patches.permute(0, 2, 3, 1).reshape(-1, 4)
        # Compute RBF kernel between each patch and all prototypes
        diff = patches.unsqueeze(1) - self.prototypes.unsqueeze(0)  # (N, P, 4)
        dist_sq = torch.sum(diff * diff, dim=2)  # (N, P)
        kernel_vals = torch.exp(-self.gamma * dist_sq)  # (N, P)
        # Reshape back to image layout
        kernel_vals = kernel_vals.view(-1, 14, 14, self.num_prototypes)  # (batch, 14, 14, P)
        kernel_vals = kernel_vals.permute(0, 3, 1, 2).contiguous()  # (batch, P, 14, 14)
        # Flatten for classifier
        return kernel_vals.view(kernel_vals.size(0), -1)


class QuanvolutionKernelClassifier(nn.Module):
    """Classifier that uses the QuanvolutionKernelFilter followed by a linear head."""
    def __init__(self, gamma: float = 1.0, num_prototypes: int = 10, num_classes: int = 10) -> None:
        super().__init__()
        self.filter = QuanvolutionKernelFilter(gamma, num_prototypes)
        self.linear = nn.Linear(num_prototypes * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.filter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionKernelFilter", "QuanvolutionKernelClassifier"]
