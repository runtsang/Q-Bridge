import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQuanvolutionClassifier(nn.Module):
    """
    Classical hybrid classifier that replaces the original quanvolution filter
    with a simple 2x2 stride‑2 convolution and a linear head.
    The design keeps the same feature dimensionality (4×14×14) as the
    quantum baseline, enabling direct comparison of training dynamics.
    """
    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        # 1‑channel input → 4‑channel feature map
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # Linear classifier matching the quantum filter output size
        self.fc = nn.Linear(4 * 14 * 14, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: convolution → flatten → linear → log‑softmax.
        """
        features = self.conv(x)
        flat = features.view(features.size(0), -1)
        logits = self.fc(flat)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionClassifier"]
