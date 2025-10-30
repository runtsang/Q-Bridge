import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridQuanvolutionSamplerClassifier(nn.Module):
    """Hybrid classifier: classical quanvolution filter + classical sampler network."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.filter = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sampler = nn.Sequential(
            nn.Linear(4, 4),
            nn.Tanh(),
            nn.Linear(4, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        features = self.pool(features).view(x.size(0), -1)
        logits = self.sampler(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionSamplerClassifier"]
